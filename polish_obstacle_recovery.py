import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import time
import types
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

BASE_CKPT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")

SYSTEM_CONTRACT = (
    "<system>You are YORI, an on-device executive agent.\n"
    "- For conversation, questions, advice, or chit-chat: answer directly inside <thought> with NO <action>.\n"
    "- For device operations and app workflows: inspect the visible <screen> elements, reason step-by-step in <thought>:\n"
    "  🎯 Goal: [Clarified user objective]\n"
    "  📋 Plan: [Numbered steps to achieve goal]\n"
    "  ⚡ Immediate Action: [Why this specific action is chosen based on visible screen]\n"
    "- Then emit ONE action from:\n"
    "  OPEN_APP(appName)\n"
    "  OPEN_SETTINGS(settingName) // e.g. \"WIFI\", \"BLUETOOTH\", \"DISPLAY\", \"SOUND\", \"BATTERY\", \"APPS\"\n"
    "  CLICK(elementLabel | [index])\n"
    "  TYPE(elementLabel | [index], text)\n"
    "  SCROLL(UP | DOWN | LEFT | RIGHT)\n"
    "  PRESS_KEY(BACK | HOME | ENTER)\n"
    "  TORCH(1 | 0)\n"
    "  VOLUME(UP | DOWN | MUTE)\n"
    "  SET_ALARM(HH:MM)\n"
    "  DONE(summary)\n</system>\n"
)

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=16, lora_alpha=32):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, original_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, r))

    def forward(self, x):
        base = self.original_linear(x)
        lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base + lora.to(base.dtype)

def make_sample(query: str, screen: str, goal: str, plan: str, action_reason: str, tool_call: str):
    prompt = f"{SYSTEM_CONTRACT}<user>{query}</user>\n<screen>\n{screen.strip()}\n</screen>\n<thought>\n"
    target = (
        f"🎯 Goal: {goal.strip()}\n"
        f"📋 Plan: {plan.strip()}\n"
        f"⚡ Immediate Action: {action_reason.strip()}\n"
        f"</thought>\n"
        f"<action>{tool_call.strip()}</action>"
    )
    return {"prompt": prompt, "target": target}

def build_obstacle_alignment_dataset():
    samples = []
    
    # 1. Cookie Overlay Obstacle
    for _ in range(25):
        samples.append(make_sample(
            "Open chrome, navigate to indeed jobs, and apply for AI Engineer position",
            '[SCREEN: com.android.chrome]\n[1] TextView "Cookie Preferences"\n[2] TextView "We use cookies to enhance your experience"\n[3] Button "Accept all cookies"\n[4] Button "Reject all"',
            "Open Chrome and apply for AI Engineer position on Indeed",
            "1. Clear blocking cookie modal 2. Fill application form 3. Submit",
            "A cookie consent overlay is blocking the application form. I will tap 'Accept all cookies' to clear the obstacle.",
            'CLICK("Accept all cookies")'
        ))

    # 2. Permission Modal Obstacle
    for _ in range(25):
        samples.append(make_sample(
            "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp",
            '[SCREEN: com.google.android.permissioncontroller]\n[1] TextView "Allow Gallery to access photos and videos on this device?"\n[2] Button "Allow"\n[3] Button "Don\'t allow"',
            "Find recent photo in Gallery Camera album and share via WhatsApp",
            "1. Grant required storage permission 2. Open Camera album 3. Share photo",
            "A system permission dialog is requesting photo access. I will tap 'Allow' so Gallery can display photos.",
            'CLICK("Allow")'
        ))

    # 3. Off-Screen Scroll Obstacle
    for _ in range(25):
        samples.append(make_sample(
            "Complete checkout and place order",
            '[SCREEN: com.amazon.mShop.android.shopping]\n[1] TextView "Cart (1 item)"\n[2] TextView "Wireless Headphones - $79.99"\n[3] TextView "Shipping Address: 123 Main St"\n[4] TextView "Payment: Visa ending 4021"',
            "Complete checkout and place order",
            "1. Scroll down to reveal Place Order button 2. Click Place Order",
            "The 'Place Your Order' button is below the current visible viewport. I will scroll down to reveal it.",
            'SCROLL("DOWN")'
        ))

    # 4. App Update Dialog Obstacle
    for _ in range(25):
        samples.append(make_sample(
            "Send message on WhatsApp to Sister Maryam",
            '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Update WhatsApp"\n[2] TextView "A new version of WhatsApp is available."\n[3] Button "Later"\n[4] Button "Update"',
            "Send WhatsApp message to Sister Maryam",
            "1. Dismiss update dialog 2. Open chat with Sister Maryam 3. Send message",
            "An update dialog is intercepting the screen. I will dismiss it by tapping 'Later' to proceed with the message.",
            'CLICK("Later")'
        ))

    # 5. Negative Feedback Recovery
    for _ in range(25):
        samples.append(make_sample(
            "Open chatgpt and write a message about HyperGQE",
            '[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[5] View "Ask ChatGPT"\n[6] View "Attachment"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]',
            "Open ChatGPT and compose message about HyperGQE",
            "1. Bypass unresponsive Menu 2. Type directly into Ask ChatGPT 3. Send",
            "Previous click on Menu had no effect. I will adapt spontaneously and type directly into the visible 'Ask ChatGPT' field.",
            'TYPE("Ask ChatGPT", "Explain HyperGQE architecture")'
        ))

    random.shuffle(samples)
    print(f"Total Obstacle Alignment Samples: {len(samples)}", flush=True)
    return samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== OBSTACLE RESOLUTION ALIGNMENT PASS ON {device} ===", flush=True)
    t0 = time.time()

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.float32)
    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=16, lora_alpha=32).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=16, lora_alpha=32).to(device))

    tokenizer = Tokenizer.from_file(TOK_PATH)
    samples = build_obstacle_alignment_dataset()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3.5e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    STEPS = 45
    BATCH_SIZE = 4
    model.train()

    for step in range(1, STEPS + 1):
        batch = random.sample(samples, BATCH_SIZE)
        optimizer.zero_grad()
        loss_total = 0.0

        for item in batch:
            full = item["prompt"] + item["target"]
            full_ids = tokenizer.encode(full).ids
            prompt_ids = tokenizer.encode(item["prompt"]).ids

            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            p_len = len(prompt_ids) - 1
            labels[0, :p_len] = -100

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

            scaler.scale(loss / len(batch)).backward()
            loss_total += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 15 == 0 or step == 1 or step == STEPS:
            avg_loss = loss_total / len(batch)
            print(f"Alignment Step {step:2d}/{STEPS} | Loss: {avg_loss:.4f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    print("Merging alignment LoRA into FP32 base...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for attr in ["q_proj", "v_proj", "o_proj"]:
                mod = getattr(block.attn, attr)
                delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                mod.original_linear.weight.data.add_(delta.to(mod.original_linear.weight.dtype))
                setattr(block.attn, attr, mod.original_linear)
            for attr in ["gate", "up", "down"]:
                mod = getattr(block.ffn, attr)
                delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                mod.original_linear.weight.data.add_(delta.to(mod.original_linear.weight.dtype))
                setattr(block.ffn, attr, mod.original_linear)

    print(f"Saving merged checkpoint to {OUT_PT}...", flush=True)
    torch.save(model.state_dict(), OUT_PT)

    print(f"Serializing to binary EER at {OUT_EER}...", flush=True)
    serialize_checkpoint_to_eer(OUT_PT, OUT_EER, "FP16")
    print(f"[SUCCESS] Obstacle resolution alignment finished in {time.time()-t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
