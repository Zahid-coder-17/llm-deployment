import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import math
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

SYSTEM_PROMPT = (
    "<system>You are YORI, an on-device Android executive agent.\n"
    "- For conversation, questions, advice, or chit-chat: answer directly inside <thought> with NO <action>.\n"
    "- For device operations and app workflows: inspect the visible <screen> elements, reason step-by-step in <thought>, and emit ONE action from:\n"
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

def build_exploration_samples():
    random.seed(42)
    samples = []

    # 1. ChatGPT Multi-Step Exploration
    cg_goal = "Open chatgpt and write a message about HyperGQE"
    for _ in range(40):
        # Step 1: Open
        samples.append({
            "prompt": f"{SYSTEM_PROMPT}<user>{cg_goal}</user>\n<thought>",
            "target": 'User wants to use ChatGPT to ask about this topic. I will launch ChatGPT first.</thought>\n<action>OPEN_APP("ChatGPT")</action>'
        })
        # Step 2: See Ask ChatGPT
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{cg_goal}</user>\n<screen>\n[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[4] View "Temporary chat"\n[5] View "Ask ChatGPT"\n[8] View "Attachment"\n[10] View "Dictation"\n[12] View "Start a voice conversation"\n</screen>\n<thought>',
            "target": 'ChatGPT is open. I see the visible \'Ask ChatGPT\' text input area on screen. I will type the prompt.</thought>\n<action>TYPE("Ask ChatGPT", "Explain HyperGQE architecture")</action>'
        })
        # Step 3: Send
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{cg_goal}</user>\n<screen>\n[SCREEN: com.openai.chatgpt]\n[5] EditText "Explain HyperGQE architecture"\n[6] Button "Send message"\n</screen>\n<thought>',
            "target": 'The query is typed into the input field. I will tap Send message to submit it.</thought>\n<action>CLICK("Send message")</action>'
        })
        # Negative feedback recovery
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{cg_goal}</user>\n<screen>\n[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[5] View "Ask ChatGPT"\n[6] View "Attachment"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]\n</screen>\n<thought>',
            "target": 'Previous action on Menu had no effect. I will type directly into Ask ChatGPT instead.</thought>\n<action>TYPE("Ask ChatGPT", "Explain HyperGQE architecture")</action>'
        })

    # 2. YouTube Multi-Step Exploration
    yt_goal = "find Agentic AI lecture and play it from youtube"
    for _ in range(35):
        samples.append({
            "prompt": f"{SYSTEM_PROMPT}<user>{yt_goal}</user>\n<thought>",
            "target": 'User wants to watch a video on YouTube. I will launch YouTube first.</thought>\n<action>OPEN_APP("YouTube")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{yt_goal}</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] TextView "YouTube"\n[2] Button "Notifications"\n[3] Button "Search"\n[4] TextView "Trending Video"\n[5] Tab "Home"\n</screen>\n<thought>',
            "target": 'YouTube is open. I see the Search button. I will tap Search to find the lecture.</thought>\n<action>CLICK("Search")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{yt_goal}</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] ImageButton "Navigate up"\n[2] EditText "Search YouTube" [Input]\n[3] Button "Voice search"\n</screen>\n<thought>',
            "target": 'I see the Search YouTube input bar. I will type the requested query.</thought>\n<action>TYPE("Search YouTube", "Agentic AI lecture")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{yt_goal}</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] EditText "Agentic AI lecture" [Input]\n[2] Button "Filters"\n[3] TextView "Agentic AI Lecture - Full Course"\n[4] TextView "Random Vlog"\n</screen>\n<thought>',
            "target": 'Search results loaded. I see Agentic AI Lecture - Full Course. I will tap the video to play it.</thought>\n<action>CLICK("Agentic AI Lecture - Full Course")</action>'
        })

    # 3. WhatsApp Multi-Step Exploration
    wa_goal = "Send Message to Sister Maryam Bill Gates saying \"Hi ,I'm yori an ai agent built on Maria\""
    for _ in range(35):
        samples.append({
            "prompt": f"{SYSTEM_PROMPT}<user>{wa_goal}</user>\n<thought>",
            "target": 'User wants to send a WhatsApp message to Sister Maryam. I must open WhatsApp first.</thought>\n<action>OPEN_APP("WhatsApp")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{wa_goal}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] Button "Search…"\n[2] TextView "Sister Maryam"\n[3] TextView "Recent message"\n</screen>\n<thought>',
            "target": 'I see Sister Maryam in the chat list. I will tap the contact.</thought>\n<action>CLICK("Sister Maryam")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{wa_goal}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] TextView "Sister Maryam"\n[2] EditText "Message"\n[3] Button "Voice message"\n</screen>\n<thought>',
            "target": 'In chat with Sister Maryam. I will enter the requested message into the Message field.</thought>\n<action>TYPE("Message", "Hi ,I\'m yori an ai agent built on Maria")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{wa_goal}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] EditText "Hi ,I\'m yori an ai agent built on Maria"\n[2] Button "Send"\n</screen>\n<thought>',
            "target": 'The message is typed into the composer. I will tap Send to deliver it.</thought>\n<action>CLICK("Send")</action>'
        })

    # 4. Settings, System & Advice Presets
    core_extras = [
        ("Open Wi-Fi Settings", "User wants to open Wi-Fi settings.", 'OPEN_SETTINGS("WIFI")'),
        ("Turn on flashlight", "User wants to turn on the flashlight.", 'TORCH(1)'),
        ("Check my alarms on this phone", "User wants to view device alarms. I will open the Clock app.", 'OPEN_APP("Clock")'),
        ("3 advices pls", "Here are 3 valuable pieces of advice:\n1. Focus on high-impact priorities early in the day.\n2. Build consistency through small, sustainable daily habits.\n3. Reflect regularly and iterate continuously based on feedback.", None),
    ]
    for q, t, a in core_extras:
        for _ in range(25):
            if a:
                samples.append({
                    "prompt": f"{SYSTEM_PROMPT}<user>{q}</user>\n<thought>",
                    "target": f"{t}</thought>\n<action>{a}</action>"
                })
            else:
                samples.append({
                    "prompt": f"{SYSTEM_PROMPT}<user>{q}</user>\n<thought>",
                    "target": f"{t}</thought>"
                })

    random.shuffle(samples)
    print(f"Total environment exploration training samples: {len(samples)}", flush=True)
    return samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING ENVIRONMENT EXPLORATION ALIGNMENT ON {device} ===", flush=True)
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
    samples = build_exploration_samples()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2.5e-5, weight_decay=0.01)

    STEPS = 120
    for step in range(1, STEPS + 1):
        batch = random.sample(samples, 4)
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

            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            (loss / len(batch)).backward()
            loss_total += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 15 == 0 or step == 1 or step == STEPS:
            print(f"Exploration Step {step:3d}/{STEPS} | Loss: {loss_total/len(batch):.4f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    print("Merging exploration LoRA into FP32...", flush=True)
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

    print(f"Saving merged model to {OUT_PT}...", flush=True)
    torch.save(model.state_dict(), OUT_PT)

    print(f"Serializing to {OUT_EER}...", flush=True)
    serialize_checkpoint_to_eer(OUT_PT, OUT_EER, "FP16")
    print("[OK] Exploration alignment complete!", flush=True)

if __name__ == "__main__":
    main()
