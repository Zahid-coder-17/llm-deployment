import os
import sys
import time
import types
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

EVAL_DIR = r"D:\embeddra_eval"
sys.path.insert(0, os.path.join(EVAL_DIR, "embeddra-edge"))
sys.path.insert(0, r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\embeddra-edge\export\mobile_export")

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

CKPT_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
OUT_EER_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.eer"
TOK_PATH = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")

CHAT_PAIRS = [
    ("Hello", "Hey! I'm Yori, your local AI assistant by Embeddra AI. How can I help you on your phone today?"),
    ("Hi Yori", "Hi there! I'm running 100% locally on your phone right now. Ask me to open any app, toggle settings, send messages, or just chat!"),
    ("Hey, how are you?", "I'm doing great and running smoothly on your device's ARM NEON processor! What would you like to do?"),
    ("Who are you?", "I'm Yori, an offline executive AI assistant built by Embeddra AI. I can chat locally and control your phone's apps, settings, calls, and messages."),
    ("What can you do?", "I can open any app, toggle your flashlight or volume, check battery, send WhatsApp or SMS messages, make calls, set alarms or timers, search the web, and chat with you completely offline."),
    ("Who built you?", "I was created by Embeddra AI to run privately and in real time directly on mobile and desktop devices."),
    ("Tell me a joke", "Why did the smartphone need glasses? Because it lost all its contacts!"),
    ("Explain artificial intelligence simply", "Artificial intelligence is software designed to recognize patterns, reason about information, and take helpful actions—just like how I understand your requests and control your phone."),
    ("What is edge AI?", "Edge AI means running artificial intelligence models directly on your phone or laptop instead of sending your data to cloud servers. It's faster, private, and works offline."),
    ("Summarize what a transformer model does", "A transformer model uses self-attention to understand relationships between words in a prompt and predicts the next best tokens in sequence."),
    ("Give me 3 productivity tips", "1. Break big tasks into 25-minute focus blocks.\n2. Mute non-essential notifications while working.\n3. Review your top 3 priorities every morning."),
    ("Write a short polite message saying I am busy", "Hi! I'm currently tied up in a meeting, but I'll get back to you as soon as I'm free."),
    ("Help me write a quick reply to my friend", "Hey! Just saw your message—let's catch up a bit later today!"),
    ("Good morning Yori", "Good morning! Hope you have a productive day ahead. Need me to check your alarms or messages?"),
    ("Good night", "Good night! Let me know if you want me to set an alarm for tomorrow morning."),
    ("Thank you", "You're very welcome! Anytime you need help on your phone, just ask."),
    ("Thanks Yori", "Happy to help! Let me know what's next."),
    ("Are you connected to the internet?", "Nope! My neural core runs completely offline on your phone's local hardware for maximum privacy and speed."),
    ("What is 25 multiplied by 4?", "25 multiplied by 4 equals 100."),
    ("Give me a motivational quote", "Small consistent steps every day build extraordinary results over time."),
    ("How do I save battery on Android?", "You can lower screen brightness, turn off Bluetooth or GPS when unused, and enable Battery Saver in Settings."),
    ("Can you control my phone?", "Yes! Just tell me to turn on the flashlight, adjust volume, open any app, call a contact, set an alarm, or navigate any screen."),
]

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 16, lora_alpha: int = 32):
        super().__init__()
        self.original_linear = original_linear
        for p in self.original_linear.parameters():
            p.requires_grad = False
        self.scaling = lora_alpha / r
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, original_linear.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(original_linear.out_features, r, dtype=dtype))
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_linear(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI LOCAL CHAT + EXECUTIVE SFT ON {device} ===", flush=True)
    t0 = time.time()

    model = MariaForCausalLM.from_pretrained_checkpoint(CKPT_PATH, device=device, dtype=torch.bfloat16)
    tokenizer = Tokenizer.from_file(TOK_PATH)

    for p in model.parameters():
        p.requires_grad = False

    lora_layers = []
    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj).to(device)
        block.attn.v_proj = LoRALinear(block.attn.v_proj).to(device)
        lora_layers.extend([block.attn.q_proj, block.attn.v_proj])

    # Build tokenized samples (mask prompt tokens with -100)
    samples = []
    for u, a in CHAT_PAIRS:
        p_str = f"<user>{u}</user>\n<thought>"
        a_str = f"{a}</thought><action>DONE(\"Chat\")</action>"
        p_ids = tokenizer.encode(p_str).ids
        a_ids = tokenizer.encode(a_str).ids
        full_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        samples.append((
            torch.tensor([full_ids[:-1]], dtype=torch.long, device=device),
            torch.tensor([labels[1:]], dtype=torch.long, device=device)
        ))

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    model.train()

    STEPS = 110
    for step in range(1, STEPS + 1):
        x, y = random.choice(samples)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step % 20 == 0 or step == 1:
            print(f"Step {step:3d}/{STEPS} | Loss: {loss.item():.4f} | Elapsed: {time.time() - t0:.1f}s", flush=True)

    # Fuse LoRA back into original linear weights
    print("\nFusing Chat LoRA into base weights...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for attr in ["q_proj", "v_proj"]:
                layer = getattr(block.attn, attr)
                if isinstance(layer, LoRALinear):
                    delta = (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
                    fused = (layer.original_linear.weight.float() + delta).to(layer.original_linear.weight.dtype)
                    layer.original_linear.weight.copy_(fused)
                    setattr(block.attn, attr, layer.original_linear)

    print("Saving updated fused checkpoint...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k in sd:
            sd[k] = v.cpu().to(sd[k].dtype)
    torch.save(ckpt, CKPT_PATH)

    print("Serializing updated .eer binary...", flush=True)
    serialize_checkpoint_to_eer(CKPT_PATH, OUT_EER_PATH, target_dtype="fp16")
    print(f"ALL DONE in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()

