import os
import sys
import json
import time
import types
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests
from tokenizers import Tokenizer

EVAL_DIR = r"D:\embeddra_eval"
sys.path.insert(0, os.path.join(EVAL_DIR, "embeddra-edge"))
sys.path.insert(0, r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\embeddra-edge\export\mobile_export")

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

# Start from the clean 20B base or merged checkpoint; using merged keeps UI skills while adding full chat fluency
CKPT_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
OUT_EER_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.eer"
TOK_PATH = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")
UI_DATA_PATH = os.path.join(EVAL_DIR, "android_agent", "train_reasoning_agent.jsonl")

YORI_IDENTITY_PAIRS = [
    ("Hi", "Hey! I'm Yori, your local AI assistant by Embeddra AI. How can I help you today?"),
    ("Hello", "Hello! I'm Yori, running completely offline on your phone. What would you like to talk about or get done?"),
    ("Hey Yori", "Hey there! Ready whenever you are—ask me anything or tell me what to control on your phone."),
    ("Who are you?", "I'm Yori, an offline executive AI assistant built by Embeddra AI. I run locally on your device to chat and control your phone."),
    ("Who built you?", "I was created by Embeddra AI to run natively and privately on smartphones and laptops without cloud servers."),
    ("What can you do?", "I can chat offline, answer questions, summarize text, write messages, and control your phone—opening apps, toggling flashlight or volume, making calls, and setting alarms."),
    ("Are you connected to the internet?", "No, I run 100% offline directly on your device hardware so your conversations stay completely private."),
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
    print(f"=== YORI FULL INSTRUCTION + LOCAL CHAT SFT ON {device} ===", flush=True)
    t0 = time.time()

    tokenizer = Tokenizer.from_file(TOK_PATH)

    hf_cache = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache"
    os.makedirs(hf_cache, exist_ok=True)
    alpaca_path = os.path.join(hf_cache, "alpaca_data_cleaned.json")
    if not os.path.exists(alpaca_path):
        print("1. Downloading yahma/alpaca-cleaned JSON...", flush=True)
        resp = requests.get("https://huggingface.co/datasets/yahma/alpaca-cleaned/resolve/main/alpaca_data_cleaned.json", timeout=30)
        resp.raise_for_status()
        with open(alpaca_path, "wb") as f:
            f.write(resp.content)
    with open(alpaca_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    print(f"Loaded {len(ds)} raw Alpaca cleaned records.", flush=True)
    chat_samples = []

    # Add Yori identity samples (weighted)
    for _ in range(8):
        for u, a in YORI_IDENTITY_PAIRS:
            p_ids = tokenizer.encode(f"<user>{u}</user>\n<thought>").ids
            a_ids = tokenizer.encode(f"{a}</thought>").ids
            full_ids = p_ids + a_ids
            labels = [-100] * len(p_ids) + a_ids
            chat_samples.append((full_ids, labels))

    # Add 1,600 crisp, high-quality instruction/chat samples
    count = 0
    for row in ds:
        instr = row["instruction"].strip()
        inp = row["input"].strip()
        out = row["output"].strip()
        if not instr or not out or len(out) > 320 or len(out) < 15:
            continue
        user_msg = f"{instr}\n{inp}".strip() if inp else instr
        p_ids = tokenizer.encode(f"<user>{user_msg}</user>\n<thought>").ids
        a_ids = tokenizer.encode(f"{out}</thought>").ids
        full_ids = p_ids + a_ids
        if len(full_ids) <= 256:
            labels = [-100] * len(p_ids) + a_ids
            chat_samples.append((full_ids, labels))
            count += 1
        if count >= 1600:
            break

    ui_samples = []
    if os.path.exists(UI_DATA_PATH):
        with open(UI_DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 120 or not line.strip():
                    break
                traj = json.loads(line)
                intent = traj["user_intent"]
                for turn in traj["turns"][:1]:
                    p_str = f"<user>{intent}</user>\n<screen>\n{turn['screen']}\n</screen>\n<thought>"
                    a_str = f"{turn['thought']}</thought>\n<action>{turn['action']}</action>"
                    p_ids = tokenizer.encode(p_str).ids
                    a_ids = tokenizer.encode(a_str).ids
                    full_ids = p_ids + a_ids
                    if len(full_ids) <= 256:
                        labels = [-100] * len(p_ids) + a_ids
                        ui_samples.append((full_ids, labels))

    print(f"2. Dataset ready: {len(chat_samples)} chat/instruction samples + {len(ui_samples)} UI replay samples.", flush=True)

    print("3. Loading Maria-1B checkpoint...", flush=True)
    model = MariaForCausalLM.from_pretrained_checkpoint(CKPT_PATH, device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, r=16, lora_alpha=32).to(device)
        block.attn.k_proj = LoRALinear(block.attn.k_proj, r=16, lora_alpha=32).to(device)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, r=16, lora_alpha=32).to(device)
        block.attn.o_proj = LoRALinear(block.attn.o_proj, r=16, lora_alpha=32).to(device)
        block.ffn.gate = LoRALinear(block.ffn.gate, r=16, lora_alpha=32).to(device)
        block.ffn.up = LoRALinear(block.ffn.up, r=16, lora_alpha=32).to(device)
        block.ffn.down = LoRALinear(block.ffn.down, r=16, lora_alpha=32).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable LoRA params: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M", flush=True)

    OPT_STEPS = 250
    GRAD_ACCUM = 4  # 1,000 total examples seen!
    optimizer = torch.optim.AdamW(trainable_params, lr=1.5e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=OPT_STEPS, eta_min=1e-5)

    model.train()
    for step in range(1, OPT_STEPS + 1):
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            if ui_samples and random.random() < 0.15:
                full_ids, labels = random.choice(ui_samples)
            else:
                full_ids, labels = random.choice(chat_samples)

            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100) / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == 1:
            print(f"Step {step:3d}/{OPT_STEPS} ({step*GRAD_ACCUM} ex) | Loss: {step_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Elapsed: {time.time() - t0:.1f}s", flush=True)

    # Live Validation on unseen intelligent prompts
    model.eval()
    print("\n--- LIVE VALIDATION ON UNSEEN CHAT PROMPTS ---", flush=True)
    test_prompts = [
        "Hey Yori, who built you?",
        "Explain quantum computing in simple terms.",
        "Give me 3 tips for staying focused while studying.",
        "Write a polite message declining a meeting tomorrow.",
        "Why do stars twinkle at night?"
    ]
    for q in test_prompts:
        p_ids = tokenizer.encode(f"<user>{q}</user>\n<thought>").ids
        curr = torch.tensor([p_ids], dtype=torch.long, device=device)
        gen = []
        with torch.no_grad():
            for _ in range(60):
                logits, _ = model(curr)
                # Slight repetition penalty for crisp validation
                step_logits = logits[0, -1, :].clone().float()
                for prev in set(gen):
                    if step_logits[prev] > 0: step_logits[prev] /= 1.2
                    else: step_logits[prev] *= 1.2
                t = step_logits.argmax().item()
                gen.append(t)
                curr = torch.cat([curr, torch.tensor([[t]], device=device)], dim=1)
                if "</thought>" in tokenizer.decode(gen):
                    break
        print(f"Q: {q}\nA: {tokenizer.decode(gen).replace('</thought>', '').strip()}\n", flush=True)

    # Fuse LoRA into base weights
    print("Fusing Q/K/V/O + FFN LoRA weights...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for mod, attrs in [(block.attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (block.ffn, ["gate", "up", "down"])]:
                for attr in attrs:
                    layer = getattr(mod, attr)
                    if isinstance(layer, LoRALinear):
                        delta = (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
                        fused = (layer.original_linear.weight.float() + delta).to(layer.original_linear.weight.dtype)
                        layer.original_linear.weight.copy_(fused)
                        setattr(mod, attr, layer.original_linear)

    print("Saving updated checkpoint...", flush=True)
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
