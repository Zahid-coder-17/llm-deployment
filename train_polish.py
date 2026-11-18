import os
import sys
import json
import math
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

BASE_CKPT = os.path.join(EVAL_DIR, "checkpoints", "maria_1b_20B_weights.pt")
LORA_V4_BEST = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\lora_v4_best.pt"
OUT_LORA_POLISH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\lora_polish_best.pt"
OUT_PT = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
OUT_EER = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.eer"
TOK_PATH = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")

ALPACA_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\alpaca_data_cleaned.json"
UNIFIED_TRAIN = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_unified_train.jsonl"
VAL_JSONL = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_unified_val.jsonl"
POLISH_TRAIN = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_polish_train.jsonl"

EDGE_CASES = [
    {"prompt": "<user>Search Google for Python asyncio tutorial</user>\n<thought>", "completion": "User wants to search Google for 'Python asyncio tutorial'.</thought>\n<action>SEARCH(\"Python asyncio tutorial\")</action>"},
    {"prompt": "<user>Search Google for best coffee shops nearby</user>\n<thought>", "completion": "User wants to search Google for 'best coffee shops nearby'.</thought>\n<action>SEARCH(\"best coffee shops nearby\")</action>"},
    {"prompt": "<user>Search on Google for weather in Hyderabad</user>\n<thought>", "completion": "User wants to search Google for 'weather in Hyderabad'.</thought>\n<action>SEARCH(\"weather in Hyderabad\")</action>"},
    {"prompt": "<user>Look up Docker compose networking on Google</user>\n<thought>", "completion": "User wants to search Google for 'Docker compose networking'.</thought>\n<action>SEARCH(\"Docker compose networking\")</action>"},
    {"prompt": "<user>Search Google about how transformers work</user>\n<thought>", "completion": "User wants to search Google for 'how transformers work'.</thought>\n<action>SEARCH(\"how transformers work\")</action>"},
    {"prompt": "<user>Find on Google flights from Bangalore to Delhi</user>\n<thought>", "completion": "User wants to search Google for 'flights from Bangalore to Delhi'.</thought>\n<action>SEARCH(\"flights from Bangalore to Delhi\")</action>"},
    {"prompt": "<user>Google search React useEffect cleanup</user>\n<thought>", "completion": "User wants to search Google for 'React useEffect cleanup'.</thought>\n<action>SEARCH(\"React useEffect cleanup\")</action>"},
    {"prompt": "<user>Can you search Google for current gold rate</user>\n<thought>", "completion": "User wants to search Google for 'current gold rate'.</thought>\n<action>SEARCH(\"current gold rate\")</action>"},
    {"prompt": "<user>Take phone off mute</user>\n<thought>", "completion": "User wants to unmute the device volume.</thought>\n<action>VOLUME(\"UNMUTE\")</action>"},
    {"prompt": "<user>Take my phone off silent</user>\n<thought>", "completion": "User wants to unmute the device volume.</thought>\n<action>VOLUME(\"UNMUTE\")</action>"},
    {"prompt": "<user>Unmute the phone</user>\n<thought>", "completion": "User wants to unmute the device volume.</thought>\n<action>VOLUME(\"UNMUTE\")</action>"},
    {"prompt": "<user>Turn sound back on for my phone</user>\n<thought>", "completion": "User wants to unmute the device volume.</thought>\n<action>VOLUME(\"UNMUTE\")</action>"},
    {"prompt": "<user>Put my phone on mute</user>\n<thought>", "completion": "User wants to mute the device volume.</thought>\n<action>VOLUME(\"MUTE\")</action>"},
    {"prompt": "<user>Mute the phone right now</user>\n<thought>", "completion": "User wants to mute the device volume.</thought>\n<action>VOLUME(\"MUTE\")</action>"},
    {"prompt": "<user>Silence my phone</user>\n<thought>", "completion": "User wants to mute the device volume.</thought>\n<action>VOLUME(\"MUTE\")</action>"},
    {"prompt": "<user>Turn phone volume up</user>\n<thought>", "completion": "User wants to increase the device volume.</thought>\n<action>VOLUME(\"UP\")</action>"},
    {"prompt": "<user>Lower phone volume</user>\n<thought>", "completion": "User wants to decrease the device volume.</thought>\n<action>VOLUME(\"DOWN\")</action>"},
]

def build_polish_dataset():
    random.seed(123)
    # 1. 55% Clean short chat (~800 examples, drop long Dolly passages)
    with open(ALPACA_PATH, "r", encoding="utf-8") as f:
        alpaca = json.load(f)
    short_chat = []
    for row in alpaca:
        instr = row["instruction"].strip()
        inp = row["input"].strip()
        out = row["output"].strip()
        if not instr or not out or len(out) < 15 or len(out) > 220 or len(instr) > 160:
            continue
        user_msg = f"{instr}\n{inp}".strip() if inp else instr
        short_chat.append({
            "category": "chat",
            "prompt": f"<user>{user_msg}</user>\n<thought>",
            "target": f"{out}</thought>"
        })
    random.shuffle(short_chat)
    chat_slice = short_chat[:760]
    yori_pairs = [
        ("Hi", "Hey! I'm Yori, your local AI assistant by Embeddra AI. How can I help you today?"),
        ("Who are you?", "I'm Yori, an offline AI executive assistant built by Embeddra AI."),
        ("Who built you?", "I was created by Embeddra AI to run locally on your phone."),
        ("What can you do?", "I can chat offline, answer questions, and control your phone's apps, settings, calls, and messages."),
        ("Explain what a flashlight is", "A flashlight is a portable hand-held electric light powered by batteries."),
    ]
    for _ in range(8):
        for u, a in yori_pairs:
            chat_slice.append({"category": "chat", "prompt": f"<user>{u}</user>\n<thought>", "target": f"{a}</thought>"})

    # 2. Load tool_call & ui_grounding from unified train
    orig_tools, orig_ui = [], []
    with open(UNIFIED_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if obj["category"] == "tool_call":
                orig_tools.append(obj)
            elif obj["category"] == "ui_grounding":
                orig_ui.append(obj)

    random.shuffle(orig_tools)
    random.shuffle(orig_ui)

    # 25% Hard tool_call fixes (~350 examples: 4x edge cases = 68 + 282 diverse tool_call)
    tool_slice = []
    for _ in range(4):
        for ec in EDGE_CASES:
            tool_slice.append({"category": "tool_call", "prompt": ec["prompt"], "target": ec["completion"]})
    tool_slice.extend(orig_tools[:282])

    # 20% ui_grounding replay (~300 examples)
    ui_slice = orig_ui[:300]

    polish_all = chat_slice + tool_slice + ui_slice
    random.shuffle(polish_all)
    with open(POLISH_TRAIN, "w", encoding="utf-8") as f:
        for ex in polish_all:
            f.write(json.dumps(ex) + "\n")

    n = len(polish_all)
    print(f"Built {POLISH_TRAIN}: {n} examples (chat={len(chat_slice)} [{100*len(chat_slice)/n:.1f}%], tool={len(tool_slice)} [{100*len(tool_slice)/n:.1f}%], ui={len(ui_slice)} [{100*len(ui_slice)/n:.1f}%])", flush=True)

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 32, lora_alpha: int = 64):
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

def load_tokenized(path, tokenizer, max_len=280):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            p_ids = tokenizer.encode(obj["prompt"]).ids
            t_ids = tokenizer.encode(obj["target"]).ids
            full_ids = p_ids + t_ids
            if 4 <= len(full_ids) <= max_len:
                labels = [-100] * len(p_ids) + t_ids
                items.append((obj["category"], full_ids, labels, obj["prompt"], obj["target"]))
    return items

def eval_val_ppl(model, val_items, device):
    model.eval()
    stats = {}
    with torch.no_grad():
        for cat, full_ids, labels, _, _ in val_items:
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100, reduction="sum").item()
            ntoks = (y.view(-1) != -100).sum().item()
            if cat not in stats: stats[cat] = [0.0, 0]
            stats[cat][0] += nll
            stats[cat][1] += ntoks
    return {c: (s[0]/max(1,s[1]), math.exp(min(s[0]/max(1,s[1]), 20.0)), s[1]) for c, s in stats.items()}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI 150-STEP TARGETED POLISH PASS ON {device} ===", flush=True)
    t0 = time.time()

    build_polish_dataset()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    train_items = load_tokenized(POLISH_TRAIN, tokenizer)
    val_items = load_tokenized(VAL_JSONL, tokenizer)

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=32, lora_alpha=64).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=32, lora_alpha=64).to(device))

    lora_ckpt = torch.load(LORA_V4_BEST, map_location=device, weights_only=False)
    model.load_state_dict(lora_ckpt, strict=False)
    print("Loaded Step 300 LoRA adapter (lora_v4_best.pt) cleanly.", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    MAX_STEPS = 150
    WARMUP_STEPS = 15
    PEAK_LR = 2e-5
    MIN_LR = 5e-6
    GRAD_ACCUM = 8
    EVAL_EVERY = 50

    optimizer = torch.optim.AdamW(trainable_params, lr=PEAK_LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        prog = float(step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return (MIN_LR / PEAK_LR) + (1.0 - (MIN_LR / PEAK_LR)) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pre = eval_val_ppl(model, val_items, device)
    best_chat_ppl = pre["chat"][1]
    best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
    print(f"  [VAL @ Step 0 (Resume)] chat PPL={pre['chat'][1]:.2f} | tool_call PPL={pre['tool_call'][1]:.2f} | ui_grounding PPL={pre['ui_grounding'][1]:.2f}", flush=True)

    idx_pool = list(range(len(train_items)))
    random.shuffle(idx_pool)
    ptr = 0

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            if ptr >= len(idx_pool):
                random.shuffle(idx_pool)
                ptr = 0
            _, full_ids, labels, _, _ = train_items[idx_pool[ptr]]
            ptr += 1
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100) / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % EVAL_EVERY == 0 or step == 1:
            print(f"Step {step:3d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | LR={scheduler.get_last_lr()[0]:.7f} | Elapsed={time.time() - t0:.1f}s", flush=True)

        if step % EVAL_EVERY == 0:
            val = eval_val_ppl(model, val_items, device)
            c_ppl, t_ppl, u_ppl = val["chat"][1], val["tool_call"][1], val["ui_grounding"][1]
            agentic_safe = (t_ppl <= 1.50) and (1.14 <= u_ppl <= 1.50)
            saved_tag = ""
            if agentic_safe and c_ppl < best_chat_ppl:
                best_chat_ppl = c_ppl
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA_POLISH)
                saved_tag = " [SAVED BEST]"
            print(f"  [VAL @ Step {step}] chat PPL={c_ppl:.2f} | tool_call PPL={t_ppl:.2f} | ui_grounding PPL={u_ppl:.2f}{saved_tag}", flush=True)

    model.load_state_dict(best_lora_state, strict=False)

    # Pre-Fusion Probe on the 8 unseen val prompts + the 2 edge cases
    model.eval()
    print("\n--- PRE-FUSION PROBE (UNSEEN VAL + EDGE CASES) ---", flush=True)
    probe_prompts = [x[3] for x in val_items if x[0] == "tool_call"][:8]
    probe_expected = [x[4].split("<action>")[1].split("</action>")[0] for x in val_items if x[0] == "tool_call"][:8]
    with torch.no_grad():
        for p_str, exp_act in zip(probe_prompts, probe_expected):
            p_ids = tokenizer.encode(p_str).ids
            curr = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = []
            for _ in range(45):
                logits, _ = model(curr)
                t = logits[0, -1, :].argmax().item()
                gen.append(t)
                curr = torch.cat([curr, torch.tensor([[t]], device=device)], dim=1)
                if "</action>" in tokenizer.decode(gen): break
            pred = tokenizer.decode(gen).replace("\n", " ").strip()
            print(f"  Prompt: {p_str.splitlines()[0]} | Exp: {exp_act} | Gen: {pred}", flush=True)

    # Fuse & Save
    print("\nFusing Best Polish LoRA into standalone checkpoint...", flush=True)
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

    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k in sd:
            sd[k] = v.cpu().to(sd[k].dtype)
    torch.save(ckpt, OUT_PT)

    serialize_checkpoint_to_eer(OUT_PT, OUT_EER, target_dtype="fp16")
    print(f"ALL DONE in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()

