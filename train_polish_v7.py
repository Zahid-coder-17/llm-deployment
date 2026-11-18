import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
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
LORA_RESUME = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\lora_polish_v6_best.pt"
OUT_LORA = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\lora_polish_v7_best.pt"
OUT_PT = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
OUT_EER = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.eer"
TOK_PATH = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")

ALPACA_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\alpaca_data_cleaned.json"
POLISH_TRAIN_V1 = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_polish_train.jsonl"
VAL_JSONL = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_unified_val.jsonl"
POLISH_TRAIN_V7 = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_polish_v7_train.jsonl"

HARD_NEGATIVE_CHAT = [
    ("Who are you?", "I'm Yori, an offline executive AI assistant built by Embeddra AI to chat locally and control your Android phone."),
    ("Write a polite message saying I am running late", "Hi! I'm running a few minutes late, sorry for the delay. I'll be there as soon as I can!"),
    ("Draft a polite message declining a meeting tomorrow", "Thank you for the invite, but I have a scheduling conflict tomorrow. Could we move it to another day?"),
    ("Draft a text to my boss asking for leave", "Hi, could I please take a day of leave tomorrow? I'll make sure all urgent tasks are covered."),
    ("Write a quick WhatsApp reply saying I'm busy", "Hey, I'm tied up right now—I'll message you back as soon as I'm free!"),
    ("Explain recursion simply", "Recursion is when a function calls itself with a smaller input until it hits a stopping condition called the base case."),
    ("Explain what a flashlight is", "A flashlight is a portable hand-held electric light powered by batteries, usually using an LED bulb and reflector."),
    ("Why is the sky blue?", "Sunlight hits air molecules in Earth's atmosphere, which scatter short blue light waves much more than red waves, making the sky look blue."),
    ("Tell me how Bluetooth works", "Bluetooth uses short-range radio waves in the 2.4 GHz band to connect nearby devices wirelessly without cables."),
    ("Explain how Wi-Fi works", "Wi-Fi uses radio signals between a wireless router and your device to transmit internet data through the air."),
]

def generate_copy_convergence_ui(n_examples=1200):
    random.seed(707)
    switch_distractors = [
        ("Switch", "Wi-Fi", "ON"), ("Switch", "NFC", "OFF"), ("Switch", "Flight mode", "OFF"),
        ("Switch", "Hotspot", "OFF"), ("Switch", "Location", "ON"), ("Switch", "Auto-rotate", "ON"),
        ("Switch", "Do not disturb", "OFF"), ("Switch", "Data saver", "OFF"),
        ("Switch", "Alarm sound", "ON"), ("Switch", "Snooze", "ON"), ("Switch", "Vibration", "ON"),
        ("Switch", "Dark mode", "OFF"), ("Switch", "Battery saver", "OFF"),
    ]
    edittext_distractors = [
        ("EditText", "Search", "[Input]"), ("EditText", "Notes", "[Input]"),
        ("EditText", "Phone number", "[Input]"), ("EditText", "Email", "[Input]"),
        ("EditText", "Description", "[Input]"), ("EditText", "First name", "[Input]"),
    ]
    textview_distractors = [
        ("TextView", "Pinned: Mom", None), ("TextView", "Tech Support", None),
        ("TextView", "Sarah", None), ("TextView", "Alex", None), ("TextView", "Office Group", None),
        ("TextView", "Connections", None), ("TextView", "SIM manager", None),
        ("TextView", "Mobile networks", None), ("TextView", "Data usage", None),
        ("TextView", "Notifications", None), ("TextView", "Storage", None),
        ("TextView", "More connection settings", None),
    ]
    general_distractors = [
        ("ImageButton", "Navigate up", None), ("Button", "Cancel", None),
        ("Button", "More options", None), ("Tab", "Chats", None), ("Tab", "Updates", None),
        ("Tab", "Calls", None), ("CheckBox", "Mon", None), ("CheckBox", "Tue", None),
        ("CheckBox", "Wed", None), ("CheckBox", "Thu", None), ("CheckBox", "Fri", None),
    ]

    packages = [
        "com.android.settings", "com.android.settings.connections",
        "com.sec.android.app.clockpackage.edit", "com.whatsapp", "com.google.android.dialer"
    ]
    click_targets = [
        ("Open Bilal's chat on WhatsApp", "TextView", "Bilal", None, "to open the chat"),
        ("Open Sarah's chat", "TextView", "Sarah", None, "to open the chat"),
        ("Open Tech Support chat", "TextView", "Tech Support", None, "to open the chat"),
        ("Open Alex's chat", "TextView", "Alex", None, "to open the chat"),
        ("Turn on Bluetooth", "Switch", "Bluetooth", "OFF", "to turn it on"),
        ("Turn off Wi-Fi", "Switch", "Wi-Fi", "ON", "to turn it off"),
        ("Enable NFC", "Switch", "NFC", "OFF", "to enable it"),
        ("Turn on Flight mode", "Switch", "Flight mode", "OFF", "to enable it"),
        ("Turn on Hotspot", "Switch", "Hotspot", "OFF", "to enable it"),
        ("Tap Save to confirm", "Button", "Save", None, "to save"),
        ("Tap Send message", "Button", "Send", None, "to send the message"),
    ]
    type_targets = [
        ("Enter 'Yori Lunch' as the alarm name", "Alarm name", "Yori Lunch"),
        ("Type 'Team Standup' into alarm label", "Alarm name", "Team Standup"),
        ("Set alarm label to 'Gym 6 AM'", "Label", "Gym 6 AM"),
        ("Search for 'Bilal' in contacts", "Search contacts", "Bilal"),
        ("Type 'Hello from Yori' in message box", "Type a message", "Hello from Yori"),
        ("Enter 'Running 10 mins late' in chat", "Message", "Running 10 mins late"),
        ("Search 'Bluetooth' in settings", "Search settings", "Bluetooth"),
        ("Enter 'Morning Workout' for alarm", "Alarm name", "Morning Workout"),
    ]

    ui_examples = []
    for i in range(n_examples):
        total_elems = random.randint(9, 16)
        target_idx = random.randint(1, total_elems)
        pkg = random.choice(packages)
        is_type = (i % 2 == 0)

        lines = [f"[SCREEN: {pkg}]"]

        if is_type:
            intent, label, text_val = random.choice(type_targets)
            pool = [d for d in edittext_distractors if d[1] != label][:4]
            pool += random.sample(switch_distractors, 5)
            pool += random.sample(textview_distractors, 6)
            pool += random.sample(general_distractors, 5)
            random.shuffle(pool)

            ptr = 0
            for idx in range(1, total_elems + 1):
                if idx == target_idx:
                    lines.append(f'[{idx}] EditText "{label}" [Input] -> [{idx}]')
                else:
                    w_type, w_lbl, w_st = pool[ptr % len(pool)]
                    ptr += 1
                    st_str = f" {w_st}" if w_st and w_st.startswith("[") else (f" ({w_st})" if w_st else "")
                    lines.append(f'[{idx}] {w_type} "{w_lbl}"{st_str} -> [{idx}]')

            thought = f'Target "{label}" is at index [{target_idx}]. I will type "{text_val}" into [{target_idx}].'
            action = f'TYPE([{target_idx}], "{text_val}")'
        else:
            intent, t_type, label, state, reason = random.choice(click_targets)
            if t_type == "Switch":
                same_type = [d for d in switch_distractors if d[1] != label]
            elif t_type == "TextView":
                same_type = [d for d in textview_distractors if d[1] != label]
            else:
                same_type = list(general_distractors)

            pool = random.sample(same_type, min(6, len(same_type)))
            pool += random.sample(switch_distractors, 4)
            pool += random.sample(textview_distractors, 6)
            pool += random.sample(general_distractors, 4)
            pool = [d for d in pool if d[1] != label]
            random.shuffle(pool)

            ptr = 0
            for idx in range(1, total_elems + 1):
                if idx == target_idx:
                    st_str = f" ({state})" if state else ""
                    lines.append(f'[{idx}] {t_type} "{label}"{st_str} -> [{idx}]')
                else:
                    w_type, w_lbl, w_st = pool[ptr % len(pool)]
                    ptr += 1
                    st_str = f" ({w_st})" if w_st else ""
                    lines.append(f'[{idx}] {w_type} "{w_lbl}"{st_str} -> [{idx}]')

            thought = f'Target "{label}" is at index [{target_idx}]. I will click [{target_idx}] {reason}.'
            action = f'CLICK([{target_idx}])'

        screen_str = "\n".join(lines)
        ui_examples.append({
            "category": "ui_grounding",
            "prompt": f"<user>{intent}</user>\n<screen>\n{screen_str}\n</screen>\n<thought>",
            "target": f"{thought}</thought>\n<action>{action}</action>"
        })
    return ui_examples

def build_polish_v7_dataset():
    random.seed(2031)
    items = []

    items.extend(generate_copy_convergence_ui(1200))

    for idx in range(180):
        u, a = HARD_NEGATIVE_CHAT[idx % len(HARD_NEGATIVE_CHAT)]
        items.append({"category": "chat", "prompt": f"<user>{u}</user>\n<thought>", "target": f"{a}</thought>"})

    with open(ALPACA_PATH, "r", encoding="utf-8") as f:
        alpaca = json.load(f)
    alp_pool = []
    for row in alpaca:
        instr = row["instruction"].strip()
        inp = row["input"].strip()
        out = row["output"].strip()
        if not instr or not out or len(out) < 20 or len(out) > 220 or len(instr) > 160:
            continue
        user_msg = f"{instr}\n{inp}".strip() if inp else instr
        alp_pool.append({"category": "chat", "prompt": f"<user>{user_msg}</user>\n<thought>", "target": f"{out}</thought>"})
    random.shuffle(alp_pool)
    items.extend(alp_pool[:120])

    tools = []
    with open(POLISH_TRAIN_V1, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["category"] == "tool_call":
                tools.append(obj)
    random.shuffle(tools)
    items.extend(tools[:150])

    random.shuffle(items)
    with open(POLISH_TRAIN_V7, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Built {POLISH_TRAIN_V7}: {len(items)} examples (1200 [1..16] UI, 300 chat, 150 tool_call)", flush=True)

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

def load_tokenized(path, tokenizer, max_len=460):
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
                items.append((obj["category"], full_ids, labels))
    return items

def eval_val_ppl(model, val_items, device):
    model.eval()
    stats = {}
    with torch.no_grad():
        for cat, full_ids, labels in val_items:
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
    print(f"=== YORI POLISH V7 (200-STEP COPY CONVERGENCE @ LR=5e-5) ON {device} ===", flush=True)
    t0 = time.time()

    build_polish_v7_dataset()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    train_items = load_tokenized(POLISH_TRAIN_V7, tokenizer)
    val_items = load_tokenized(VAL_JSONL, tokenizer)

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=32, lora_alpha=64).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=32, lora_alpha=64).to(device))

    lora_ckpt = torch.load(LORA_RESUME, map_location=device, weights_only=False)
    model.load_state_dict(lora_ckpt, strict=False)
    print(f"Resumed cleanly from {LORA_RESUME}.", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    MAX_STEPS = 200
    WARMUP_STEPS = 15
    PEAK_LR = 5e-5
    MIN_LR = 1e-5
    GRAD_ACCUM = 8

    optimizer = torch.optim.AdamW(trainable_params, lr=PEAK_LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        prog = float(step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return (MIN_LR / PEAK_LR) + (1.0 - (MIN_LR / PEAK_LR)) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pre = eval_val_ppl(model, val_items, device)
    print(f"  [VAL @ Step 0] chat PPL={pre['chat'][1]:.2f} | tool_call PPL={pre['tool_call'][1]:.2f} | ui_grounding PPL={pre['ui_grounding'][1]:.2f}", flush=True)

    idx_pool = list(range(len(train_items)))
    random.shuffle(idx_pool)
    ptr = 0
    best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            if ptr >= len(idx_pool):
                random.shuffle(idx_pool)
                ptr = 0
            _, full_ids, labels = train_items[idx_pool[ptr]]
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

        if step % 40 == 0 or step == 1 or step == MAX_STEPS:
            val = eval_val_ppl(model, val_items, device)
            c_ppl, t_ppl, u_ppl = val["chat"][1], val["tool_call"][1], val["ui_grounding"][1]
            print(f"Step {step:3d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | VAL: chat={c_ppl:.2f}, tool={t_ppl:.2f}, ui={u_ppl:.2f} | Elapsed={time.time() - t0:.1f}s", flush=True)
            if t_ppl <= 1.50:
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA)

    model.load_state_dict(best_lora_state, strict=False)

    print("\nFusing Polish V7 LoRA into standalone checkpoint...", flush=True)
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

