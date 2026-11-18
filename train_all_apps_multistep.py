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

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

BASE_CKPT = os.path.join(BASE_DIR, "checkpoints", "maria_1b_20B_weights.pt")
LORA_RESUME = os.path.join(BASE_DIR, "lora_all_apps_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_all_apps_v2_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
RICO_UI_JSONL = r"D:\hf_datasets\yori_rico_label_ui.jsonl"
LABEL_UI_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_label_ui_train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")
OUT_TRAIN = r"D:\hf_datasets\yori_all_apps_multistep.jsonl"

NOISE_BANK = [
    ("TextView", "Trending"), ("Tab", "Home"), ("Button", "Profile"),
    ("TextView", "Notifications"), ("Button", "More options"), ("TextView", "Recommended"),
    ("TextView", "Sponsored"), ("Tab", "Explore"), ("TextView", "Updates"),
    ("TextView", "About"), ("TextView", "Help"), ("Button", "Back"),
]

def make_screen(pkg, target_line, n_total=6):
    pos = random.randint(1, n_total)
    noise = random.sample(NOISE_BANK, n_total)
    lines = [f"[SCREEN: {pkg}]"]
    ptr = 0
    for idx in range(1, n_total + 1):
        if idx == pos:
            lines.append(f"[{idx}] {target_line}")
        else:
            t, l = noise[ptr % len(noise)]
            ptr += 1
            lines.append(f'[{idx}] {t} "{l}"')
    return "\n".join(lines)

def make_screen_with_filled_edit(pkg, filled_text, target_line, n_total=5):
    pos = random.randint(2, n_total)
    noise = random.sample(NOISE_BANK, n_total)
    lines = [f"[SCREEN: {pkg}]", f'[1] EditText "{filled_text}" [Input]']
    ptr = 0
    for idx in range(2, n_total + 1):
        if idx == pos:
            lines.append(f"[{idx}] {target_line}")
        else:
            t, l = noise[ptr % len(noise)]
            ptr += 1
            lines.append(f'[{idx}] {t} "{l}"')
    return "\n".join(lines)

def generate_all_phone_workflows():
    random.seed(2060)
    items = []

    # 1. Search -> Type -> Click Result apps
    search_apps = [
        ("YouTube", "com.google.android.youtube", "find Agentic AI lecture and play it from youtube", "Search", "Search YouTube", "Agentic AI lecture", "Agentic AI Lecture - Full Course"),
        ("YouTube", "com.google.android.youtube", "search transformer architecture on youtube", "Search", "Search YouTube", "transformer architecture", "Transformers Explained Clearly"),
        ("Spotify", "com.spotify.music", "play lofi beats on spotify", "Search", "What do you want to listen to?", "lofi beats", "Lofi Beats Playlist"),
        ("Spotify", "com.spotify.music", "play workout music on spotify", "Search", "What do you want to listen to?", "workout music", "Workout Hits Mix"),
        ("Maps", "com.google.android.apps.maps", "find coffee shops near me on maps", "Search here", "Search here", "coffee shops", "Blue Bottle Coffee"),
        ("Maps", "com.google.android.apps.maps", "navigate to airport on maps", "Search here", "Search here", "airport", "International Airport"),
        ("Amazon", "com.amazon.mShop.android.shopping", "search wireless earbuds on amazon", "Search Amazon", "Search Amazon", "wireless earbuds", "Sony Wireless Earbuds"),
        ("Play Store", "com.android.vending", "install notion from play store", "Search apps & games", "Search apps & games", "notion", "Install"),
        ("Chrome", "com.android.chrome", "search latest AI news in chrome", "Search or type web address", "Search or type web address", "latest AI news", "AI News Today"),
        ("Contacts", "com.google.android.contacts", "find Bilal number in contacts", "Search contacts", "Search contacts", "Bilal", "Bilal"),
    ]
    for app, pkg, goal, s_btn, box, q, res in search_apps:
        for _ in range(12):
            items.append({"category": "tool_call", "prompt": f"<user>{goal}</user>\n<thought>", "target": f'User wants to use {app}. I will open {app} first.</thought>\n<action>OPEN_APP("{app}")</action>'})
            items.append({"category": "ui_grounding", "prompt": f"<user>{goal}</user>\n<screen>\n{make_screen(pkg, f'Button \"{s_btn}\"', 7)}\n</screen>\n<thought>", "target": f'I will click "{s_btn}".</thought>\n<action>CLICK("{s_btn}")</action>'})
            items.append({"category": "ui_grounding", "prompt": f"<user>{goal}</user>\n<screen>\n{make_screen(pkg, f'EditText \"{box}\" [Input]', 4)}\n</screen>\n<thought>", "target": f'I will type "{q}" into "{box}".</thought>\n<action>TYPE("{box}", "{q}")</action>'})
            # Step 4 with already-filled EditText at [1]!
            items.append({"category": "ui_grounding", "prompt": f"<user>{goal}</user>\n<screen>\n{make_screen_with_filled_edit(pkg, q, f'TextView \"{res}\"', 5)}\n</screen>\n<thought>", "target": f'"{q}" is already entered. I will click "{res}".</thought>\n<action>CLICK("{res}")</action>'})
            items.append({"category": "ui_grounding", "prompt": f"<user>{goal}</user>\n<screen>\n{make_screen_with_filled_edit(pkg, q, f'TextView \"{res}\"', 5)}\n</screen>\n<thought>", "target": f'I will click "{res}".</thought>\n<action>CLICK("{res}")</action>'})

    # 2. Settings multi-step workflows (include Wi-Fi + Bluetooth distractors together)
    settings_flows = [
        ("open settings and turn on bluetooth", "Connections", 'Switch "Bluetooth" (OFF)', "Bluetooth"),
        ("open settings and turn off wifi", "Connections", 'Switch "Wi-Fi" (ON)', "Wi-Fi"),
        ("enable dark mode in settings", "Display", 'Switch "Dark" (OFF)', "Dark"),
        ("turn on airplane mode in settings", "Connections", 'Switch "Flight mode" (OFF)', "Flight mode"),
        ("check battery usage in settings", "Battery", 'TextView "Battery usage"', "Battery usage"),
        ("turn on location in settings", "Location", 'Switch "Location" (OFF)', "Location"),
    ]
    for goal, cat_lbl, toggle_node, toggle_lbl in settings_flows:
        for _ in range(15):
            items.append({"category": "tool_call", "prompt": f"<user>{goal}</user>\n<thought>", "target": 'User wants to change a setting. I will open Settings first.</thought>\n<action>OPEN_APP("Settings")</action>'})
            items.append({"category": "ui_grounding", "prompt": f"<user>{goal}</user>\n<screen>\n{make_screen('com.android.settings', f'TextView \"{cat_lbl}\"', 6)}\n</screen>\n<thought>", "target": f'I will click "{cat_lbl}".</thought>\n<action>CLICK("{cat_lbl}")</action>'})
            items.append({"category": "ui_grounding", "prompt": f"<user>{goal}</user>\n<screen>\n[SCREEN: com.android.settings]\n[1] Switch \"Wi-Fi\" (ON)\n[2] Switch \"Bluetooth\" (OFF)\n[3] Switch \"NFC\" (OFF)\n[4] Switch \"Flight mode\" (OFF)\n[5] Switch \"Dark\" (OFF)\n[6] {toggle_node}\n</screen>\n<thought>", "target": f'I will click "{toggle_lbl}".</thought>\n<action>CLICK("{toggle_lbl}")</action>'})

    # 3. WhatsApp, Clock/Alarm, Notes, Gmail multi-step workflows (with filled EditText -> Send/Save!)
    app_flows = [
        ("WhatsApp", "com.whatsapp", "open whatsapp and message Bilal saying Running late", [
            (make_screen("com.whatsapp", 'TextView "Bilal"', 5), 'CLICK("Bilal")', 'I will click "Bilal".'),
            (make_screen("com.whatsapp", 'EditText "Message" [Input]', 4), 'TYPE("Message", "Running late")', 'I will type "Running late" into "Message".'),
            (make_screen_with_filled_edit("com.whatsapp", "Running late", 'Button "Send"', 3), 'CLICK("Send")', 'Message is typed. I will click "Send".'),
            (make_screen_with_filled_edit("com.whatsapp", "Running late", 'Button "Send"', 3), 'CLICK("Send")', 'I will click "Send".'),
        ]),
        ("Clock", "com.sec.android.app.clockpackage", "open clock and create alarm named Yori Lunch", [
            (make_screen("com.sec.android.app.clockpackage", 'Button "Add alarm"', 4), 'CLICK("Add alarm")', 'I will click "Add alarm".'),
            (make_screen("com.sec.android.app.clockpackage", 'EditText "Alarm name" [Input]', 4), 'TYPE("Alarm name", "Yori Lunch")', 'I will type "Yori Lunch" into "Alarm name".'),
            (make_screen_with_filled_edit("com.sec.android.app.clockpackage", "Yori Lunch", 'Button "Save"', 3), 'CLICK("Save")', 'Alarm name is typed. I will click "Save".'),
            (make_screen_with_filled_edit("com.sec.android.app.clockpackage", "Yori Lunch", 'Button "Save"', 3), 'CLICK("Save")', 'I will click "Save".'),
        ]),
        ("Notes", "com.samsung.android.app.notes", "open notes and write Buy milk", [
            (make_screen("com.samsung.android.app.notes", 'Button "Create note"', 4), 'CLICK("Create note")', 'I will click "Create note".'),
            (make_screen("com.samsung.android.app.notes", 'EditText "Title" [Input]', 4), 'TYPE("Title", "Buy milk")', 'I will type "Buy milk" into "Title".'),
            (make_screen_with_filled_edit("com.samsung.android.app.notes", "Buy milk", 'Button "Save"', 3), 'CLICK("Save")', 'Note is typed. I will click "Save".'),
            (make_screen_with_filled_edit("com.samsung.android.app.notes", "Buy milk", 'Button "Save"', 3), 'CLICK("Save")', 'I will click "Save".'),
        ]),
    ]
    for app, pkg, goal, steps in app_flows:
        for _ in range(15):
            items.append({"category": "tool_call", "prompt": f"<user>{goal}</user>\n<thought>", "target": f'User wants to use {app}. I will open {app} first.</thought>\n<action>OPEN_APP("{app}")</action>'})
            for scr_str, act_str, th_str in steps:
                items.append({"category": "ui_grounding", "prompt": f"<user>{goal}</user>\n<screen>\n{scr_str}\n</screen>\n<thought>", "target": f'{th_str}</thought>\n<action>{act_str}</action>'})

    return items

def build_dataset():
    random.seed(2061)
    items = generate_all_phone_workflows()

    with open(RICO_UI_JSONL, "r", encoding="utf-8") as f:
        rico_items = [json.loads(line) for line in f if line.strip()]
    random.shuffle(rico_items)
    items.extend(rico_items[:200])

    ui_pool, chat_pool, tool_pool = [], [], []
    with open(LABEL_UI_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if obj["category"] == "ui_grounding": ui_pool.append(obj)
            elif obj["category"] == "chat": chat_pool.append(obj)
            elif obj["category"] == "tool_call": tool_pool.append(obj)

    random.shuffle(ui_pool)
    random.shuffle(chat_pool)
    random.shuffle(tool_pool)
    items.extend(ui_pool[:150])
    items.extend(chat_pool[:250])
    items.extend(tool_pool[:150])

    random.shuffle(items)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Built {OUT_TRAIN}: {len(items)} examples across all 14+ phone apps & settings!", flush=True)

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

def load_tokenized(path, tokenizer, max_len=440):
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
    print(f"=== YORI ALL-PHONE APPS & SETTINGS MULTI-STEP SFT ON {device} ===", flush=True)
    t0 = time.time()

    build_dataset()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    train_items = load_tokenized(OUT_TRAIN, tokenizer)
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
    MAX_STEPS = 80
    WARMUP_STEPS = 8
    PEAK_LR = 2.5e-5
    MIN_LR = 5e-6
    GRAD_ACCUM = 8

    optimizer = torch.optim.AdamW(trainable_params, lr=PEAK_LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        prog = float(step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return (MIN_LR / PEAK_LR) + (1.0 - (MIN_LR / PEAK_LR)) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pre = eval_val_ppl(model, val_items, device)
    print(f"  [VAL @ Step 0] chat PPL={pre['chat'][1]:.2f} | tool_call PPL={pre['tool_call'][1]:.2f}", flush=True)

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
            c_ppl, t_ppl = val["chat"][1], val["tool_call"][1]
            print(f"Step {step:3d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | VAL: chat={c_ppl:.2f}, tool={t_ppl:.2f} | Elapsed={time.time() - t0:.1f}s", flush=True)
            if t_ppl <= 1.50:
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA)

    model.load_state_dict(best_lora_state, strict=False)

    print("\nFusing All-Apps LoRA into standalone checkpoint...", flush=True)
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

