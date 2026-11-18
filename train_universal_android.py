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
LORA_RESUME = os.path.join(BASE_DIR, "lora_rico_sft_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_universal_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")

RICO_UI_JSONL = r"D:\hf_datasets\yori_rico_label_ui.jsonl"
LABEL_UI_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_label_ui_train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")
UNIV_TRAIN = r"D:\hf_datasets\yori_universal_train.jsonl"

def generate_universal_multistep_examples():
    random.seed(2035)
    examples = []

    apps_and_tasks = [
        ("YouTube", ["find Agentic AI lecture and play it from youtube", "search LoRA fine tuning on youtube", "play Python tutorial on youtube", "watch robotics demo on youtube", "open youtube and search transformer architecture"]),
        ("Spotify", ["play lofi beats on spotify", "find blinding lights and play it on spotify", "search jazz playlist in spotify", "open spotify and play workout music"]),
        ("Maps", ["find coffee shops near me on maps", "search Hyderabad airport in maps", "navigate to nearest pharmacy using maps", "open maps and find gas station"]),
        ("Chrome", ["open chrome and search latest AI news", "find weather forecast in chrome"]),
        ("Amazon", ["search wireless earbuds on amazon", "find laptop stand in amazon"]),
        ("Settings", ["open settings to check wifi", "go to settings"]),
        ("WhatsApp", ["open whatsapp to message Bilal", "find Sarah's chat on whatsapp"]),
        ("Play Store", ["find notion app on play store", "search telegram in play store"]),
    ]

    for app, prompts in apps_and_tasks:
        for p in prompts:
            for _ in range(4):
                examples.append({
                    "category": "tool_call",
                    "prompt": f"<user>{p}</user>\n<thought>",
                    "target": f'User wants to use {app}. I will open {app} first.</thought>\n<action>OPEN_APP("{app}")</action>'
                })

    # Universal multi-turn screen steps (Search button -> Search EditText -> Click Result)
    search_screens = [
        ("com.google.android.youtube", "YouTube", "Search", "Search YouTube", "Agentic AI lecture", "Agentic AI Lecture - Full Course"),
        ("com.spotify.music", "Spotify", "Search", "What do you want to listen to?", "lofi beats", "Lofi Beats Playlist"),
        ("com.google.android.apps.maps", "Maps", "Search here", "Search here", "coffee shops", "Blue Bottle Coffee"),
        ("com.android.chrome", "Chrome", "Search or type web address", "Search or type web address", "latest AI news", "AI News Today"),
        ("com.whatsapp", "WhatsApp", "Search", "Search...", "Bilal", "Bilal"),
    ]
    for pkg, app, search_btn, search_box, query_txt, result_lbl in search_screens:
        for _ in range(10):
            # Step 2: Click Search icon/button
            s2 = f'[SCREEN: {pkg}]\n[1] TextView "{app}"\n[2] Button "{search_btn}"\n[3] Tab "Home"\n[4] Tab "Explore"\n[5] Button "Profile"'
            examples.append({
                "category": "ui_grounding",
                "prompt": f"<user>find {query_txt} and play it from {app.lower()}</user>\n<screen>\n{s2}\n</screen>\n<thought>",
                "target": f'I see "{search_btn}" on {app}. I will click "{search_btn}".</thought>\n<action>CLICK("{search_btn}")</action>'
            })
            # Step 3: Type query into search EditText
            s3 = f'[SCREEN: {pkg}]\n[1] ImageButton "Navigate up"\n[2] EditText "{search_box}" [Input]\n[3] Button "Voice search"'
            examples.append({
                "category": "ui_grounding",
                "prompt": f"<user>find {query_txt} and play it from {app.lower()}</user>\n<screen>\n{s3}\n</screen>\n<thought>",
                "target": f'I will type "{query_txt}" into "{search_box}".</thought>\n<action>TYPE("{search_box}", "{query_txt}")</action>'
            })
            # Step 4: Click matching result
            s4 = f'[SCREEN: {pkg}]\n[1] EditText "{query_txt}" [Input]\n[2] Button "Filters"\n[3] TextView "{result_lbl}"\n[4] TextView "Other video"\n[5] TextView "Sponsored"'
            examples.append({
                "category": "ui_grounding",
                "prompt": f"<user>find {query_txt} and play it from {app.lower()}</user>\n<screen>\n{s4}\n</screen>\n<thought>",
                "target": f'I see "{result_lbl}" in the results. I will click "{result_lbl}".</thought>\n<action>CLICK("{result_lbl}")</action>'
            })

    # Extra OS operations (LOCK_SCREEN, SCREENSHOT, GLOBAL_*)
    os_ops = [
        ("Lock my phone", "LOCK_SCREEN()"),
        ("Lock the screen", "LOCK_SCREEN()"),
        ("Take a screenshot", "SCREENSHOT()"),
        ("Go back", "GLOBAL_BACK()"),
        ("Go to home screen", "GLOBAL_HOME()"),
        ("Open notifications", "GLOBAL_NOTIFICATIONS()"),
        ("Open quick settings", "GLOBAL_QUICK_SETTINGS()"),
        ("Show recent apps", "GLOBAL_RECENTS()"),
    ]
    for u, act in os_ops:
        for _ in range(3):
            examples.append({
                "category": "tool_call",
                "prompt": f"<user>{u}</user>\n<thought>",
                "target": f"User wants to perform {act}.</thought>\n<action>{act}</action>"
            })

    return examples

def build_universal_dataset():
    random.seed(2036)
    items = generate_universal_multistep_examples()

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

    items.extend(ui_pool[:120])
    items.extend(chat_pool[:250])
    items.extend(tool_pool[:150])

    random.shuffle(items)
    with open(UNIV_TRAIN, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Built {UNIV_TRAIN}: {len(items)} examples", flush=True)

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
    print(f"=== YORI UNIVERSAL ANDROID MULTI-APP + MULTI-STEP SFT ON {device} ===", flush=True)
    t0 = time.time()

    build_universal_dataset()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    train_items = load_tokenized(UNIV_TRAIN, tokenizer)
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
    MAX_STEPS = 100
    WARMUP_STEPS = 10
    PEAK_LR = 2e-5
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

        if step % 25 == 0 or step == 1 or step == MAX_STEPS:
            val = eval_val_ppl(model, val_items, device)
            c_ppl, t_ppl = val["chat"][1], val["tool_call"][1]
            print(f"Step {step:3d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | VAL: chat={c_ppl:.2f}, tool={t_ppl:.2f} | Elapsed={time.time() - t0:.1f}s", flush=True)
            if t_ppl <= 1.50:
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA)

    model.load_state_dict(best_lora_state, strict=False)

    print("\nFusing Universal Android LoRA into standalone checkpoint...", flush=True)
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

