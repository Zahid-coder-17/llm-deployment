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
LORA_RESUME = os.path.join(BASE_DIR, "lora_grpo_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_step34_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
LABEL_UI_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_label_ui_train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")
OUT_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_step34_polish.jsonl")

def build_dataset():
    random.seed(2050)
    items = []

    step34_specs = [
        ("YouTube", "com.google.android.youtube", "find Agentic AI lecture and play it from youtube", "Search", "Search YouTube", "Agentic AI lecture", "Agentic AI Lecture - Full Course"),
        ("YouTube", "com.google.android.youtube", "search transformer architecture on youtube", "Search", "Search YouTube", "transformer architecture", "Transformers Explained Clearly"),
        ("Spotify", "com.spotify.music", "play lofi beats on spotify", "Search", "What do you want to listen to?", "lofi beats", "Lofi Beats Playlist"),
        ("Spotify", "com.spotify.music", "find jazz classics on spotify", "Search", "What do you want to listen to?", "jazz classics", "Jazz Classics Radio"),
        ("Maps", "com.google.android.apps.maps", "find coffee shops near me on maps", "Search here", "Search here", "coffee shops", "Blue Bottle Coffee"),
        ("Maps", "com.google.android.apps.maps", "find gas station on maps", "Search here", "Search here", "gas station", "Shell Gas Station"),
        ("Amazon", "com.amazon.mShop.android.shopping", "search wireless earbuds on amazon", "Search Amazon", "Search Amazon", "wireless earbuds", "Sony Wireless Earbuds"),
        ("Play Store", "com.android.vending", "find notion app on play store", "Search apps & games", "Search apps & games", "notion app", "Notion: Notes, Tasks, AI"),
        ("WhatsApp", "com.whatsapp", "find Bilal chat on whatsapp", "Search", "Search...", "Bilal", "Bilal"),
        ("Chrome", "com.android.chrome", "search latest AI news in chrome", "Search or type web address", "Search or type web address", "latest AI news", "AI News Today"),
    ]
    for _ in range(50):
        app_name, pkg, u_goal, search_btn, box_lbl, query_txt, res_lbl = random.choice(step34_specs)
        # Step 1: OPEN_APP
        items.append({
            "category": "tool_call",
            "prompt": f"<user>{u_goal}</user>\n<thought>",
            "target": f'User wants to use {app_name}. I will open {app_name} first.</thought>\n<action>OPEN_APP("{app_name}")</action>'
        })
        # Step 2: Click Search
        s2 = f'[SCREEN: {pkg}]\n[1] TextView "{app_name}"\n[2] Button "{search_btn}"\n[3] Tab "Home"\n[4] TextView "Trending"\n[5] Button "Profile"'
        items.append({
            "category": "ui_grounding",
            "prompt": f"<user>{u_goal}</user>\n<screen>\n{s2}\n</screen>\n<thought>",
            "target": f'I will click "{search_btn}".</thought>\n<action>CLICK("{search_btn}")</action>'
        })
        # Step 3: Type into EditText
        s3 = f'[SCREEN: {pkg}]\n[1] ImageButton "Back"\n[2] EditText "{box_lbl}" [Input]\n[3] Button "Voice"'
        items.append({
            "category": "ui_grounding",
            "prompt": f"<user>{u_goal}</user>\n<screen>\n{s3}\n</screen>\n<thought>",
            "target": f'I will type "{query_txt}" into "{box_lbl}".</thought>\n<action>TYPE("{box_lbl}", "{query_txt}")</action>'
        })
        # Step 4: Click Result
        distractors = ["Sponsored Ad", "Related Item", "More Results", "Filters"]
        r_pos = random.randint(2, 5)
        r_lines = [f"[SCREEN: {pkg}]", f'[1] EditText "{query_txt}" [Input]']
        d_ptr = 0
        for idx in range(2, 6):
            if idx == r_pos:
                r_lines.append(f'[{idx}] TextView "{res_lbl}"')
            else:
                r_lines.append(f'[{idx}] TextView "{distractors[d_ptr % len(distractors)]}"')
                d_ptr += 1
        items.append({
            "category": "ui_grounding",
            "prompt": f"<user>{u_goal}</user>\n<screen>\n" + "\n".join(r_lines) + "\n</screen>\n<thought>",
            "target": f'I will click "{res_lbl}".</thought>\n<action>CLICK("{res_lbl}")</action>'
        })

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
    items.extend(chat_pool[:80])
    items.extend(tool_pool[:40])

    random.shuffle(items)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Built {OUT_TRAIN}: {len(items)} examples (160 Step3/4 + 120 UI + 80 chat + 40 tool_call)", flush=True)

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
    print(f"=== YORI STEP 3/4 + GRPO-V1 REPLAY POLISH ON {device} ===", flush=True)
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
    print(f"Restored 8/8 checkpoint {LORA_RESUME}.", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    MAX_STEPS = 50
    WARMUP_STEPS = 5
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

        if step % 25 == 0 or step == 1 or step == MAX_STEPS:
            val = eval_val_ppl(model, val_items, device)
            c_ppl, t_ppl = val["chat"][1], val["tool_call"][1]
            print(f"Step {step:2d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | VAL: chat={c_ppl:.2f}, tool={t_ppl:.2f} | Elapsed={time.time() - t0:.1f}s", flush=True)
            if t_ppl <= 1.50:
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA)

    model.load_state_dict(best_lora_state, strict=False)

    print("\nFusing Step 3/4 Polish LoRA into standalone checkpoint...", flush=True)
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

