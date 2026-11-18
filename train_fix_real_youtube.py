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
LORA_RESUME = os.path.join(BASE_DIR, "lora_master_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_master_v2_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
PREV_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_master_train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")
OUT_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_fix_yt_train.jsonl")


def build_dataset():
    random.seed(4001)
    items = []

    yt_topics = [
        ("Agentic AI lecture", "Agentic AI Lecture - Full Course"),
        ("lo-fi beats", "Lofi Hip Hop Radio - Beats to Relax/Study to"),
        ("python tutorial", "Python for Beginners - Full Course"),
        ("transformer architecture", "Transformers Explained Clearly"),
        ("deep learning course", "Deep Learning Crash Course"),
        ("Andrej Karpathy lecture", "Intro to Large Language Models by Andrej Karpathy"),
        ("machine learning course", "Machine Learning Specialization"),
        ("data structures in python", "Data Structures Easy to Advanced"),
    ]

    # Real YouTube Home screens with shorts / video cards / distractors
    home_screens = [
        # Exactly what Samsung Galaxy A35 dumps:
        (
            '[SCREEN: com.google.android.youtube]\n'
            '[2] ImageView "YouTube"\n'
            '[3] Button "Notifications"\n'
            '[4] Button "Search"\n'
            '[6] ViewGroup "Shorts"\n'
            '[8] Button "Action menu"\n'
            '[9] Button "That one newly married in wedding..… – play Short"\n'
            '[10] Button "More actions"\n'
            '[11] ViewGroup "That one newly married in wedding..…"\n'
            '[12] Button "Rekha in simi garewal podcast – play Short"'
        ),
        (
            '[SCREEN: com.google.android.youtube]\n'
            '[1] ImageView "YouTube"\n'
            '[2] Button "Cast"\n'
            '[3] Button "Notifications"\n'
            '[4] Button "Search"\n'
            '[5] ViewGroup "Trending now – play Short"\n'
            '[6] Button "Daily News Bulletin"\n'
            '[7] Button "Comedy Club Live"'
        ),
        (
            '[SCREEN: com.google.android.youtube]\n'
            '[2] ImageView "YouTube"\n'
            '[4] Button "Search"\n'
            '[5] Button "Home"\n'
            '[6] ViewGroup "Shorts"\n'
            '[8] Button "Funny Moments Compilation – play Short"'
        ),
    ]

    for q_txt, res_title in yt_topics:
        user_prompt = f"find {q_txt} and play it from youtube"

        # Turn 1: No screen
        for _ in range(8):
            items.append({
                "category": "tool_call",
                "prompt": f"<user>{user_prompt}</user>\n<thought>",
                "target": 'User wants to use YouTube. I will open YouTube first.</thought>\n<action>OPEN_APP("YouTube")</action>'
            })

        # Turn 2: Home screens with distractors -> CLICK("Search")
        for hs in home_screens:
            for _ in range(6):
                items.append({
                    "category": "ui_grounding",
                    "prompt": f"<user>{user_prompt}</user>\n<screen>\n{hs}\n</screen>\n<thought>",
                    "target": 'I will click "Search".</thought>\n<action>CLICK("Search")</action>'
                })

        # Turn 3: Search screen with suggestions -> TYPE("Search YouTube", ...)
        search_box_screens = [
            (
                '[SCREEN: com.google.android.youtube]\n'
                '[1] Button "Voice search"\n'
                '[2] Button "Navigate up"\n'
                '[3] EditText "Search YouTube"\n'
                '[6] TextView "trending songs"\n'
                '[9] TextView "popular podcast"\n'
                '[12] TextView "news live"'
            ),
            (
                '[SCREEN: com.google.android.youtube]\n'
                '[1] Button "Voice search"\n'
                '[2] Button "Navigate up"\n'
                '[3] EditText "Search YouTube"\n'
                '[6] TextView "sonali bendre karishma kapoor"\n'
                '[9] TextView "hyderabad diaries"\n'
                '[12] TextView "geeta kapoor dance"'
            ),
        ]
        for sb in search_box_screens:
            for _ in range(8):
                items.append({
                    "category": "ui_grounding",
                    "prompt": f"<user>{user_prompt}</user>\n<screen>\n{sb}\n</screen>\n<thought>",
                    "target": f'I will type "{q_txt}" into "Search YouTube".</thought>\n<action>TYPE("Search YouTube", "{q_txt}")</action>'
                })

        # Turn 4: Results screen -> CLICK("{res_title}")
        results_screen = (
            '[SCREEN: com.google.android.youtube]\n'
            f'[1] EditText "{q_txt}"\n'
            '[2] Button "Clear search"\n'
            '[3] Button "Filters"\n'
            f'[4] ViewGroup "{res_title}"\n'
            '[5] ViewGroup "Sponsored Ad"'
        )
        for _ in range(8):
            items.append({
                "category": "ui_grounding",
                "prompt": f"<user>{user_prompt}</user>\n<screen>\n{results_screen}\n</screen>\n<thought>",
                "target": f'I will click "{res_title}".</thought>\n<action>CLICK("{res_title}")</action>'
            })

    # Replay all previous master items to maintain 100% smoke/QA scores
    with open(PREV_TRAIN, "r", encoding="utf-8") as f:
        prev_items = [json.loads(line) for line in f if line.strip()]

    # Filter out old ambiguous YouTube Turn 2 examples from prev
    clean_prev = []
    for it in prev_items:
        p = it["prompt"]
        if "from youtube" in p and "[SCREEN: com.google.android.youtube]" in p and "CLICK(\"Search\")" in it["target"]:
            continue  # replace with our comprehensive home screens
        clean_prev.append(it)

    items.extend(clean_prev)
    random.shuffle(items)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Built fix dataset: {len(items)} examples -> {OUT_TRAIN}", flush=True)


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 32, lora_alpha: int = 64):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.scaling = lora_alpha / r
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, original_linear.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(original_linear.out_features, r, dtype=dtype))
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original_linear(x)
        lora = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return orig + lora


def load_tokenized(path, tokenizer, max_len=440):
    eos_id = tokenizer.token_to_id("<|end_of_text|>") or 128001
    bos_id = tokenizer.token_to_id("<|begin_of_text|>") or 128000
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            p_ids = [bos_id] + tokenizer.encode(obj["prompt"]).ids
            t_ids = tokenizer.encode(obj["target"]).ids + [eos_id]
            full_ids = (p_ids + t_ids)[:max_len]
            labels = ([-100] * len(p_ids) + t_ids)[:max_len]
            if any(x != -100 for x in labels):
                data.append((obj.get("category", "chat"), full_ids, labels))
    return data


@torch.no_grad()
def eval_val_ppl(model, val_items, device):
    model.eval()
    stats = {"chat": [0.0, 0], "tool_call": [0.0, 0]}
    for cat, full_ids, labels in val_items:
        c = "tool_call" if cat != "chat" else "chat"
        x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100, reduction="sum")
        n_tok = int((y != -100).sum().item())
        stats[c][0] += float(loss.item())
        stats[c][1] += n_tok
    return {k: (v[0] / max(1, v[1]), math.exp(min(20.0, v[0] / max(1, v[1])))) for k, v in stats.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== FIX REAL YOUTUBE MULTI-STEP ON {device} ===", flush=True)
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
    MAX_STEPS = 45
    WARMUP_STEPS = 4
    PEAK_LR = 2.0e-5
    MIN_LR = 4e-6
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
    best_t_ppl = float("inf")
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

        if step % 15 == 0 or step == 1 or step == MAX_STEPS:
            val = eval_val_ppl(model, val_items, device)
            c_ppl, t_ppl = val["chat"][1], val["tool_call"][1]
            print(f"Step {step:2d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | VAL: chat={c_ppl:.2f}, tool={t_ppl:.2f} | Elapsed={time.time() - t0:.1f}s", flush=True)
            if t_ppl <= best_t_ppl or step == MAX_STEPS:
                best_t_ppl = min(best_t_ppl, t_ppl)
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA)

    model.load_state_dict(best_lora_state, strict=False)

    print("\nFusing LoRA into standalone checkpoint...", flush=True)
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

