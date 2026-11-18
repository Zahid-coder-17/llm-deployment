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
LORA_RESUME = os.path.join(BASE_DIR, "lora_final_polish_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_e2e_polish_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
PREV_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_final_polish.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")
OUT_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_e2e_polish.jsonl")


def build_dataset():
    random.seed(2099)
    items = []

    # 1. Exact YouTube Search screen with EditText "Search YouTube" + noisy suggestions -> TYPE("Search YouTube", ...)
    yt_queries = [
        ("find Agentic AI lecture and play it from youtube", "Agentic AI lecture"),
        ("search lo-fi beats on youtube and play it", "lo-fi beats"),
        ("find python tutorial on youtube and play it", "python tutorial"),
        ("search transformer architecture explained on youtube", "transformer architecture explained"),
    ]
    for user_q, search_q in yt_queries:
        scr_search = (
            '[SCREEN: com.google.android.youtube]\n'
            '[1] Button "Voice search"\n'
            '[2] Button "Navigate up"\n'
            '[3] EditText "Search YouTube"\n'
            '[6] TextView "sonali bendre karishma kapoor"\n'
            '[9] TextView "hyderabad diaries"\n'
            '[12] TextView "geeta kapoor dance"'
        )
        tgt_type = f'I will type "{search_q}" into "Search YouTube".</thought>\n<action>TYPE("Search YouTube", "{search_q}")</action>'
        for _ in range(30):
            items.append({"category": "ui_grounding", "prompt": f"<user>{user_q}</user>\n<screen>\n{scr_search}\n</screen>\n<thought>", "target": tgt_type})

        scr_results = (
            '[SCREEN: com.google.android.youtube]\n'
            f'[1] EditText "{search_q}"\n'
            '[2] Button "Clear"\n'
            f'[3] ViewGroup "{search_q} - Full Lecture"\n'
            f'[4] ViewGroup "{search_q} for Beginners"'
        )
        tgt_click = f'I will click "{search_q} - Full Lecture" to play it.</thought>\n<action>CLICK("{search_q} - Full Lecture")</action>'
        for _ in range(20):
            items.append({"category": "ui_grounding", "prompt": f"<user>{user_q}</user>\n<screen>\n{scr_results}\n</screen>\n<thought>", "target": tgt_click})

    # 2. Screen Summarize on YORI itself & real Android screens (NO <action>)
    summarize_examples = [
        (
            "summarize what is on my screen",
            '[SCREEN: com.embeddra.edge]\n[1] TextView "Y"\n[2] TextView "YORI"\n[3] TextView "OFFLINE"\n[4] Button "LLM Ready ✓ • Inspector: Active ✓"\n[5] Button "New Chat"\n[6] TextView "Explain recursion simply"\n[9] TextView "Tapped \'Logic\'"\n[10] TextView "take a screenshot"\n[13] TextView "📸 Took screenshot"',
            'You are on the YORI offline assistant screen. Previous messages show "Explain recursion simply" and "take a screenshot" (screenshot taken).</thought>'
        ),
        (
            "what is on my screen",
            '[SCREEN: com.embeddra.edge]\n[2] TextView "YORI"\n[3] TextView "OFFLINE"\n[4] Button "LLM Ready ✓ • Inspector: Active ✓"\n[6] TextView "Hello! I\'m Yori. What task would you like me to execute on your device today?"',
            'You are on the YORI home chat screen, showing that the offline LLM and Inspector are ready.</thought>'
        ),
        (
            "summarize what is on my screen",
            '[SCREEN: com.google.android.youtube]\n[2] ImageView "YouTube"\n[4] Button "Search"\n[6] ViewGroup "Shorts"\n[9] Button "Local tailor didi. – play Short"',
            'You are on the YouTube Home screen showing the Search button and Shorts feed.</thought>'
        ),
    ]
    for q, scr, ans in summarize_examples:
        for _ in range(30):
            items.append({"category": "chat", "prompt": f"<user>{q}</user>\n<screen>\n{scr}\n</screen>\n<thought>", "target": ans})

    # 3. Clean Pure Chat examples (explicitly terminating at </thought><|end_of_text|>)
    clean_chats = [
        ("Explain recursion simply", "Recursion is when a function calls itself with a smaller input until it reaches a stopping condition called the base case.</thought>"),
        ("Who are you?", "I am Yori, your private on-device AI assistant running locally on your phone.</thought>"),
        ("Why is the sky blue?", "Sunlight scatters in Earth's atmosphere, and shorter blue wavelengths scatter more than other colors, making the sky appear blue.</thought>"),
        ("Write a polite message saying I am running late", "Hi! I'm running a few minutes late and will be there as soon as possible. Thank you for your patience!</thought>"),
        ("Explain what a flashlight is", "A flashlight is a portable hand-held electric light powered by batteries or a rechargeable cell.</thought>"),
    ]
    for q, ans in clean_chats:
        for _ in range(25):
            items.append({"category": "chat", "prompt": f"<user>{q}</user>\n<thought>", "target": ans})

    # 4. Replay previous polish & smoke items to prevent regression
    if os.path.exists(PREV_TRAIN):
        with open(PREV_TRAIN, "r", encoding="utf-8") as f:
            prev_items = [json.loads(line) for line in f if line.strip()]
        random.shuffle(prev_items)
        items.extend(prev_items[:320])

    random.shuffle(items)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"Built E2E polish dataset: {len(items)} items -> {OUT_TRAIN}", flush=True)


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 32, lora_alpha: int = 64):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.scaling = lora_alpha / r
        in_f = original_linear.in_features
        out_f = original_linear.out_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f, dtype=torch.bfloat16) * (1.0 / math.sqrt(r)))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=torch.bfloat16))

    def forward(self, x):
        return self.original_linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


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
    print(f"=== YORI E2E POLISH (YT SEARCH + SUMMARIZE + CHAT EOS) ON {device} ===", flush=True)
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
    MAX_STEPS = 90
    WARMUP_STEPS = 6
    PEAK_LR = 4.0e-5
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

        if step % 30 == 0 or step == 1 or step == MAX_STEPS:
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

