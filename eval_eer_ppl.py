import os
import sys
import math
import json
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

import verify_eer_smoke as v

EER_PATH = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")

def load_val_items(path, tokenizer, max_len=1024):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== EVALUATING PPL ON .EER RUNTIME ON {device} ===", flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    val_items = load_val_items(VAL_JSONL, tokenizer)
    print(f"Loaded {len(val_items)} validation samples.", flush=True)

    sd = v.load_state_dict_from_eer(EER_PATH)
    cfg = v.MariaConfig()
    model = v.MariaForCausalLM(cfg).to(device=device, dtype=torch.float16)
    model.load_state_dict(sd, strict=False)
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

    print("\n" + "=" * 60, flush=True)
    print("  PERPLEXITY (PPL) EVALUATION RESULTS ON .EER BINARY", flush=True)
    print("=" * 60, flush=True)
    for cat in ["chat", "tool_call", "ui_grounding"]:
        if cat in stats:
            s = stats[cat]
            avg_nll = s[0] / max(1, s[1])
            ppl = math.exp(min(avg_nll, 20.0))
            print(f"  * {cat.upper():12s} | NLL Loss: {avg_nll:.4f} | Perplexity (PPL): {ppl:.2f} (tokens={s[1]})", flush=True)
    print("=" * 60, flush=True)

if __name__ == "__main__":
    main()
