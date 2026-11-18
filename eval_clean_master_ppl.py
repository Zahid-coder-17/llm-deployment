import os
import sys
import math
import json
import types
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

import verify_eer_smoke as v
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from train_pristine_unified_yori import build_pristine_dataset, SYSTEM_CONTRACT

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

EER_PATH = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== EVALUATING CLEAN MASTER MODEL PERPLEXITY ON {device} ===", flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    print(f"Loading weights from binary .eer: {EER_PATH}...", flush=True)
    sd = v.load_state_dict_from_eer(EER_PATH)
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float16)
    model.load_state_dict(sd, strict=False)
    model.eval()

    dataset = build_pristine_dataset()
    # Sample a balanced validation set of 100 items across all categories
    random = __import__("random")
    random.seed(999)
    val_samples = random.sample(dataset, 100)

    cat_stats = {}
    overall_nll = 0.0
    overall_tokens = 0

    with torch.no_grad():
        for item in val_samples:
            cat = item["category"]
            full = item["prompt"] + item["target"]
            full_ids = tokenizer.encode(full).ids
            prompt_ids = tokenizer.encode(item["prompt"]).ids

            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            p_len = len(prompt_ids) - 1
            labels[0, :p_len] = -100

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100, reduction="sum").item()

            ntoks = (labels != -100).sum().item()
            if cat not in cat_stats:
                cat_stats[cat] = [0.0, 0]
            cat_stats[cat][0] += nll
            cat_stats[cat][1] += ntoks

            overall_nll += nll
            overall_tokens += ntoks

    print("\n" + "=" * 75, flush=True)
    print("  CLEAN MASTER MODEL PERPLEXITY (PPL) EVALUATION RESULTS", flush=True)
    print("=" * 75, flush=True)

    for cat in ["chat", "draft_summary", "core_action", "black_learning_ui", "obstacle_recovery"]:
        if cat in cat_stats:
            total_nll, ntoks = cat_stats[cat]
            avg_nll = total_nll / max(1, ntoks)
            ppl = math.exp(min(avg_nll, 20.0))
            print(f"  * {cat.upper():20s} | NLL Loss: {avg_nll:.4f} | Perplexity (PPL): {ppl:6.2f} (tokens={ntoks})", flush=True)

    avg_overall_nll = overall_nll / max(1, overall_tokens)
    overall_ppl = math.exp(min(avg_overall_nll, 20.0))
    print("-" * 75, flush=True)
    print(f"  * OVERALL UNIFIED MODEL  | NLL Loss: {avg_overall_nll:.4f} | Perplexity (PPL): {overall_ppl:6.2f} (total tokens={overall_tokens})", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
