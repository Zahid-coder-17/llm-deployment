import os
import sys
import json
import math
import types
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

EVAL_DIR = r"D:\embeddra_eval"
sys.path.insert(0, os.path.join(EVAL_DIR, "embeddra-edge"))
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

CKPT_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
TOK_PATH = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")
ALPACA_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\alpaca_data_cleaned.json"
UI_TEST_PATH = os.path.join(EVAL_DIR, "android_agent", "test_reasoning_agent.jsonl")

def compute_ppl(model, token_sequences, device):
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for full_ids, labels in token_sequences:
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100, reduction="sum")
            valid_toks = (y.view(-1) != -100).sum().item()
            if valid_toks > 0:
                total_nll += nll.item()
                total_tokens += valid_toks
    mean_loss = total_nll / max(1, total_tokens)
    ppl = math.exp(min(mean_loss, 20.0))
    return mean_loss, ppl, total_tokens

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Perplexity on {device} for:\n{CKPT_PATH}", flush=True)
    model = MariaForCausalLM.from_pretrained_checkpoint(CKPT_PATH, device=device, dtype=torch.bfloat16)
    model.eval()
    tokenizer = Tokenizer.from_file(TOK_PATH)

    # 1. Held-out Alpaca Chat / Instruction (unseen slice starting at index 5000)
    with open(ALPACA_PATH, "r", encoding="utf-8") as f:
        ds = json.load(f)
    alpaca_seqs = []
    for row in ds[5000:5500]:
        instr = row["instruction"].strip()
        inp = row["input"].strip()
        out = row["output"].strip()
        if not instr or not out or len(out) > 320:
            continue
        user_msg = f"{instr}\n{inp}".strip() if inp else instr
        p_ids = tokenizer.encode(f"<user>{user_msg}</user>\n<thought>").ids
        a_ids = tokenizer.encode(f"{out}</thought>").ids
        full_ids = p_ids + a_ids
        if len(full_ids) <= 256:
            labels = [-100] * len(p_ids) + a_ids
            alpaca_seqs.append((full_ids, labels))
        if len(alpaca_seqs) >= 50:
            break

    # 2. Held-out Android UI Executive Test Set
    ui_seqs = []
    with open(UI_TEST_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 50 or not line.strip():
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
                    ui_seqs.append((full_ids, labels))

    # 3. Raw English LM Perplexity (full sequence cross-entropy)
    prose_texts = [
        "The Solar System consists of the Sun and the objects that orbit it, including eight planets, their moons, and countless asteroids and comets.",
        "Modern smartphones combine multi-core ARM processors, dedicated neural accelerators, and high-speed memory to run artificial intelligence locally.",
        "Photosynthesis is the process used by plants and algae to convert light energy into chemical energy that fuels their growth."
    ]
    prose_seqs = []
    for txt in prose_texts:
        ids = tokenizer.encode(txt).ids
        prose_seqs.append((ids, ids))

    loss_chat, ppl_chat, tok_chat = compute_ppl(model, alpaca_seqs, device)
    loss_ui, ppl_ui, tok_ui = compute_ppl(model, ui_seqs, device)
    loss_prose, ppl_prose, tok_prose = compute_ppl(model, prose_seqs, device)

    print("\n" + "=" * 70)
    print(" PERPLEXITY EVALUATION REPORT (FUSED CHECKPOINT)")
    print("=" * 70)
    print(f" 1. Held-Out Chat / Instruction ({len(alpaca_seqs)} ex, {tok_chat} toks):")
    print(f"    Loss = {loss_chat:.4f} | Perplexity (PPL) = {ppl_chat:.2f}")
    print(f" 2. Held-Out Android UI Control ({len(ui_seqs)} ex, {tok_ui} toks):")
    print(f"    Loss = {loss_ui:.4f} | Perplexity (PPL) = {ppl_ui:.2f}")
    print(f" 3. Raw English Prose LM ({len(prose_seqs)} ex, {tok_prose} toks):")
    print(f"    Loss = {loss_prose:.4f} | Perplexity (PPL) = {ppl_prose:.2f}")
    print("=" * 70)

if __name__ == "__main__":
    main()

