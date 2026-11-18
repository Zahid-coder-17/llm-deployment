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

# Start from clean 20B base weights to eliminate prior ui_grounding memorization & chat forgetting
BASE_CKPT = os.path.join(EVAL_DIR, "checkpoints", "maria_1b_20B_weights.pt")
OUT_PT = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
OUT_EER = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.eer"
TOK_PATH = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")
TRAIN_JSONL = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_unified_train.jsonl"
VAL_JSONL = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\yori_unified_val.jsonl"

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
            if not line.strip():
                continue
            obj = json.loads(line)
            p_ids = tokenizer.encode(obj["prompt"]).ids
            t_ids = tokenizer.encode(obj["target"]).ids
            full_ids = p_ids + t_ids
            if 4 <= len(full_ids) <= max_len:
                # Strict loss masking: -100 for all input prompt tokens (<user>, <screen>, <thought>)
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
            if cat not in stats:
                stats[cat] = [0.0, 0]
            stats[cat][0] += nll
            stats[cat][1] += ntoks
    out = {}
    for cat, (nll_sum, tok_sum) in stats.items():
        loss = nll_sum / max(1, tok_sum)
        out[cat] = (loss, math.exp(min(loss, 20.0)), tok_sum)
    return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI V4 BALANCED LORA TRAINING (FROM CLEAN 20B BASE) ON {device} ===", flush=True)
    t0 = time.time()

    tokenizer = Tokenizer.from_file(TOK_PATH)
    train_items = load_tokenized(TRAIN_JSONL, tokenizer)
    val_items = load_tokenized(VAL_JSONL, tokenizer)
    print(f"Tokenized {len(train_items)} train examples and {len(val_items)} held-out val examples.", flush=True)

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=32, lora_alpha=64).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=32, lora_alpha=64).to(device))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable LoRA params (r=32): {sum(p.numel() for p in trainable_params) / 1e6:.2f}M", flush=True)

    GRAD_ACCUM = 8
    OPT_STEPS = 500  # 4,000 example passes (~1.27 epochs over 3,154 examples)
    WARMUP_STEPS = 40  # 8% warmup
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        prog = float(step - WARMUP_STEPS) / max(1, OPT_STEPS - WARMUP_STEPS)
        return max(0.10, 0.5 * (1.0 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pre_ppl = eval_val_ppl(model, val_items, device)
    print("\n--- BASELINE 20B HELD-OUT PERPLEXITY (STEP 0) ---")
    for c, (l, p, n) in pre_ppl.items():
        print(f"  {c:<14}: Loss={l:.4f} | PPL={p:.2f} ({n} toks)", flush=True)

    idx_pool = list(range(len(train_items)))
    random.shuffle(idx_pool)
    ptr = 0
    best_lora_state = None

    print(f"\nStarting {OPT_STEPS} optimizer steps (LR=1e-4, Warmup={WARMUP_STEPS}, GradAccum={GRAD_ACCUM})...", flush=True)
    for step in range(1, OPT_STEPS + 1):
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

        if step % 50 == 0 or step == 1:
            print(f"Step {step:3d}/{OPT_STEPS} | TrainLoss={step_loss:.4f} | LR={scheduler.get_last_lr()[0]:.6f} | Elapsed={time.time() - t0:.1f}s", flush=True)

        if step % 100 == 0:
            val_ppl = eval_val_ppl(model, val_items, device)
            c_ppl = val_ppl["chat"][1]
            t_ppl = val_ppl["tool_call"][1]
            u_ppl = val_ppl["ui_grounding"][1]
            print(f"  [VAL @ Step {step}] chat PPL={c_ppl:.2f} | tool_call PPL={t_ppl:.2f} | ui_grounding PPL={u_ppl:.2f}", flush=True)

            if u_ppl < 1.15:
                print(f"  [WARN @ Step {step}] ui_grounding PPL ({u_ppl:.2f}) < 1.15 (risk of memorization). Stopping early with best prior LoRA!", flush=True)
                if best_lora_state is None:
                    best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                break

            best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
            torch.save(best_lora_state, r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\lora_v4_best.pt")

            # Stop early if all 3 targets hit the sweet spot
            if c_ppl <= 8.0 and t_ppl <= 2.5 and 1.15 <= u_ppl <= 1.80:
                print(f"  [EARLY STOP] All target PPL thresholds achieved at Step {step}!", flush=True)
                break

    if best_lora_state is not None:
        model.load_state_dict(best_lora_state, strict=False)

    # Sample <action> & chat generations on unseen validation prompts before fusing
    model.eval()
    print("\n--- SAMPLE GENERATIONS ON UNSEEN VAL PROMPTS (PRE-FUSION) ---", flush=True)
    tool_val = [x for x in val_items if x[0] == "tool_call"][:8]
    with torch.no_grad():
        for _, _, _, prompt_str, exp_target in tool_val:
            p_ids = tokenizer.encode(prompt_str).ids
            curr = torch.tensor([p_ids], dtype=torch.long, device=device)
            gen = []
            for _ in range(45):
                logits, _ = model(curr)
                t = logits[0, -1, :].argmax().item()
                gen.append(t)
                curr = torch.cat([curr, torch.tensor([[t]], device=device)], dim=1)
                if "</action>" in tokenizer.decode(gen):
                    break
            pred = tokenizer.decode(gen).replace("\n", " ").strip()
            exp_act = exp_target.split("<action>")[1].split("</action>")[0]
            print(f"  Prompt: {prompt_str.splitlines()[0]} | Exp: {exp_act} | Gen: {pred}", flush=True)

    # Fuse & Save
    print("\nFusing LoRA weights into standalone checkpoint...", flush=True)
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

