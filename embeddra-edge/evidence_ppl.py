import os
import sys
import time
import math
import hashlib
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from hypergqe.model_adapter.maria_model import MariaForCausalLM

def get_file_sha256(filepath, max_mb=64):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        chunk = f.read(max_mb * 1024 * 1024)
        h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("STEP 1: HARDWARE & CHECKPOINT INTEGRITY EVIDENCE")
    print("=" * 80)
    
    ckpt_path = os.path.join(workspace_root, "checkpoints", "maria_1b_20B_weights.pt")
    tok_path = os.path.join(workspace_root, "checkpoints", "tokenizer.json")
    
    print(f"Checkpoint File: {ckpt_path}")
    print(f"  * File Size   : {os.path.getsize(ckpt_path):,} bytes ({os.path.getsize(ckpt_path)/(1024**3):.2f} GB)")
    print(f"  * Head SHA-256: {get_file_sha256(ckpt_path)} (first 64MB)")
    print(f"Tokenizer File : {tok_path}")
    print(f"  * File Size   : {os.path.getsize(tok_path):,} bytes")
    print(f"  * Head SHA-256: {get_file_sha256(tok_path)} (first 64MB)")
    
    print("\n" + "=" * 80)
    print("STEP 2: MODEL INITIALIZATION & STATE DICT VERIFICATION")
    print("=" * 80)
    t0 = time.time()
    tokenizer = Tokenizer.from_file(tok_path)
    model = MariaForCausalLM.from_pretrained_checkpoint(ckpt_path, device="cpu", dtype=torch.bfloat16)
    model.eval()
    print(f"Model successfully materialized in memory in {time.time() - t0:.2f}s")
    
    text = (
        "The human epidermal growth factor receptor 2 (HER2) is a member of the epidermal "
        "growth factor receptor family of receptor tyrosine kinases. Amplification or overexpression "
        "of this oncogene has been shown to play an important role in the development and progression "
        "of certain aggressive types of breast cancer. Trastuzumab is a monoclonal antibody that targets "
        "the extracellular domain of the HER2 protein, inhibiting cell proliferation and promoting "
        "antibody-dependent cellular cytotoxicity."
    )
    
    encoded = tokenizer.encode(text)
    token_ids = encoded.ids
    tokens_str = encoded.tokens
    N = len(token_ids)
    
    print("\n" + "=" * 80)
    print("STEP 3: TOKENIZATION OF THE TEST SEQUENCE (PubMed Sample)")
    print("=" * 80)
    print(f"Total Tokens: {N}")
    print("First 15 Tokens [Token ID -> String]:")
    for i in range(min(15, N)):
        print(f"  Position {i:2d}: Token ID {token_ids[i]:<6d} -> {repr(tokens_str[i])}")
        
    print("\n" + "=" * 80)
    print("STEP 4: FORWARD PASS & LOGIT TENSOR INSPECTION")
    print("=" * 80)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    
    with torch.no_grad():
        logits, _ = model(input_ids)
        
    print(f"Input Tensor Shape : {list(input_ids.shape)} (Batch=1, SeqLen={N})")
    print(f"Logits Tensor Shape: {list(logits.shape)} (Batch=1, SeqLen={N}, Vocab={logits.shape[-1]})")
    print(f"Logits Min: {logits.min().item():.4f}, Max: {logits.max().item():.4f}, Mean: {logits.mean().item():.4f}")
    
    print("\n" + "=" * 80)
    print("STEP 5: TOKEN-BY-TOKEN CROSS-ENTROPY LOSS EVIDENCE")
    print("=" * 80)
    print("Examining individual transition loss: -log P(token_{t+1} | tokens_{0..t})")
    print(f"{'Pos':<4} | {'Target Token':<18} | {'Target ID':<10} | {'Logit':<10} | {'Prob':<10} | {'NLL Loss':<10} | {'Top-1 Prediction':<18}")
    print("-" * 88)
    
    losses = []
    for t in range(min(12, N - 1)):
        # Logits at position t predict token at position t + 1
        pos_logits = logits[0, t, :].float()
        probs = F.softmax(pos_logits, dim=-1)
        
        target_id = token_ids[t + 1]
        target_str = repr(tokens_str[t + 1])
        target_logit = pos_logits[target_id].item()
        target_prob = probs[target_id].item()
        token_loss = -math.log(max(target_prob, 1e-12))
        losses.append(token_loss)
        
        top1_id = torch.argmax(pos_logits).item()
        top1_str = repr(tokenizer.decode([top1_id]))
        
        print(f"{t:<4d} | {target_str:<18} | {target_id:<10d} | {target_logit:<10.3f} | {target_prob:<10.5f} | {token_loss:<10.4f} | {top1_str:<18}")

    print("\n" + "=" * 80)
    print("STEP 6: FULL CAUSAL CROSS-ENTROPY CALCULATION")
    print("=" * 80)
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    
    ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
    loss_val = ce_loss.item()
    ppl = math.exp(loss_val)
    
    print(f"Total Predictable Token Transitions (N - 1): {N - 1}")
    print(f"Mean Cross-Entropy Loss (L_CE)             : {loss_val:.6f}")
    print(f"Exact Formula                              : Perplexity = exp({loss_val:.6f})")
    print(f"Calculated Perplexity (PPL)                : {ppl:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()

