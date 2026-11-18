import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import math
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.append(os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.append(os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer
import train_master_universal_agent as t

BASE_CKPT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== POLISHING MASTER UNIVERSAL AGENT ON {device} ===", flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)

    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, t.LoRALinear(getattr(block.attn, attr), r=32, lora_alpha=64).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, t.LoRALinear(getattr(block.ffn, attr), r=32, lora_alpha=64).to(device))

    train_samples = t.build_universal_training_samples()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    MAX_STEPS = 150
    WARMUP_STEPS = 5
    PEAK_LR = 2.0e-5
    MIN_LR = 1.0e-6
    GRAD_ACCUM = 2

    optimizer = torch.optim.AdamW(trainable_params, lr=PEAK_LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        prog = float(step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return (MIN_LR / PEAK_LR) + (1.0 - (MIN_LR / PEAK_LR)) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    tokenized_samples = []
    for item in train_samples:
        p_ids = tokenizer.encode(item["prompt"]).ids
        t_ids = tokenizer.encode(item["target"]).ids + [128001]
        full_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        tokenized_samples.append((full_ids, labels))

    idx_pool = list(range(len(tokenized_samples)))
    random.shuffle(idx_pool)
    ptr = 0
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(GRAD_ACCUM):
            if ptr >= len(idx_pool):
                random.shuffle(idx_pool)
                ptr = 0
            full_ids, labels = tokenized_samples[idx_pool[ptr]]
            ptr += 1

            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)

            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100) / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item() * GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        avg_loss = step_loss / GRAD_ACCUM
        if step % 15 == 0 or step == 1 or step == MAX_STEPS:
            print(f"Polish Step {step:3d}/{MAX_STEPS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {time.time() - t0:.1f}s", flush=True)

    print("Merging polish LoRA into FP32...", flush=True)
    model.eval()
    with torch.no_grad():
        for block in model.blocks:
            for mod, attrs in [(block.attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (block.ffn, ["gate", "up", "down"])]:
                for attr in attrs:
                    layer = getattr(mod, attr)
                    if isinstance(layer, t.LoRALinear):
                        delta = (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
                        fused = (layer.original_linear.weight.float() + delta).to(layer.original_linear.weight.dtype)
                        layer.original_linear.weight.copy_(fused)
                        setattr(mod, attr, layer.original_linear)

    print(f"Saving merged model to {OUT_PT}...", flush=True)
    fixed_sd = {}
    for k, v in model.state_dict().items():
        if "lora_" in k: continue
        clean_key = k.replace(".original_linear.", ".")
        fixed_sd[clean_key] = v.cpu()
    torch.save(fixed_sd, OUT_PT)

    print(f"Serializing to {OUT_EER}...", flush=True)
    serialize_checkpoint_to_eer(OUT_PT, OUT_EER)
    print("Polish complete!", flush=True)

if __name__ == "__main__":
    main()
