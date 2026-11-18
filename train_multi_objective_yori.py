import os
import sys
import json
import time
import math
import random
import struct
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
cache_dir = os.path.join(BASE_DIR, ".torch_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ["TORCH_DYNAMO_CACHE_DIR"] = cache_dir
os.environ["TRITON_CACHE_DIR"] = cache_dir

sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

BASE_CKPT = os.path.join(BASE_DIR, "checkpoints", "maria_1b_20B_weights.pt")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
DATASET_PATH = os.path.join(BASE_DIR, "yori_pristine_master_dataset.jsonl")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
OUT_LORA = os.path.join(BASE_DIR, "lora_multi_objective.pt")

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=16, lora_alpha=32):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, original_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, r))

    def forward(self, x):
        base = self.original_linear(x)
        lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base + lora.to(base.dtype)

def export_to_eer(pt_or_sd, eer_path: str):
    print(f"\n[EXPORT] Converting state dict -> .eer binary: {eer_path}...", flush=True)
    if isinstance(pt_or_sd, dict):
        sd = pt_or_sd
    else:
        sd = torch.load(pt_or_sd, map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]

    MAGIC = b"EER1"
    VERSION = 1
    VOCAB_SIZE = 128256
    HIDDEN_SIZE = 2048
    INTERMEDIATE = 5632
    NUM_LAYERS = 18
    NUM_HEADS = 16
    NUM_KV_HEADS = 4
    HEAD_DIM = 128
    MAX_SEQ_LEN = 8192
    DTYPE_FP16 = 1

    entries = []
    tensor_names = sorted(list(sd.keys()))
    for name in tensor_names:
        t = sd[name].detach().cpu().half().contiguous()
        shape = list(t.shape)
        data_bytes = t.numpy().tobytes()
        entries.append((name, shape, DTYPE_FP16, len(data_bytes), data_bytes))

    index_buf = bytearray()
    for name, shape, dtype_code, size_bytes, _ in entries:
        name_bytes = name.encode("utf-8")
        index_buf.extend(struct.pack("<HB", len(name_bytes), len(shape)))
        index_buf.extend(name_bytes)
        index_buf.extend(struct.pack(f"<{len(shape)}I", *shape))
        index_buf.extend(struct.pack("<BQQ", dtype_code, 0, size_bytes))

    header_len = 48 + 4
    index_len = len(index_buf)
    data_start = header_len + index_len

    final_index_buf = bytearray()
    blob_offset = data_start
    for name, shape, dtype_code, size_bytes, data_bytes in entries:
        name_bytes = name.encode("utf-8")
        final_index_buf.extend(struct.pack("<HB", len(name_bytes), len(shape)))
        final_index_buf.extend(name_bytes)
        final_index_buf.extend(struct.pack(f"<{len(shape)}I", *shape))
        final_index_buf.extend(struct.pack("<BQQ", dtype_code, blob_offset, size_bytes))
        blob_offset += size_bytes

    header = struct.pack(
        "<4sIIIIIIIIIII",
        MAGIC, VERSION, VOCAB_SIZE, HIDDEN_SIZE, INTERMEDIATE,
        NUM_LAYERS, NUM_HEADS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN,
        DTYPE_FP16, len(entries)
    )

    with open(eer_path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(final_index_buf)))
        f.write(final_index_buf)
        for _, _, _, _, data_bytes in entries:
            f.write(data_bytes)

    print(f"[OK] Exported {len(entries)} tensors to {eer_path} ({os.path.getsize(eer_path)/(1024*1024):.1f} MB)", flush=True)

def evaluate_split_perplexities(model, val_dataset, device):
    """Calculates chat PPL and tool PPL separately on held-out validation set."""
    model.eval()
    chat_nll, chat_toks = 0.0, 0
    tool_nll, tool_toks = 0.0, 0

    with torch.no_grad():
        for full_ids, p_len, is_chat in val_dataset:
            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
            labels[0, :p_len] = -100
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100, reduction="sum").item()
            ntoks = (labels != -100).sum().item()
            if is_chat:
                chat_nll += nll
                chat_toks += ntoks
            else:
                tool_nll += nll
                tool_toks += ntoks

    model.train()
    chat_loss = chat_nll / max(1, chat_toks)
    tool_loss = tool_nll / max(1, tool_toks)
    chat_ppl = math.exp(min(chat_loss, 20.0))
    tool_ppl = math.exp(min(tool_loss, 20.0))
    total_loss = (chat_nll + tool_nll) / max(1, chat_toks + tool_toks)
    total_ppl = math.exp(min(total_loss, 20.0))
    return chat_ppl, tool_ppl, total_loss, total_ppl

def evaluate_fast_behavior(model, tokenizer, device):
    """Fast behavioral probe checking 3 critical tasks."""
    model.eval()
    system_contract = (
        "<system>You are YORI, an on-device executive agent built on the Maria-1B architecture by Embeddra AI Research Lab.\n"
        "- For conversation, questions, advice, identity, or documents: answer directly inside <thought> with NO <action>.\n"
        "- For device operations and app workflows: inspect the visible <screen> and execution <history>, reason step-by-step in <thought>:\n"
        "  🎯 Goal: [Clarified user objective]\n"
        "  📋 Plan: [Numbered steps to achieve goal]\n"
        "  ⚡ Immediate Action: [Specific rationale grounded in visible screen and action history]\n"
        "- Then emit ONE action from:\n"
        "  OPEN_APP(appName)\n"
        "  OPEN_SETTINGS(settingName)\n"
        "  CLICK(elementLabel | [index])\n"
        "  TYPE(elementLabel | [index], text)\n"
        "  SCROLL(UP | DOWN | LEFT | RIGHT)\n"
        "  PRESS_KEY(BACK | HOME | ENTER)\n"
        "  TORCH(1 | 0)\n"
        "  VOLUME(UP | DOWN | MUTE)\n"
        "  SET_ALARM(HH:MM)\n"
        "  DONE(summary)\n</system>\n"
    )

    tests = [
        ("Give me 3 valuable pieces of life and career advice", False, None),
        ("Turn on the flashlight", True, "TORCH(1)"),
        ("Set an alarm for 7:30 AM", True, "SET_ALARM(\"07:30\")"),
    ]

    score = 0.0
    with torch.no_grad():
        for query, is_tool, expected_act in tests:
            prompt = f"{system_contract}<user>{query}</user>\n<thought>\n"
            ids = tokenizer.encode(prompt).ids
            curr = torch.tensor([ids], dtype=torch.long, device=device)
            gen = []
            for _ in range(60):
                logits = model(curr)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_tok = logits[0, -1, :].argmax().item()
                if next_tok in [128001, 128009]:
                    break
                gen.append(next_tok)
                curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)
                dec = tokenizer.decode(gen)
                if is_tool and "</action>" in dec:
                    break
                elif not is_tool and "</thought>" in dec:
                    break

            out = tokenizer.decode(gen)
            if not is_tool:
                # Must not leak action and must close thought
                if "<action>" not in out and "</thought>" in out:
                    score += 1.0
            else:
                if expected_act in out:
                    score += 1.0
                elif "<action>" in out and "</action>" in out:
                    score += 0.5

    model.train()
    return score / len(tests)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MULTI-OBJECTIVE BALANCED YORI TRAINING ON {device} ===", flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float32)

    print(f"Loading base checkpoint: {BASE_CKPT}...", flush=True)
    pt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = pt["model_state_dict"] if "model_state_dict" in pt else pt
    model.load_state_dict(sd, strict=False)

    for p in model.parameters():
        p.requires_grad = False

    print("Injecting LoRA adapters (r=16, alpha=32) into Attention and SwiGLU FFN...", flush=True)
    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, r=16, lora_alpha=32).to(device)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, r=16, lora_alpha=32).to(device)
        block.attn.o_proj = LoRALinear(block.attn.o_proj, r=16, lora_alpha=32).to(device)
        block.ffn.gate = LoRALinear(block.ffn.gate, r=16, lora_alpha=32).to(device)
        block.ffn.up = LoRALinear(block.ffn.up, r=16, lora_alpha=32).to(device)
        block.ffn.down = LoRALinear(block.ffn.down, r=16, lora_alpha=32).to(device)

    model.to(device=device)

    # Load Balanced Dataset
    print(f"Loading balanced pristine dataset from {DATASET_PATH}...", flush=True)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        raw_samples = [json.loads(line) for line in f]

    print(f"Total Unique Samples: {len(raw_samples)}", flush=True)

    # Pre-tokenize all samples
    print("Pre-tokenizing samples into token IDs...", flush=True)
    tokenized_all = []
    for item in raw_samples:
        full = item["prompt"] + item["target"]
        full_ids = tokenizer.encode(full).ids
        prompt_ids = tokenizer.encode(item["prompt"]).ids
        p_len = len(prompt_ids) - 1
        is_chat = item["category"] in ["advice", "identity", "doc_qa"]
        tokenized_all.append((full_ids, p_len, is_chat))

    # Stratified 85/15 Train / Validation Split
    random.seed(42)
    random.shuffle(tokenized_all)
    val_size = max(25, int(len(tokenized_all) * 0.15))
    val_data = tokenized_all[:val_size]
    train_data = tokenized_all[val_size:]
    print(f"Split: {len(train_data)} Training samples | {len(val_data)} Held-Out Validation samples", flush=True)

    init_chat_ppl, init_tool_ppl, init_val_loss, init_val_ppl = evaluate_split_perplexities(model, val_data, device)
    print(f"Initial Metrics -> Chat PPL: {init_chat_ppl:.2f} | Tool PPL: {init_tool_ppl:.2f} | Total Loss: {init_val_loss:.4f} (Total PPL: {init_val_ppl:.2f})", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1.1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    BATCH_SIZE = 4
    EPOCHS = 3
    model.train()

    t0 = time.time()
    step = 0
    total_steps = (len(train_data) // BATCH_SIZE) * EPOCHS

    best_score = -1.0
    best_metrics = {}

    print(f"\n--- Training Across {EPOCHS} Epochs ({total_steps} Steps) Targeting Optimal Multi-Objective ---", flush=True)

    for epoch in range(1, EPOCHS + 1):
        shuffled = list(train_data)
        random.shuffle(shuffled)
        for b_idx in range(0, len(shuffled) - BATCH_SIZE + 1, BATCH_SIZE):
            step += 1
            batch = shuffled[b_idx : b_idx + BATCH_SIZE]
            optimizer.zero_grad()
            loss_total = 0.0

            for full_ids, p_len, is_chat in batch:
                input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
                labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
                labels[0, :p_len] = -100

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(input_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    # Weight chat samples slightly to ensure conversational advice converges
                    weight = 1.35 if is_chat else 1.0
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100) * weight

                scaler.scale(loss / len(batch)).backward()
                loss_total += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            avg_loss = loss_total / len(batch)

            if step % 15 == 0 or step == total_steps:
                chat_ppl, tool_ppl, val_loss, val_ppl = evaluate_split_perplexities(model, val_data, device)
                probe_acc = evaluate_fast_behavior(model, tokenizer, device)

                # Multi-Objective Score:
                # 1. Probe behavioral accuracy (weight 0.45)
                # 2. Tool PPL in 2-4 target (weight 0.30)
                # 3. Chat PPL in 12-32 target (weight 0.25)
                tool_score = max(0.0, 1.0 - abs(tool_ppl - 3.0) / 4.0)
                chat_score = max(0.0, 1.0 - abs(chat_ppl - 20.0) / 25.0)
                composite_score = 0.45 * probe_acc + 0.30 * tool_score + 0.25 * chat_score

                print(f"Step {step:3d}/{total_steps} | TrLoss: {avg_loss:.4f} | ChatPPL: {chat_ppl:5.2f} | ToolPPL: {tool_ppl:5.2f} | ProbeAcc: {probe_acc*100:4.1f}% | Score: {composite_score:.3f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

                if composite_score > best_score:
                    best_score = composite_score
                    best_metrics = {
                        "step": step,
                        "chat_ppl": chat_ppl,
                        "tool_ppl": tool_ppl,
                        "val_loss": val_loss,
                        "probe_acc": probe_acc
                    }
                    lora_sd = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
                    torch.save(lora_sd, OUT_LORA)

    print(f"\n[OK] Training finished! Best Checkpoint at Step {best_metrics.get('step')} (Score: {best_score:.3f})", flush=True)
    print(f"  * Chat PPL: {best_metrics.get('chat_ppl'):.2f}")
    print(f"  * Tool PPL: {best_metrics.get('tool_ppl'):.2f}")
    print(f"  * Probe Accuracy: {best_metrics.get('probe_acc')*100:.1f}%")

    # Reload best sweet-spot weights
    print(f"\nReloading best checkpoint from {OUT_LORA}...", flush=True)
    best_sd = torch.load(OUT_LORA, map_location="cpu", weights_only=False)
    model.load_state_dict(best_sd, strict=False)

    # Merge LoRA weights into base weights
    print("\nMerging LoRA weights into base weights...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for attr in ["q_proj", "v_proj", "o_proj"]:
                mod = getattr(block.attn, attr)
                delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                mod.original_linear.weight.data.add_(delta.to(mod.original_linear.weight.dtype))
                setattr(block.attn, attr, mod.original_linear)
            for attr in ["gate", "up", "down"]:
                mod = getattr(block.ffn, attr)
                delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                mod.original_linear.weight.data.add_(delta.to(mod.original_linear.weight.dtype))
                setattr(block.ffn, attr, mod.original_linear)

    # Export directly to .eer binary
    export_to_eer(model.state_dict(), OUT_EER)
    print("\n=== MULTI-OBJECTIVE BALANCED YORI EXPORT COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
