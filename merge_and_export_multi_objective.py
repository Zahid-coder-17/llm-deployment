import os
import sys
import struct
import types
import torch
import torch.nn as nn

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
OUT_LORA = os.path.join(BASE_DIR, "lora_multi_objective.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")

def export_to_eer(sd, eer_path: str):
    print(f"\n[EXPORT] Converting state dict -> .eer binary: {eer_path}...", flush=True)
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

def main():
    print(f"Loading base checkpoint: {BASE_CKPT}...", flush=True)
    pt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = pt["model_state_dict"] if "model_state_dict" in pt else pt
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(dtype=torch.float32)
    model.load_state_dict(sd, strict=False)

    print(f"Loading best multi-objective LoRA checkpoint: {OUT_LORA}...", flush=True)
    lora_sd = torch.load(OUT_LORA, map_location="cpu", weights_only=False)

    scaling = 32.0 / 16.0 # alpha / r = 2.0
    merged_count = 0
    with torch.no_grad():
        for block_idx, block in enumerate(model.blocks):
            for attr in ["q_proj", "v_proj", "o_proj"]:
                mod = getattr(block.attn, attr)
                a_key = f"blocks.{block_idx}.attn.{attr}.lora_A"
                b_key = f"blocks.{block_idx}.attn.{attr}.lora_B"
                if a_key in lora_sd and b_key in lora_sd:
                    a = lora_sd[a_key].float()
                    b = lora_sd[b_key].float()
                    delta = (b @ a) * scaling
                    mod.weight.data.add_(delta.to(mod.weight.dtype))
                    merged_count += 1

            for attr in ["gate", "up", "down"]:
                mod = getattr(block.ffn, attr)
                a_key = f"blocks.{block_idx}.ffn.{attr}.lora_A"
                b_key = f"blocks.{block_idx}.ffn.{attr}.lora_B"
                if a_key in lora_sd and b_key in lora_sd:
                    a = lora_sd[a_key].float()
                    b = lora_sd[b_key].float()
                    delta = (b @ a) * scaling
                    mod.weight.data.add_(delta.to(mod.weight.dtype))
                    merged_count += 1

    print(f"[OK] Merged {merged_count} LoRA modules into base weights successfully!", flush=True)
    export_to_eer(model.state_dict(), OUT_EER)
    print("\n=== MERGE AND EXPORT COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
