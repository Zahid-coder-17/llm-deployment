import os
import sys
import mmap
import struct
import types
import torch
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

EER_PATH = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")

def load_state_dict_from_eer(eer_path: str):
    f = open(eer_path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    magic, ver, vocab, hidden, inter, layers, heads, kv_heads, head_dim, max_seq, dtype_code, num_tensors = struct.unpack("<4sIIIIIIIIIII", mm[:48])
    assert magic == b"EER1", f"Invalid EER magic: {magic}"
    index_len = struct.unpack("<I", mm[48:52])[0]
    pos = 52
    end_idx = 52 + index_len

    sd = {}
    while pos < end_idx:
        name_len, ndim = struct.unpack("<HB", mm[pos:pos+3])
        pos += 3
        name = mm[pos:pos+name_len].decode("utf-8")
        pos += name_len
        shape = struct.unpack(f"<{ndim}I", mm[pos:pos+4*ndim])
        pos += 4 * ndim
        t_dtype, offset, size_bytes = struct.unpack("<BQQ", mm[pos:pos+17])
        pos += 17
        raw_bytes = mm[offset:offset+size_bytes]
        tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.float16).reshape(shape)
        sd[name] = tensor

    mm.close()
    f.close()
    print(f"[OK] Parsed .eer binary ({magic.decode()} v{ver}, {len(sd)} FP16 tensors, hidden={hidden}, layers={layers})", flush=True)
    return sd

def generate_from_eer(model, tokenizer, prompt: str, device, max_tokens=65, stop_on_thought=False):
    ids = tokenizer.encode(prompt).ids
    curr = torch.tensor([ids], dtype=torch.long, device=device)
    gen = []
    with torch.no_grad():
        for _ in range(max_tokens):
            logits, _ = model(curr)
            step_logits = logits[0, -1, :].clone().float()
            for prev in set(gen):
                if step_logits[prev] > 0: step_logits[prev] /= 1.15
                else: step_logits[prev] *= 1.15
            t = step_logits.argmax().item()
            gen.append(t)
            curr = torch.cat([curr, torch.tensor([[t]], device=device)], dim=1)
            dec = tokenizer.decode(gen)
            if "</action>" in dec:
                break
            if stop_on_thought and "</thought>" in dec:
                # Generate 4 more tokens to verify whether <action> is hallucinated after </thought>
                for _ in range(4):
                    logits2, _ = model(curr)
                    t2 = logits2[0, -1, :].argmax().item()
                    gen.append(t2)
                    curr = torch.cat([curr, torch.tensor([[t2]], device=device)], dim=1)
                break
    return tokenizer.decode(gen)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = load_state_dict_from_eer(EER_PATH)
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float16)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded .eer FP16 weights into MariaForCausalLM on {device} (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    model.eval()
    tokenizer = Tokenizer.from_file(TOK_PATH)

    # =========================================================================
    # SMOKE TEST 1: Pure Chat Termination Check
    # =========================================================================
    print("\n" + "=" * 75)
    print(" SMOKE TEST 1: PURE CHAT TERMINATION CHECK (FROM .eer BINARY)")
    print("=" * 75)
    chat_prompts = [
        "Who are you?",
        "Explain recursion simply",
        "Why is the sky blue?",
        "Write a polite message saying I am running late",
        "Explain what a flashlight is"
    ]
    clean_terminations = 0
    for q in chat_prompts:
        out = generate_from_eer(model, tokenizer, f"<user>{q}</user>\n<thought>", device, max_tokens=96, stop_on_thought=True)
        has_thought_end = "</thought>" in out
        hallucinated_action = "<action>" in out
        ok = has_thought_end and not hallucinated_action
        if ok: clean_terminations += 1
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} Q: {q}")
        print(f"       Out: {out.replace(chr(10), ' ')}")
    print(f"Chat Termination Result: {clean_terminations}/{len(chat_prompts)} terminated cleanly at </thought> without <action>", flush=True)

    # =========================================================================
    # SMOKE TEST 2: Multi-Turn UI Grounding Check ([1]..[12])
    # =========================================================================
    print("\n" + "=" * 75)
    print(" SMOKE TEST 2: MULTI-TURN UI GROUNDING CHECK ([1]..[12] FROM .eer BINARY)")
    print("=" * 75)
    ui_tests = [
        (
            "Turn on Bluetooth",
            '[SCREEN: com.android.settings.connections]\n[1] TextView "Connections"\n[2] Switch "Wi-Fi" (ON)\n[3] TextView "Wi-Fi calling"\n[4] Switch "NFC" (OFF)\n[5] Switch "Flight mode" (OFF)\n[6] TextView "SIM manager"\n[7] TextView "Mobile networks"\n[8] TextView "Data usage"\n[9] Switch "Bluetooth" (OFF)\n[10] TextView "More connection settings"',
            'CLICK("Bluetooth")'
        ),
        (
            "Enter 'Yori Lunch' as the alarm name",
            '[SCREEN: com.sec.android.app.clockpackage.edit]\n[1] Button "Cancel"\n[2] Button "Save"\n[3] TimePicker "Hour"\n[4] TimePicker "Minute"\n[5] CheckBox "Mon"\n[6] CheckBox "Tue"\n[7] CheckBox "Wed"\n[8] CheckBox "Thu"\n[9] CheckBox "Fri"\n[10] Switch "Alarm sound" (ON)\n[11] EditText "Alarm name" [Input]\n[12] Switch "Snooze" (ON)',
            'TYPE("Alarm name", "Yori Lunch")'
        ),
        (
            "Open Bilal's chat on WhatsApp",
            '[SCREEN: com.whatsapp]\n[1] TextView "WhatsApp"\n[2] Button "Camera"\n[3] Button "Search"\n[4] Button "More options"\n[5] Tab "Chats"\n[6] Tab "Updates"\n[7] Tab "Calls"\n[8] TextView "Pinned: Mom"\n[9] TextView "Tech Support"\n[10] TextView "Sarah"\n[11] TextView "Bilal"\n[12] Button "New chat"',
            'CLICK("Bilal")'
        )
    ]
    ui_passed = 0
    for intent, screen, expected in ui_tests:
        prompt = f"<user>{intent}</user>\n<screen>\n{screen}\n</screen>\n<thought>"
        out = generate_from_eer(model, tokenizer, prompt, device, max_tokens=55, stop_on_thought=False)
        pred_act = out.split("<action>")[1].split("</action>")[0].strip() if "<action>" in out else out.strip()
        ok = (pred_act == expected)
        if ok: ui_passed += 1
        tag = "[PASS]" if ok else "[FAIL]"
        print(f"{tag} Intent: {intent} | Expected: {expected} | Predicted: {pred_act}")
        print(f"       Thought: {out.split('</thought>')[0].strip()}")
    print(f"UI Grounding Result: {ui_passed}/{len(ui_tests)} matched dynamic screen layout!", flush=True)

if __name__ == "__main__":
    main()

