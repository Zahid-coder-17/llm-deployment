import sys
import os
import types
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = MariaConfig()
model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float32)
pt = torch.load("checkpoints/maria_1b_20B_weights.pt", map_location="cpu", weights_only=False)
sd = pt["model_state_dict"] if "model_state_dict" in pt else pt
model.load_state_dict(sd, strict=False)

lora_sd = torch.load("lora_pristine_master.pt", map_location="cpu", weights_only=False)
with torch.no_grad():
    for block_idx, block in enumerate(model.blocks):
        for attr in ["q_proj", "v_proj", "o_proj"]:
            a = lora_sd.get(f"blocks.{block_idx}.attn.{attr}.lora_A")
            b = lora_sd.get(f"blocks.{block_idx}.attn.{attr}.lora_B")
            if a is not None and b is not None:
                orig = getattr(block.attn, attr)
                orig.weight.data.add_(((b @ a) * 2.0).to(device=orig.weight.device, dtype=orig.weight.dtype))
        for attr in ["gate", "up", "down"]:
            a = lora_sd.get(f"blocks.{block_idx}.ffn.{attr}.lora_A")
            b = lora_sd.get(f"blocks.{block_idx}.ffn.{attr}.lora_B")
            if a is not None and b is not None:
                orig = getattr(block.ffn, attr)
                orig.weight.data.add_(((b @ a) * 2.0).to(device=orig.weight.device, dtype=orig.weight.dtype))

model.eval()
tok = Tokenizer.from_file("checkpoints/tokenizer.json")

SYSTEM_CONTRACT = (
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

def run_inference(query: str, screen: str = None, history: str = None, is_action: bool = False):
    prompt = SYSTEM_CONTRACT
    if history:
        prompt += f"<history>\n{history.strip()}\n</history>\n"
    prompt += f"<user>{query.strip()}</user>\n"
    if screen:
        prompt += f"<screen>\n{screen.strip()}\n</screen>\n"
    prompt += "<thought>\n"

    ids = tok.encode(prompt).ids
    curr = torch.tensor([ids], dtype=torch.long, device=device)
    gen = []

    for _ in range(250):
        with torch.no_grad():
            logits = model(curr)
            if isinstance(logits, tuple):
                logits = logits[0]
            next_logits = logits[0, -1, :].clone()

            if is_action:
                next_tok = next_logits.argmax().item()
            else:
                # Selective repetition penalty (exempt whitespace, newlines, digits, punctuation)
                for tok_id in set(gen[-32:]):
                    if tok_id in [198, 220, 13] or (15 <= tok_id <= 25):
                        continue
                    if next_logits[tok_id] > 0:
                        next_logits[tok_id] /= 1.05
                    else:
                        next_logits[tok_id] *= 1.05
                probs = F.softmax(next_logits / 0.7, dim=-1)
                next_tok = torch.multinomial(probs, 1).item()

        if next_tok in [128001, 128009]:
            break
        gen.append(next_tok)
        curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)
        full = tok.decode(gen)
        if is_action and "</action>" in full:
            break
        elif not is_action and "</thought>" in full:
            break

    return tok.decode(gen)

print("=================================================================", flush=True)
print("  EVALUATING PRISTINE YORI (EMBEDDRA AI RESEARCH LAB) ON CUDA", flush=True)
print("=================================================================", flush=True)

# 1. Identity Check
print("\n--- TEST 1: IDENTITY CHECK ---", flush=True)
print("Query: 'Who are you?'", flush=True)
out_id = run_inference("Who are you?", is_action=False)
print(out_id, flush=True)

# 2. Prompt 1: 3 Pieces of Advice
print("\n--- TEST 2: PROMPT 1 (ADVICE GENERATION) ---", flush=True)
print("Query: 'Give me 3 valuable pieces of life and career advice'", flush=True)
out_p1 = run_inference("Give me 3 valuable pieces of life and career advice", is_action=False)
print(out_p1, flush=True)

# 3. Prompt 2: Multi-Step Trajectory Across 5 Turns
print("\n--- TEST 3: PROMPT 2 (MULTI-STEP CLOSED-LOOP UI WITH HISTORY) ---", flush=True)
q_ms = "Summarize message from Bilal on WhatsApp and reply"

# Turn 1
print("\n[TURN 1: Initial user request, app not in foreground]", flush=True)
out_t1 = run_inference(q_ms, screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "WhatsApp"', history=None, is_action=True)
print(out_t1, flush=True)

# Turn 2
print("\n[TURN 2: Chat list visible]", flush=True)
hist2 = 'Step 1: OPEN_APP("WhatsApp")'
screen2 = '[SCREEN: com.whatsapp.w4b]\n[1] TextView "WhatsApp"\n[2] TextView "Bilal: Are we meeting at 4 PM?"\n[3] TextView "Mom: Call me when free"'
out_t2 = run_inference(q_ms, screen=screen2, history=hist2, is_action=True)
print(out_t2, flush=True)

# Turn 3
print("\n[TURN 3: Inside Bilal thread]", flush=True)
hist3 = 'Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal: Are we meeting at 4 PM?")'
screen3 = '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Bilal"\n[2] TextView "Are we meeting at 4 PM?"\n[3] EditText "Message" [Input]'
out_t3 = run_inference(q_ms, screen=screen3, history=hist3, is_action=True)
print(out_t3, flush=True)

# Turn 4
print("\n[TURN 4: Reply typed, Send button visible]", flush=True)
hist4 = 'Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal") -> Step 3: TYPE("Message", "Yes, see you at 4 PM")'
screen4 = '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, see you at 4 PM" [Typed]\n[2] Button "Send"'
out_t4 = run_inference(q_ms, screen=screen4, history=hist4, is_action=True)
print(out_t4, flush=True)

# Turn 5
print("\n[TURN 5: Delivery confirmed]", flush=True)
hist5 = 'Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal") -> Step 3: TYPE(...) -> Step 4: CLICK("Send")'
screen5 = '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, see you at 4 PM"\n[2] ImageView "Delivered"'
out_t5 = run_inference(q_ms, screen=screen5, history=hist5, is_action=True)
print(out_t5, flush=True)
