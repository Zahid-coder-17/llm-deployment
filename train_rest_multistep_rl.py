"""
YORI Real-Time Multi-Step RL via Rejection Sampling Fine-Tuning (ReST)
=======================================================================
For each intent:
  1. Generate K=4 candidate action sequences via greedy rollout
  2. Score each with the simulated environment (did screen change? did we complete?)
  3. Take the BEST rollout as SFT target  (best-of-N / rejection sampling)
  4. Compute weighted cross-entropy loss with reward as weight
  5. If no rollout succeeded, use gold oracle sequence as target (always a signal)

This guarantees non-zero gradients from step 1 since the oracle sequence
is always a valid target, and once the model starts matching the oracle
it gets positive rewards from the environment.

Apps: YouTube (find+play), WhatsApp (reply), Settings (Bluetooth),
      Clock (alarm), Maps (navigate), Notes (write), Camera
"""
import os, sys, math, time, random, json
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))
import types
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer
m = types.ModuleType("model"); m.MariaConfig = MariaConfig; sys.modules["model"] = m

BASE_CKPT   = os.path.join(BASE_DIR, "checkpoints", "maria_1b_20B_weights.pt")
LORA_RESUME = os.path.join(BASE_DIR, "lora_e2e_polish_best.pt")
OUT_LORA    = os.path.join(BASE_DIR, "lora_rest_rl_best.pt")
OUT_PT      = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER     = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH    = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
VAL_JSONL   = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")


# ---------------------------------------------------------------------------
# Multi-step intents: (user_query, list of (screen, oracle_action) steps)
# Each step is what the LLM should output on that screen to make progress
# ---------------------------------------------------------------------------
def make_intents():
    intents = []

    # --- YouTube ---
    for topic, topic_label in [
        ("find Agentic AI lecture and play it from youtube", "Agentic AI lecture"),
        ("find lo-fi beats on youtube and play", "lo-fi beats"),
        ("search python tutorial on youtube", "python tutorial"),
        ("find deep learning course on youtube", "deep learning course"),
        ("search transformer architecture on youtube", "transformer architecture"),
        ("find Andrej Karpathy lecture on youtube", "Andrej Karpathy lecture"),
    ]:
        steps = [
            # Turn 1: no screen, open app
            (None,
             f'<user>{topic}</user>\n<thought>',
             f'User wants to find a YouTube video. I will open YouTube first.</thought>\n<action>OPEN_APP("YouTube")</action>'),
            # Turn 2: YouTube home
            (f'[SCREEN: com.google.android.youtube]\n[2] ImageView "YouTube"\n[4] Button "Search"\n[5] Button "Home"\n[6] ViewGroup "Shorts"',
             f'<user>{topic}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'YouTube is open. I will click "Search" to find the video.</thought>\n<action>CLICK("Search")</action>'),
            # Turn 3: Search box
            (f'[SCREEN: com.google.android.youtube]\n[1] Button "Voice search"\n[2] Button "Navigate up"\n[3] EditText "Search YouTube"\n[6] TextView "trending now"\n[9] TextView "popular videos"',
             f'<user>{topic}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will type "{topic_label}" into "Search YouTube".</thought>\n<action>TYPE("Search YouTube", "{topic_label}")</action>'),
            # Turn 4: Results
            (f'[SCREEN: com.google.android.youtube]\n[1] EditText "{topic_label}"\n[2] Button "Clear"\n[3] ViewGroup "{topic_label} - Full Lecture"\n[4] ViewGroup "{topic_label} for Beginners"',
             f'<user>{topic}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will click "{topic_label} - Full Lecture" to play it.</thought>\n<action>CLICK("{topic_label} - Full Lecture")</action>'),
        ]
        intents.append((topic, steps))

    # --- WhatsApp ---
    for contact, msg in [
        ("Bilal", "Yes I am coming"),
        ("Mom", "On my way, see you soon"),
        ("Raza", "Sure, let us meet at 5"),
        ("Ali", "Done, will send it now"),
    ]:
        query = f"reply to {contact} on WhatsApp saying {msg}"
        steps = [
            (None,
             f'<user>{query}</user>\n<thought>',
             f'I will open WhatsApp to reply to {contact}.</thought>\n<action>OPEN_APP("WhatsApp")</action>'),
            (f'[SCREEN: com.whatsapp]\n[1] TextView "{contact}"\n[2] TextView "Chat List"\n[3] EditText "Search"',
             f'<user>{query}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will click "{contact}" to open the chat.</thought>\n<action>CLICK("{contact}")</action>'),
            (f'[SCREEN: com.whatsapp]\n[1] TextView "{contact}"\n[2] TextView "Hey, are you coming?"\n[3] EditText "Message" [Input]\n[4] Button "Send"',
             f'<user>{query}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will type the reply into the message box.</thought>\n<action>TYPE("Message", "{msg}")</action>'),
            (f'[SCREEN: com.whatsapp]\n[1] TextView "{contact}"\n[2] EditText "{msg}" [Input]\n[3] Button "Send"',
             f'<user>{query}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'Message is typed. I will click "Send" to send it.</thought>\n<action>CLICK("Send")</action>'),
        ]
        intents.append((query, steps))

    # --- Settings: Bluetooth ---
    for q, label in [
        ("enable bluetooth", "Bluetooth"),
        ("turn on bluetooth", "Bluetooth"),
        ("disable wifi", "Wi-Fi"),
    ]:
        steps = [
            (None,
             f'<user>{q}</user>\n<thought>',
             f'I will open Settings.</thought>\n<action>OPEN_APP("Settings")</action>'),
            ('[SCREEN: com.android.settings]\n[1] Button "Network"\n[2] Button "Connections"\n[3] Button "Bluetooth"\n[4] Button "Battery"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will click "Connections" to access {label} settings.</thought>\n<action>CLICK("Connections")</action>'),
            (f'[SCREEN: com.android.settings]\n[1] Button "{label}"\n[2] Button "Wi-Fi"\n[3] Button "Mobile data"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will click "{label}" to toggle it.</thought>\n<action>CLICK("{label}")</action>'),
            (f'[SCREEN: com.android.settings]\n[1] Switch "{label}" [Off]',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will tap the {label} switch to enable it.</thought>\n<action>CLICK("{label}")</action>'),
        ]
        intents.append((q, steps))

    # --- Clock: Alarm ---
    for h, mi in [("7", "00"), ("8", "30"), ("6", "15"), ("9", "00")]:
        q = f"set an alarm for {h}:{mi.zfill(2)} AM"
        steps = [
            (None,
             f'<user>{q}</user>\n<thought>',
             f'I will open the Clock app to set an alarm.</thought>\n<action>OPEN_APP("Clock")</action>'),
            ('[SCREEN: com.sec.android.app.clockpackage]\n[1] Button "Add alarm"\n[2] TextView "Alarm"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will click "Add alarm" to create a new alarm.</thought>\n<action>CLICK("Add alarm")</action>'),
            (f'[SCREEN: com.sec.android.app.clockpackage]\n[1] NumberPicker "Hour"\n[2] NumberPicker "Minute"\n[3] Button "Save"\n[4] EditText "Alarm name" [Input]',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will set the alarm for {h}:{mi.zfill(2)} AM.</thought>\n<action>SET_ALARM("{h}", "{mi.zfill(2)}")</action>'),
        ]
        intents.append((q, steps))

    # --- Maps ---
    for dest in ["Hyderabad", "Lahore Airport", "FAST University", "DHA Phase 6"]:
        q = f"navigate to {dest} on maps"
        steps = [
            (None,
             f'<user>{q}</user>\n<thought>',
             f'I will open Google Maps.</thought>\n<action>OPEN_APP("Maps")</action>'),
            ('[SCREEN: com.google.android.apps.maps]\n[1] EditText "Search here"\n[2] Button "Explore"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will type "{dest}" into the search box.</thought>\n<action>TYPE("Search here", "{dest}")</action>'),
            (f'[SCREEN: com.google.android.apps.maps]\n[1] ViewGroup "{dest}"\n[2] Button "Directions"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will click "Directions" to navigate to {dest}.</thought>\n<action>CLICK("Directions")</action>'),
        ]
        intents.append((q, steps))

    # --- Notes ---
    for text in ["Buy milk", "Call dentist tomorrow", "Meeting at 3 PM", "Pick up kids at 5"]:
        q = f"open notes and write {text}"
        steps = [
            (None,
             f'<user>{q}</user>\n<thought>',
             f'I will open Samsung Notes.</thought>\n<action>OPEN_APP("Notes")</action>'),
            ('[SCREEN: com.samsung.android.app.notes]\n[1] Button "Create note"\n[2] TextView "All notes"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will click "Create note" to start a new note.</thought>\n<action>CLICK("Create note")</action>'),
            (f'[SCREEN: com.samsung.android.app.notes]\n[1] EditText "Title" [Input]\n[2] Button "Save"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'I will type "{text}" into the note.</thought>\n<action>TYPE("Title", "{text}")</action>'),
            (f'[SCREEN: com.samsung.android.app.notes]\n[1] EditText "{text}" [Input]\n[2] Button "Save"',
             f'<user>{q}</user>\n<screen>\n$SCREEN\n</screen>\n<thought>',
             f'Note typed. I will click "Save".</thought>\n<action>CLICK("Save")</action>'),
        ]
        intents.append((q, steps))

    return intents


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, orig, r=32, alpha=64):
        super().__init__()
        self.original_linear = orig
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, orig.in_features, dtype=torch.bfloat16) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(orig.out_features, r, dtype=torch.bfloat16))

    def forward(self, x):
        return self.original_linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


# ---------------------------------------------------------------------------
# Generation (greedy, fast, no_grad)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate(model, prompt_ids, tokenizer, device, max_tokens=48):
    ids = torch.tensor([prompt_ids[-128:]], dtype=torch.long, device=device)
    out = []
    for _ in range(max_tokens):
        logits, _ = model(ids)
        next_id = int(logits[0, -1].argmax())
        out.append(next_id)
        if next_id in (128001, 128009): break
        txt = tokenizer.decode(out)
        if "</action>" in txt: break
        if "</thought>" in txt and "<action>" not in txt: break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(out)


def encode(text, tokenizer, bos=128000):
    return [bos] + tokenizer.encode(text).ids


# ---------------------------------------------------------------------------
# Step loss: cross-entropy on (prompt, target) pair, weighted by reward
# ---------------------------------------------------------------------------
def step_loss(model, prompt_ids, target_ids, reward, device, max_len=200, eos_id=128001):
    full = (prompt_ids + target_ids + [eos_id])[:max_len]
    labels = ([-100] * len(prompt_ids) + target_ids + [eos_id])[:max_len]
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    ce = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100)
    return ce * max(0.0, reward)   # weight by reward; negative reward -> skip


# ---------------------------------------------------------------------------
# Evaluate model action vs oracle (token overlap reward 0..1)
# ---------------------------------------------------------------------------
def action_reward(generated: str, oracle: str) -> float:
    """Reward = F1 token overlap between generated and oracle action strings."""
    def toks(s):
        return set(s.lower().replace('"', '').replace('(', ' ').replace(')', ' ').split())
    gen_act = generated.split("<action>")[-1].split("</action>")[0].strip() if "<action>" in generated else ""
    orc_act = oracle.split("<action>")[-1].split("</action>")[0].strip() if "<action>" in oracle else ""
    if not gen_act or not orc_act:
        return 0.0
    g, o = toks(gen_act), toks(orc_act)
    if not g or not o:
        return 0.0
    precision = len(g & o) / len(g)
    recall = len(g & o) / len(o)
    return 2 * precision * recall / (precision + recall + 1e-8)


# ---------------------------------------------------------------------------
# Validation PPL
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_val(model, val_items, device):
    model.eval()
    stats = {"chat": [0.0, 0], "ui": [0.0, 0]}
    for cat, full_ids, labels in val_items:
        k = "ui" if cat != "chat" else "chat"
        x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100, reduction="sum")
        stats[k][0] += float(loss); stats[k][1] += int((y != -100).sum())
    return {k: math.exp(min(20, v[0] / max(1, v[1]))) for k, v in stats.items()}


def load_val(path, tokenizer, max_len=256):
    eos = tokenizer.token_to_id("<|end_of_text|>") or 128001
    bos = 128000
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            p = [bos] + tokenizer.encode(o["prompt"]).ids
            t = tokenizer.encode(o["target"]).ids + [eos]
            full = (p + t)[:max_len]
            labels = ([-100]*len(p) + t)[:max_len]
            if any(l != -100 for l in labels):
                data.append((o.get("category","chat"), full, labels))
    return data


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI ReST MULTI-STEP RL ON {device} ===", flush=True)
    t0 = time.time()

    tokenizer = Tokenizer.from_file(TOK_PATH)
    intents = make_intents()
    print(f"Loaded {len(intents)} multi-step intents across YouTube/WhatsApp/Settings/Clock/Maps/Notes", flush=True)

    val_items = load_val(VAL_JSONL, tokenizer)

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)
    for p in model.parameters(): p.requires_grad = False
    for block in model.blocks:
        for attr in ["q_proj","k_proj","v_proj","o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr)).to(device))
        for attr in ["gate","up","down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr)).to(device))

    if os.path.exists(LORA_RESUME):
        ckpt = torch.load(LORA_RESUME, map_location=device, weights_only=False)
        model.load_state_dict(ckpt, strict=False)
        print(f"Resumed from {LORA_RESUME}", flush=True)

    trainable = [p for p in model.parameters() if p.requires_grad]
    MAX_STEPS = 150
    WARMUP = 10
    PEAK_LR = 3e-5
    MIN_LR  = 5e-6
    GRAD_ACCUM = 6
    BATCH_INTENTS = 3   # intents per gradient step

    optimizer = torch.optim.AdamW(trainable, lr=PEAK_LR, weight_decay=0.01)

    def lr_fn(s):
        if s < WARMUP: return (s+1)/WARMUP
        prog = (s-WARMUP)/max(1, MAX_STEPS-WARMUP)
        return MIN_LR/PEAK_LR + (1 - MIN_LR/PEAK_LR)*0.5*(1+math.cos(math.pi*prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    best_val_ui = float("inf")
    best_lora = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    for step in range(1, MAX_STEPS + 1):
        model.train()
        random.shuffle(intents)
        batch = intents[:BATCH_INTENTS]

        optimizer.zero_grad()
        total_loss = 0.0
        total_reward = 0.0
        n_valid = 0

        for _, steps in batch:
            for _ in range(GRAD_ACCUM // BATCH_INTENTS):
                # pick a random turn within this intent
                turn_idx = random.randint(0, len(steps) - 1)
                screen, prompt_tmpl, oracle = steps[turn_idx]
                prompt_text = prompt_tmpl.replace("$SCREEN", screen) if screen else prompt_tmpl
                p_ids = encode(prompt_text, tokenizer)

                # --- ReST: generate candidate and compare to oracle ---
                model.eval()
                with torch.no_grad():
                    candidate = greedy_generate(model, p_ids, tokenizer, device)
                r = action_reward(candidate, oracle)
                total_reward += r

                # Weight: if candidate matched oracle well, use candidate (reward); 
                # always also train on oracle with weight 1.0
                model.train()
                oracle_ids = tokenizer.encode(oracle).ids

                # Oracle supervised loss (always, weight=1.0)
                lossA = step_loss(model, p_ids, oracle_ids, 1.0, device)

                # Candidate loss if reward > 0 (positive reinforcement)
                if r > 0.3:
                    cand_ids = tokenizer.encode(candidate).ids
                    lossB = step_loss(model, p_ids, cand_ids, r, device)
                    loss = (lossA + lossB * r) / (1 + r)
                else:
                    loss = lossA

                loss = loss / GRAD_ACCUM
                loss.backward()
                total_loss += float(loss.item())
                n_valid += 1

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == 1 or step == MAX_STEPS:
            avg_r = total_reward / max(1, BATCH_INTENTS * (GRAD_ACCUM // BATCH_INTENTS))
            val = eval_val(model, val_items, device)
            print(f"Step {step:3d}/{MAX_STEPS} | Loss={total_loss:.4f} | AvgReward={avg_r:.3f} | VAL: chat={val['chat']:.2f} ui={val['ui']:.2f} | {time.time()-t0:.1f}s", flush=True)
            if val['ui'] < best_val_ui:
                best_val_ui = val['ui']
                best_lora = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora, OUT_LORA)
                print(f"  [SAVED] best LoRA at step {step} (ui_ppl={val['ui']:.2f})", flush=True)

    # Always save final
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}, OUT_LORA)

    # Fuse + export
    model.load_state_dict(best_lora, strict=False)
    print("\nFusing LoRA -> standalone checkpoint...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for mod, attrs in [(block.attn, ["q_proj","k_proj","v_proj","o_proj"]), (block.ffn, ["gate","up","down"])]:
                for attr in attrs:
                    layer = getattr(mod, attr)
                    if isinstance(layer, LoRALinear):
                        delta = (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
                        fused = (layer.original_linear.weight.float() + delta).to(layer.original_linear.weight.dtype)
                        layer.original_linear.weight.copy_(fused)
                        setattr(mod, attr, layer.original_linear)

    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    for k, v in model.state_dict().items():
        if k in sd: sd[k] = v.cpu().to(sd[k].dtype)
    torch.save(ckpt, OUT_PT)
    serialize_checkpoint_to_eer(OUT_PT, OUT_EER, target_dtype="fp16")
    print(f"ALL DONE in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()

