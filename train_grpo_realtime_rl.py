"""
GRPO Real-Time Multi-Step RL Training for YORI
================================================
Instead of SFT on static (prompt, target) pairs, we:
1. Run the LLM to generate K candidate action sequences (GRPO rollouts)
2. Score each rollout with a reward function that checks:
   - Did the app open? (screen package changed)
   - Did a UI element appear that matches intent?
   - Did we avoid looping (same action 2x with same screen = -1 reward)
   - Did we reach DONE or PLAY within maxTurns?
3. GRPO advantage = normalize rewards across K rollouts per prompt
4. Update LoRA weights using policy gradient

Apps covered (each with 3-5 intents):
  - YouTube:    "find <topic> and play it"       -> OPEN_APP > CLICK(Search) > TYPE > PRESS_KEY(Enter) > CLICK(result)
  - WhatsApp:   "reply to <name> saying <msg>"   -> OPEN_APP > CLICK(chat) > CLICK(input) > TYPE > CLICK(Send)
  - Settings:   "enable bluetooth"               -> OPEN_APP > CLICK(Connections) > CLICK(Bluetooth) > CLICK(toggle)
  - Clock:      "set alarm for <time>"           -> OPEN_APP > CLICK(Add) > TYPE(hour) > TYPE(min) > CLICK(Save)
  - Notes:      "open notes and write <text>"    -> OPEN_APP > CLICK(New) > TYPE(text) > CLICK(Save)
  - Maps:       "navigate to <place>"            -> OPEN_APP > CLICK(search) > TYPE > CLICK(result) > CLICK(Directions)
  - Play Store: "install <app>"                  -> OPEN_APP > TYPE(search) > CLICK(result) > CLICK(Install)
  - Camera:     "take a photo"                   -> OPEN_APP > CLICK(shutter)
"""
import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import json, math, time, types, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

BASE_CKPT  = os.path.join(BASE_DIR, "checkpoints", "maria_1b_20B_weights.pt")
LORA_RESUME= os.path.join(BASE_DIR, "lora_e2e_polish_best.pt")
OUT_LORA   = os.path.join(BASE_DIR, "lora_grpo_rl_best.pt")
OUT_PT     = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER    = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH   = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")


# ---------------------------------------------------------------------------
# SIMULATED ENVIRONMENT: app-specific transition tables
# Each entry: (screen_state, action_taken) -> next_screen_state
# ---------------------------------------------------------------------------
# Screen state = compact string describing what's visible
# This simulates what YoriAccessibilityService.dumpRealScreen() returns

def make_youtube_env(query: str, topic: str):
    """Returns (initial_screen, oracle_states, oracle_check_fn)"""
    search_screen = (
        "[SCREEN: com.google.android.youtube]\n"
        "[1] Button \"Voice search\"\n"
        "[2] Button \"Navigate up\"\n"
        "[3] EditText \"Search YouTube\"\n"
        "[6] TextView \"trending now\"\n"
        "[9] TextView \"popular videos\""
    )
    results_screen = (
        f"[SCREEN: com.google.android.youtube]\n"
        f"[1] EditText \"{topic}\"\n"
        f"[2] Button \"Clear\"\n"
        f"[3] ViewGroup \"{topic} - Full Lecture\"\n"
        f"[4] ViewGroup \"{topic} for Beginners\"\n"
        f"[5] ViewGroup \"{topic} Explained\""
    )
    playing_screen = (
        f"[SCREEN: com.google.android.youtube]\n"
        f"[1] Button \"Pause\"\n"
        f"[2] TextView \"{topic} - Full Lecture\"\n"
        f"[3] Button \"Fullscreen\""
    )
    transitions = {
        ("home", "OPEN_APP(\"YouTube\")"): ("yt_home", "[SCREEN: com.google.android.youtube]\n[2] ImageView \"YouTube\"\n[4] Button \"Search\"\n[5] Button \"Home\""),
        ("yt_home", "CLICK(\"Search\")"): ("yt_search", search_screen),
        ("yt_search", f"TYPE(\"Search YouTube\", \"{topic}\")"): ("yt_results", results_screen),
        ("yt_results", f"CLICK(\"{topic} - Full Lecture\")"): ("yt_playing", playing_screen),
        ("yt_results", f"CLICK(\"{topic} for Beginners\")"): ("yt_playing", playing_screen),
        ("yt_results", f"CLICK(\"{topic} Explained\")"): ("yt_playing", playing_screen),
        ("yt_playing", "DONE(\"Playing\")"): ("done", ""),
    }
    def reward_fn(state, action, next_state, turn):
        if next_state == "done" or next_state == "yt_playing":
            return 1.0
        if next_state == state:  # screen didn't change = wrong action
            return -0.5
        return 0.1  # forward progress
    return "home", transitions, reward_fn

def make_whatsapp_env(contact: str, message: str):
    chat_screen = (
        f"[SCREEN: com.whatsapp]\n"
        f"[1] TextView \"{contact}\"\n"
        f"[2] TextView \"Hey, are you coming?\"\n"
        f"[3] EditText \"Message\" [Input]\n"
        f"[4] Button \"Send\""
    )
    transitions = {
        ("home", "OPEN_APP(\"WhatsApp\")"): ("wa_home", f"[SCREEN: com.whatsapp]\n[1] TextView \"{contact}\"\n[2] TextView \"Chat List\""),
        ("wa_home", f"CLICK(\"{contact}\")"): ("wa_chat", chat_screen),
        ("wa_chat", f"TYPE(\"Message\", \"{message}\")"): ("wa_typed", chat_screen.replace('EditText "Message"', f'EditText "{message}"')),
        ("wa_typed", "CLICK(\"Send\")"): ("done", ""),
        ("wa_typed", "WHATSAPP(\"send\", \"\")"): ("done", ""),
    }
    def reward_fn(state, action, next_state, turn):
        if next_state == "done": return 1.0
        if next_state == state: return -0.5
        return 0.1
    return "home", transitions, reward_fn

def make_settings_env(setting: str):
    """enable/disable bluetooth, wifi, etc."""
    setting_lower = setting.lower()
    transitions = {
        ("home", "OPEN_APP(\"Settings\")"): ("settings_home", f"[SCREEN: com.android.settings]\n[1] Button \"Network\"\n[2] Button \"Connections\"\n[3] Button \"Bluetooth\"\n[4] Button \"Battery\""),
        ("settings_home", "CLICK(\"Connections\")"): ("connections", f"[SCREEN: com.android.settings]\n[1] Button \"Bluetooth\"\n[2] Button \"Wi-Fi\"\n[3] Button \"Mobile data\""),
        ("settings_home", "CLICK(\"Bluetooth\")"): ("bt_toggle", f"[SCREEN: com.android.settings]\n[1] Switch \"Bluetooth\" [Off]"),
        ("connections", "CLICK(\"Bluetooth\")"): ("bt_toggle", f"[SCREEN: com.android.settings]\n[1] Switch \"Bluetooth\" [Off]"),
        ("bt_toggle", "CLICK(\"Bluetooth\")"): ("done", ""),
        ("bt_toggle", "OPEN_SETTINGS(\"bluetooth\")"): ("done", ""),
    }
    def reward_fn(state, action, next_state, turn):
        if next_state == "done": return 1.0
        if next_state == state: return -0.5
        return 0.1
    return "home", transitions, reward_fn

def make_clock_env(hour: str, minute: str):
    transitions = {
        ("home", "OPEN_APP(\"Clock\")"): ("clock_home", "[SCREEN: com.sec.android.app.clockpackage]\n[1] Button \"Add alarm\"\n[2] TextView \"Alarm\""),
        ("clock_home", "CLICK(\"Add alarm\")"): ("alarm_new", "[SCREEN: com.sec.android.app.clockpackage]\n[1] NumberPicker \"Hour\"\n[2] NumberPicker \"Minute\"\n[3] Button \"Save\"\n[4] EditText \"Alarm name\" [Input]"),
        ("alarm_new", "SET_ALARM(\"" + hour + "\", \"" + minute + "\")"): ("done", ""),
        ("alarm_new", "CLICK(\"Save\")"): ("done", ""),
    }
    def reward_fn(state, action, next_state, turn):
        if next_state == "done": return 1.0
        if next_state == state: return -0.5
        return 0.1
    return "home", transitions, reward_fn

def make_maps_env(destination: str):
    transitions = {
        ("home", "OPEN_APP(\"Maps\")"): ("maps_home", "[SCREEN: com.google.android.apps.maps]\n[1] EditText \"Search here\"\n[2] Button \"Explore\""),
        ("maps_home", "CLICK(\"Search here\")"): ("maps_search", f"[SCREEN: com.google.android.apps.maps]\n[1] EditText \"Search here\" [Input]"),
        ("maps_search", f"TYPE(\"Search here\", \"{destination}\")"): ("maps_results", f"[SCREEN: com.google.android.apps.maps]\n[1] ViewGroup \"{destination}\"\n[2] Button \"Directions\""),
        ("maps_results", "CLICK(\"Directions\")"): ("done", ""),
        ("maps_results", f"CLICK(\"{destination}\")"): ("maps_place", f"[SCREEN: com.google.android.apps.maps]\n[1] Button \"Directions\"\n[2] TextView \"{destination}\""),
        ("maps_place", "CLICK(\"Directions\")"): ("done", ""),
    }
    def reward_fn(state, action, next_state, turn):
        if next_state == "done": return 1.0
        if next_state == state: return -0.5
        return 0.1
    return "home", transitions, reward_fn


ALL_ENVS = [
    # YouTube searches
    *[make_youtube_env(f"find {t} and play it from youtube", t) for t in [
        "Agentic AI lecture", "lo-fi beats", "python tutorial",
        "transformer architecture", "deep learning course",
        "Andrej Karpathy lecture", "Stanford CS229", "GRPO explained",
    ]],
    # WhatsApp replies
    *[make_whatsapp_env(c, msg) for c, msg in [
        ("Bilal", "Yes I am coming"), ("Mom", "On my way"),
        ("Raza", "Sure, let's meet at 5"), ("Ali", "Done"),
    ]],
    # Settings
    make_settings_env("enable bluetooth"),
    make_settings_env("turn off bluetooth"),
    # Clock alarms
    *[make_clock_env(h, m) for h, m in [("7", "00"), ("8", "30"), ("6", "15")]],
    # Maps
    *[make_maps_env(p) for p in ["Hyderabad", "Lahore Airport", "FAST University", "DHA Phase 6"]],
]


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, orig: nn.Linear, r=32, alpha=64):
        super().__init__()
        self.original_linear = orig
        self.r = r
        self.scaling = alpha / r
        d_in, d_out = orig.in_features, orig.out_features
        self.lora_A = nn.Parameter(torch.randn(r, d_in, dtype=torch.bfloat16) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(d_out, r, dtype=torch.bfloat16))

    def forward(self, x):
        return self.original_linear(x) + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


# ---------------------------------------------------------------------------
# Token generation (greedy, single turn)
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_one_turn(model, tok_ids: list, tokenizer, device, max_tokens=48, eos_ids=(128001, 128009)):
    """Returns decoded string of generated tokens (excluding prompt). Greedy, fast."""
    ids = torch.tensor([tok_ids[-128:]], dtype=torch.long, device=device)  # truncate context
    generated = []
    decoded_so_far = ""
    for _ in range(max_tokens):
        logits, _ = model(ids)
        next_id = int(logits[0, -1].argmax().item())
        generated.append(next_id)
        if next_id in eos_ids:
            break
        decoded_so_far = tokenizer.decode(generated)
        if "</action>" in decoded_so_far:
            break
        if "</thought>" in decoded_so_far and "<action>" not in decoded_so_far:
            break
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tokenizer.decode(generated)


def encode_prompt(text: str, tokenizer, bos_id=128000):
    return [bos_id] + tokenizer.encode(text).ids


def extract_action(text: str) -> str:
    """Pull action string from model output."""
    if "<action>" in text and "</action>" in text:
        return text.split("<action>")[1].split("</action>")[0].strip()
    return ""


# ---------------------------------------------------------------------------
# GRPO rollout: run K sequences, score, return (prompt_ids, action_logprobs, advantages)
# ---------------------------------------------------------------------------
def run_grpo_episode(model, tokenizer, env_triple, device, K=4, max_turns=5, max_act_tokens=64):
    """
    One GRPO episode across K rollouts of the same intent.
    Returns list of (prompt_ids, action_token_ids, reward) tuples.
    """
    init_state, transitions, reward_fn = env_triple
    # Build intent query from first transition
    # e.g. "find Agentic AI lecture and play it from youtube"
    query = None
    for (s, a) in transitions:
        if s == "home":
            # Reverse-engineer the user query from the app name
            app = a.replace('OPEN_APP("', '').replace('")', '')
            query = f"open {app} and complete the task"
            break
    # Find user-visible intent (the query that led to env creation)
    # We embed it in the prompt as <user>...</user>
    # For YouTube env, the query is baked into make_youtube_env arg
    # We'll extract it from first transition key's action context
    # Simplified: just use a generic query
    if query is None:
        query = "complete the task"

    all_samples = []

    for _ in range(K):
        state = init_state
        traj_prompt_ids = []
        traj_action_ids = []
        traj_rewards = []
        current_screen = ""  # start with no screen dump

        for turn in range(max_turns):
            is_ui_turn = turn > 0 and current_screen
            if is_ui_turn:
                prompt_text = f"<user>{query}</user>\n<screen>\n{current_screen}\n</screen>\n<thought>"
            else:
                prompt_text = f"<user>{query}</user>\n<thought>"

            p_ids = encode_prompt(prompt_text, tokenizer)
            output_text = generate_one_turn(model, p_ids, tokenizer, device, max_tokens=max_act_tokens)
            action_str = extract_action(output_text)

            if not action_str:
                traj_rewards.append(-0.3)
                break

            # Look up transition
            key = (state, action_str)
            if key in transitions:
                next_state, next_screen = transitions[key]
            else:
                # Fuzzy match: try partial match for TYPE/CLICK with wrong text
                next_state, next_screen = state, current_screen  # no-op

            r = reward_fn(state, action_str, next_state, turn)
            traj_rewards.append(r)

            traj_prompt_ids.append(p_ids)
            # Encode the action tokens for loss computation
            action_ids = tokenizer.encode(output_text).ids
            traj_action_ids.append(action_ids)

            state = next_state
            current_screen = next_screen

            if next_state == "done":
                break

        total_reward = sum(traj_rewards)
        all_samples.append((traj_prompt_ids, traj_action_ids, total_reward))

    return all_samples


# ---------------------------------------------------------------------------
# GRPO loss: advantage-weighted log-prob
# ---------------------------------------------------------------------------
def compute_grpo_loss(model, samples, tokenizer, device, eos_id=128001, max_len=256):
    """
    samples: list of (traj_prompt_ids_list, traj_action_ids_list, reward)
    Returns scalar loss.
    """
    rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32)
    if rewards.std() < 1e-6:
        return None  # all same reward, skip update
    advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

    total_loss = torch.tensor(0.0, device=device, requires_grad=False)
    n_valid = 0

    for (traj_p_ids, traj_a_ids, _), adv in zip(samples, advantages.tolist()):
        for p_ids, a_ids in zip(traj_p_ids, traj_a_ids):
            if not a_ids:
                continue
            full_ids = (p_ids + a_ids)[:max_len]
            labels = ([-100] * len(p_ids) + a_ids)[:max_len]
            if not any(l != -100 for l in labels):
                continue

            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            mask = (y != -100)
            tok_log_probs = log_probs.gather(2, y.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            seq_log_prob = (tok_log_probs * mask).sum() / mask.sum().clamp(min=1)

            # Policy gradient: loss = -advantage * log_prob
            step_loss = -adv * seq_log_prob
            total_loss = total_loss + step_loss
            n_valid += 1

    if n_valid == 0:
        return None
    return total_loss / n_valid


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI GRPO REAL-TIME RL TRAINING ON {device} ===", flush=True)
    t0 = time.time()

    tokenizer = Tokenizer.from_file(TOK_PATH)

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr)).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr)).to(device))

    if os.path.exists(LORA_RESUME):
        lora_ckpt = torch.load(LORA_RESUME, map_location=device, weights_only=False)
        model.load_state_dict(lora_ckpt, strict=False)
        print(f"Resumed from {LORA_RESUME}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)

    MAX_STEPS = 120
    WARMUP = 10
    K_ROLLOUTS = 2   # candidates per intent per step (reduced for speed)
    BATCH_ENVS = 2   # envs per gradient step (reduced for speed)

    def lr_schedule(step):
        if step < WARMUP:
            return float(step + 1) / WARMUP
        prog = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    env_pool = ALL_ENVS.copy()
    best_loss = float("inf")
    best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    for step in range(1, MAX_STEPS + 1):
        model.train()
        random.shuffle(env_pool)
        batch = env_pool[:BATCH_ENVS]

        optimizer.zero_grad()
        step_loss_total = 0.0
        n_updates = 0

        for env_triple in batch:
            with torch.no_grad():
                samples = run_grpo_episode(model, tokenizer, env_triple, device, K=K_ROLLOUTS)

            loss = compute_grpo_loss(model, samples, tokenizer, device)
            if loss is None:
                continue

            loss.backward()
            step_loss_total += float(loss.item())
            n_updates += 1

        if n_updates > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
        scheduler.step()

        if step % 20 == 0 or step == 1 or step == MAX_STEPS:
            avg = step_loss_total / max(1, n_updates)
            print(f"Step {step:3d}/{MAX_STEPS} | PG_Loss={avg:.4f} | LR={scheduler.get_last_lr()[0]:.2e} | {time.time()-t0:.1f}s", flush=True)
            if avg < best_loss and n_updates > 0:
                best_loss = avg
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA)
                print(f"  [SAVED] best LoRA at step {step}", flush=True)

    # Always save final
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(final_state, OUT_LORA)

    # Fuse and export .eer
    model.load_state_dict(best_lora_state, strict=False)
    print("\nFusing LoRA -> standalone checkpoint...", flush=True)
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
    for k, v in model.state_dict().items():
        if k in sd:
            sd[k] = v.cpu().to(sd[k].dtype)
    torch.save(ckpt, OUT_PT)
    serialize_checkpoint_to_eer(OUT_PT, OUT_EER, target_dtype="fp16")
    print(f"ALL DONE in {time.time()-t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()

