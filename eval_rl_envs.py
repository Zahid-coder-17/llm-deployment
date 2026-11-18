import os
import sys
import re
import types
import time
import torch
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from verify_eer_smoke import load_state_dict_from_eer, EER_PATH, TOK_PATH

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

SYSTEM_PROMPT = (
    "<system>You are YORI, an on-device Android executive agent.\n"
    "- For conversation, questions, advice, or chit-chat: answer directly inside <thought> with NO <action>.\n"
    "- For device operations and app workflows: inspect the visible <screen> elements, reason step-by-step in <thought>, and emit ONE action from:\n"
    "  OPEN_APP(appName)\n"
    "  OPEN_SETTINGS(settingName) // e.g. \"WIFI\", \"BLUETOOTH\", \"DISPLAY\", \"SOUND\", \"BATTERY\", \"APPS\"\n"
    "  CLICK(elementLabel | [index])\n"
    "  TYPE(elementLabel | [index], text)\n"
    "  SCROLL(UP | DOWN | LEFT | RIGHT)\n"
    "  PRESS_KEY(BACK | HOME | ENTER)\n"
    "  TORCH(1 | 0)\n"
    "  VOLUME(UP | DOWN | MUTE)\n"
    "  SET_ALARM(HH:MM)\n"
    "  DONE(summary)\n</system>\n"
)

def generate_greedy(model, tokenizer, prompt: str, device, max_tokens=65):
    ids = tokenizer.encode(prompt).ids
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    curr = inp
    pos = 0
    gen_tokens = []

    with torch.no_grad():
        for _ in range(max_tokens):
            logits, past = model(curr, position_offset=pos, past_key_values=past, use_cache=True)
            t = logits[0, -1, :].argmax().item()
            gen_tokens.append(t)
            pos += curr.shape[1]
            curr = torch.tensor([[t]], dtype=torch.long, device=device)
            dec = tokenizer.decode(gen_tokens)
            if "</action>" in dec or (t == 128001):
                break
    return tokenizer.decode(gen_tokens)

def parse_action(out: str):
    if "<action>" not in out:
        return "NONE", out.strip()
    act = out.split("<action>")[1].split("</action>")[0].strip()
    return act, out.split("</thought>")[0].replace("<thought>", "").strip()

class SimulatedAndroidEnv:
    def __init__(self, name, goal, states, initial_state="START"):
        self.name = name
        self.goal = goal
        self.states = states
        self.curr_state = initial_state
        self.history = []

    def step(self, action_str: str):
        state_def = self.states[self.curr_state]
        for expected_pat, next_state, reward in state_def["transitions"]:
            if re.search(expected_pat, action_str, re.IGNORECASE):
                self.curr_state = next_state
                done = (next_state == "DONE")
                return reward, done, f"Transitioned -> {next_state}"
        return -1.0, False, f"No-op in state {self.curr_state} for {action_str}"

def build_test_environments():
    return [
        SimulatedAndroidEnv(
            "ChatGPT Multi-Step Workflow",
            "Open chatgpt and write a message about HyperGQE",
            {
                "START": {
                    "screen": None,
                    "transitions": [(r'OPEN_APP\("ChatGPT"\)', "CG_OPENED", 1.0)]
                },
                "CG_OPENED": {
                    "screen": '[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[4] View "Temporary chat"\n[5] View "Ask ChatGPT"\n[8] View "Attachment"\n[10] View "Dictation"\n[12] View "Start a voice conversation"',
                    "transitions": [(r'TYPE\(("Ask ChatGPT"|\[5\]|"Message"),\s*".*HyperGQE.*"\)', "CG_TYPED", 1.0)]
                },
                "CG_TYPED": {
                    "screen": '[SCREEN: com.openai.chatgpt]\n[5] EditText "Explain HyperGQE architecture"\n[6] Button "Send message"',
                    "transitions": [(r'CLICK\(("Send message"|\[6\]|"Send")\)', "DONE", 1.0)]
                }
            }
        ),
        SimulatedAndroidEnv(
            "YouTube Multi-Step Workflow",
            "find Agentic AI lecture and play it from youtube",
            {
                "START": {
                    "screen": None,
                    "transitions": [(r'OPEN_APP\("YouTube"\)', "YT_HOME", 1.0)]
                },
                "YT_HOME": {
                    "screen": '[SCREEN: com.google.android.youtube]\n[1] TextView "YouTube"\n[2] Button "Notifications"\n[3] Button "Search"\n[4] TextView "Trending Video"\n[5] Tab "Home"',
                    "transitions": [(r'CLICK\(("Search"|\[3\])\)', "YT_SEARCH_BOX", 1.0)]
                },
                "YT_SEARCH_BOX": {
                    "screen": '[SCREEN: com.google.android.youtube]\n[1] ImageButton "Navigate up"\n[2] EditText "Search YouTube" [Input]\n[3] Button "Voice search"',
                    "transitions": [(r'TYPE\(("Search YouTube"|\[2\]),\s*".*Agentic AI.*"\)', "YT_RESULTS", 1.0)]
                },
                "YT_RESULTS": {
                    "screen": '[SCREEN: com.google.android.youtube]\n[1] EditText "Agentic AI lecture" [Input]\n[2] Button "Filters"\n[3] TextView "Agentic AI Lecture - Full Course"\n[4] TextView "Random Vlog"',
                    "transitions": [(r'CLICK\(("Agentic AI Lecture - Full Course"|\[3\])\)', "DONE", 1.0)]
                },
            }
        ),
        SimulatedAndroidEnv(
            "WhatsApp Multi-Step Workflow",
            "Send Message to Sister Maryam Bill Gates saying \"Hi ,I'm yori an ai agent built on Maria\"",
            {
                "START": {
                    "screen": None,
                    "transitions": [(r'OPEN_APP\("WhatsApp"\)', "WA_HOME", 1.0)]
                },
                "WA_HOME": {
                    "screen": '[SCREEN: com.whatsapp.w4b]\n[1] Button "Search…"\n[2] TextView "Sister Maryam"\n[3] TextView "Recent message"',
                    "transitions": [(r'CLICK\(("Sister Maryam"|\[2\])\)', "WA_CHAT", 1.0)]
                },
                "WA_CHAT": {
                    "screen": '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Sister Maryam"\n[2] EditText "Message"\n[3] Button "Voice message"',
                    "transitions": [(r'TYPE\(("Message"|\[2\]),\s*".*yori.*"\)', "WA_SEND", 1.0)]
                },
                "WA_SEND": {
                    "screen": '[SCREEN: com.whatsapp.w4b]\n[1] EditText "Hi ,I\'m yori an ai agent built on Maria"\n[2] Button "Send"',
                    "transitions": [(r'CLICK\(("Send"|\[2\])\)', "DONE", 1.0)]
                }
            }
        ),
        SimulatedAndroidEnv(
            "Wi-Fi Settings Workflow",
            "Open Wi-Fi Settings",
            {
                "START": {
                    "screen": None,
                    "transitions": [(r'OPEN_SETTINGS\("WIFI"\)', "DONE", 1.0)]
                }
            }
        ),
        SimulatedAndroidEnv(
            "Flashlight Control Workflow",
            "Turn on flashlight",
            {
                "START": {
                    "screen": None,
                    "transitions": [(r'TORCH\(1\)', "DONE", 1.0)]
                }
            }
        ),
        SimulatedAndroidEnv(
            "Clock Alarms Workflow",
            "Check my alarms on this phone",
            {
                "START": {
                    "screen": None,
                    "transitions": [(r'OPEN_APP\("Clock"\)', "DONE", 1.0)]
                }
            }
        ),
        SimulatedAndroidEnv(
            "Closed-Loop Negative Feedback Recovery",
            "Open chatgpt and write a message about HyperGQE",
            {
                "START": {
                    "screen": '[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[5] View "Ask ChatGPT"\n[6] View "Attachment"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]',
                    "transitions": [(r'TYPE\(("Ask ChatGPT"|\[5\]),\s*".*HyperGQE.*"\)', "DONE", 1.0)]
                }
            }
        )
    ]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Loading model from {EER_PATH} on {device}...", flush=True)
    t0 = time.time()
    sd = load_state_dict_from_eer(EER_PATH)
    model = MariaForCausalLM(MariaConfig()).to(device=device, dtype=torch.float16)
    model.load_state_dict(sd, strict=False)
    model.eval()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    print(f"[OK] Model loaded in {time.time()-t0:.2f}s!", flush=True)

    envs = build_test_environments()
    passed_envs = 0
    total_steps = 0

    print("\n" + "=" * 75)
    print(" 🌍 YORI CLOSED-LOOP REAL-TIME ENVIRONMENT EXPLORATION SUITE")
    print("=" * 75, flush=True)

    for env_idx, env in enumerate(envs, 1):
        print(f"\n--- [ENV {env_idx}/{len(envs)}] {env.name} ---")
        print(f"Goal: \"{env.goal}\"")
        done = False
        step_idx = 0

        while not done and step_idx < 6:
            step_idx += 1
            total_steps += 1
            st = env.states[env.curr_state]
            if st["screen"]:
                prompt = f"{SYSTEM_PROMPT}<user>{env.goal}</user>\n<screen>\n{st['screen']}\n</screen>\n<thought>"
            else:
                prompt = f"{SYSTEM_PROMPT}<user>{env.goal}</user>\n<thought>"

            t_step = time.time()
            out = generate_greedy(model, tokenizer, prompt, device)
            dt = time.time() - t_step

            act, thought = parse_action(out)
            r, done, info = env.step(act)

            print(f"  Step {step_idx} | State: {env.curr_state:<12} | Act: {act:<35} | {info} ({dt:.2f}s)")
            print(f"         🧠 Thought: {thought[:80]}...")

            if done:
                passed_envs += 1
                print(f"  ✓ [SUCCESS] {env.name} solved end-to-end in {step_idx} steps!")
                break
            if r < 0:
                print(f"  ✗ [FAILED] Stalled at state {env.curr_state} for action: {act}")
                break

    print("\n" + "=" * 75)
    print(f"🏁 RESULT: {passed_envs}/{len(envs)} REAL-TIME ENVIRONMENTS SOLVED END-TO-END!")
    print(f"Total Steps Taken: {total_steps} | Pure Neural Zero-Heuristics Execution")
    print("=" * 75, flush=True)

if __name__ == "__main__":
    main()
