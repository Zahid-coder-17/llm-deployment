import os
import sys
import time
import json
import re
import types
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

import verify_eer_smoke as v
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

EER_PATH = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")

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

VALID_TOOLS = ["OPEN_APP", "OPEN_SETTINGS", "CLICK", "TYPE", "SCROLL", "PRESS_KEY", "TORCH", "VOLUME", "SET_ALARM", "DONE"]

# 100 COMPREHENSIVE REAL-TIME SCENARIOS
BENCHMARK_PROMPTS = [
    # --- A. CONVERSATIONAL & ADVICE (25 PROMPTS) ---
    {"cat": "advice", "q": "Give me 3 valuable pieces of life and career advice", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 effective strategies for managing stress?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 habits for improving coding productivity", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 rules for designing clean software architecture", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 best practices for public speaking?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 rules for maintaining physical fitness and health", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 strategies for mastering machine learning", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 best practices for personal finance and investing?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 habits for better sleep quality", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 rules for effective time management", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 principles for startup product validation?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 guidelines for negotiating job compensation", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 best practices for asynchronous remote work", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 techniques for deep work and concentration?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 rules for effective cross-functional communication", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 best practices for conducting code reviews", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 strategies for overcoming creative burnout?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 principles for API design", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 guidelines for mentoring junior engineers", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 habits for critical thinking and decision making?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 rules for mobile battery conservation", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 best practices for cybersecurity hygiene", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "What are 3 principles for designing mobile user interfaces?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Suggest 3 ways to handle unexpected professional setbacks", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "advice", "q": "Give me 3 guidelines for writing clear technical documentation", "scr": None, "hist": None, "exp_tool": None},

    # --- B. IDENTITY & ARCHITECTURE (10 PROMPTS) ---
    {"cat": "identity", "q": "Who are you?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "Who built you?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "What is your architecture?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "Are you ChatGPT or Claude?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "Where do you run your neural network?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "What is Embeddra AI Research Lab?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "Tell me about yourself and your capabilities", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "Are you an on-device local executive agent?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "What model architecture are you fine-tuned on?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "identity", "q": "Do you send my screen and queries to cloud servers?", "scr": None, "hist": None, "exp_tool": None},

    # --- C. DIRECT OS ACTIONS (25 PROMPTS) ---
    {"cat": "tool", "q": "Turn on the flashlight", "scr": None, "hist": None, "exp_tool": "TORCH"},
    {"cat": "tool", "q": "Torch on", "scr": None, "hist": None, "exp_tool": "TORCH"},
    {"cat": "tool", "q": "Turn off the flashlight", "scr": None, "hist": None, "exp_tool": "TORCH"},
    {"cat": "tool", "q": "Disable torch", "scr": None, "hist": None, "exp_tool": "TORCH"},
    {"cat": "tool", "q": "Set an alarm for 7:30 AM", "scr": None, "hist": None, "exp_tool": "SET_ALARM"},
    {"cat": "tool", "q": "Wake me up at 6:00 AM", "scr": None, "hist": None, "exp_tool": "SET_ALARM"},
    {"cat": "tool", "q": "Set alarm for 22:00", "scr": None, "hist": None, "exp_tool": "SET_ALARM"},
    {"cat": "tool", "q": "Alarm at 8:15 AM", "scr": None, "hist": None, "exp_tool": "SET_ALARM"},
    {"cat": "tool", "q": "Mute the volume", "scr": None, "hist": None, "exp_tool": "VOLUME"},
    {"cat": "tool", "q": "Put phone on silent", "scr": None, "hist": None, "exp_tool": "VOLUME"},
    {"cat": "tool", "q": "Turn up the volume", "scr": None, "hist": None, "exp_tool": "VOLUME"},
    {"cat": "tool", "q": "Volume down", "scr": None, "hist": None, "exp_tool": "VOLUME"},
    {"cat": "tool", "q": "Open YouTube", "scr": None, "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "tool", "q": "Launch Spotify", "scr": None, "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "tool", "q": "Open WhatsApp", "scr": None, "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "tool", "q": "Open Camera", "scr": None, "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "tool", "q": "Launch Google Chrome", "scr": None, "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "tool", "q": "Open Settings", "scr": None, "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "tool", "q": "Open WiFi settings", "scr": None, "hist": None, "exp_tool": "OPEN_SETTINGS"},
    {"cat": "tool", "q": "Go to Bluetooth settings", "scr": None, "hist": None, "exp_tool": "OPEN_SETTINGS"},
    {"cat": "tool", "q": "Open Display settings", "scr": None, "hist": None, "exp_tool": "OPEN_SETTINGS"},
    {"cat": "tool", "q": "Go back", "scr": None, "hist": None, "exp_tool": "PRESS_KEY"},
    {"cat": "tool", "q": "Press home button", "scr": None, "hist": None, "exp_tool": "PRESS_KEY"},
    {"cat": "tool", "q": "Scroll down the screen", "scr": None, "hist": None, "exp_tool": "SCROLL"},
    {"cat": "tool", "q": "Scroll up to top", "scr": None, "hist": None, "exp_tool": "SCROLL"},

    # --- D. MULTI-STEP CLOSED-LOOP UI NAVIGATION (25 PROMPTS) ---
    # WhatsApp flow
    {"cat": "multistep", "q": "Reply to Bilal on WhatsApp", "scr": '[SCREEN: com.sec.android.app.launcher]\n[1] Button "WhatsApp"', "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "multistep", "q": "Reply to Bilal on WhatsApp", "scr": '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Bilal: Meeting today?"\n[2] TextView "Mom"', "hist": 'Step 1: OPEN_APP("WhatsApp")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Reply to Bilal on WhatsApp", "scr": '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Bilal"\n[2] EditText "Message" [Input]', "hist": 'Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal: Meeting today?")', "exp_tool": "TYPE"},
    {"cat": "multistep", "q": "Reply to Bilal on WhatsApp", "scr": '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes at 4" [Typed]\n[2] Button "Send"', "hist": 'Step 1: OPEN_APP(...) -> Step 2: CLICK(...) -> Step 3: TYPE("Message", "Yes at 4")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Reply to Bilal on WhatsApp", "scr": '[SCREEN: com.whatsapp.w4b]\n[1] ImageView "Delivered"', "hist": 'Step 1: OPEN_APP(...) -> Step 2: CLICK(...) -> Step 3: TYPE(...) -> Step 4: CLICK("Send")', "exp_tool": "DONE"},

    # Spotify flow
    {"cat": "multistep", "q": "Play jazz on Spotify", "scr": '[SCREEN: com.sec.android.app.launcher]\n[1] Button "Spotify"', "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "multistep", "q": "Play jazz on Spotify", "scr": '[SCREEN: com.spotify.music]\n[1] Button "Search Tab"', "hist": 'Step 1: OPEN_APP("Spotify")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Play jazz on Spotify", "scr": '[SCREEN: com.spotify.music]\n[1] EditText "Search songs, artists" [Input]', "hist": 'Step 1: OPEN_APP("Spotify") -> Step 2: CLICK("Search Tab")', "exp_tool": "TYPE"},
    {"cat": "multistep", "q": "Play jazz on Spotify", "scr": '[SCREEN: com.spotify.music]\n[1] Button "Play Jazz Classics"', "hist": 'Step 1: OPEN_APP(...) -> Step 2: CLICK(...) -> Step 3: TYPE(...)', "exp_tool": "CLICK"},

    # YouTube flow
    {"cat": "multistep", "q": "Watch Python tutorial on YouTube", "scr": '[SCREEN: com.sec.android.app.launcher]\n[1] Button "YouTube"', "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "multistep", "q": "Watch Python tutorial on YouTube", "scr": '[SCREEN: com.google.android.youtube]\n[1] Button "Search"', "hist": 'Step 1: OPEN_APP("YouTube")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Watch Python tutorial on YouTube", "scr": '[SCREEN: com.google.android.youtube]\n[1] EditText "Search YouTube" [Input]', "hist": 'Step 1: OPEN_APP("YouTube") -> Step 2: CLICK("Search")', "exp_tool": "TYPE"},
    {"cat": "multistep", "q": "Watch Python tutorial on YouTube", "scr": '[SCREEN: com.google.android.youtube]\n[1] TextView "Python Full Course"\n[2] Button "Play Video"', "hist": 'Step 1: OPEN_APP(...) -> Step 2: CLICK(...) -> Step 3: TYPE(...)', "exp_tool": "CLICK"},

    # Settings toggle flow
    {"cat": "multistep", "q": "Turn on Dark Mode in Display settings", "scr": '[SCREEN: com.sec.android.app.launcher]\n[1] Button "Settings"', "hist": None, "exp_tool": "OPEN_SETTINGS"},
    {"cat": "multistep", "q": "Turn on Dark Mode in Display settings", "scr": '[SCREEN: com.android.settings]\n[1] TextView "Display"\n[2] Switch "Dark theme" (OFF)', "hist": 'Step 1: OPEN_SETTINGS("Display")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Turn on Dark Mode in Display settings", "scr": '[SCREEN: com.android.settings]\n[1] Switch "Dark theme" (ON)', "hist": 'Step 1: OPEN_SETTINGS("Display") -> Step 2: CLICK("Dark theme")', "exp_tool": "DONE"},

    # Gmail email reply flow
    {"cat": "multistep", "q": "Reply to manager email on Gmail", "scr": '[SCREEN: com.sec.android.app.launcher]\n[1] Button "Gmail"', "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "multistep", "q": "Reply to manager email on Gmail", "scr": '[SCREEN: com.google.android.gm]\n[1] TextView "Manager: Review project timeline"', "hist": 'Step 1: OPEN_APP("Gmail")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Reply to manager email on Gmail", "scr": '[SCREEN: com.google.android.gm]\n[1] EditText "Compose reply" [Input]', "hist": 'Step 1: OPEN_APP("Gmail") -> Step 2: CLICK("Manager: Review project timeline")', "exp_tool": "TYPE"},
    {"cat": "multistep", "q": "Reply to manager email on Gmail", "scr": '[SCREEN: com.google.android.gm]\n[1] Button "Send"', "hist": 'Step 1: OPEN_APP(...) -> Step 2: CLICK(...) -> Step 3: TYPE(...)', "exp_tool": "CLICK"},

    # Contacts & Phone dialer flow
    {"cat": "multistep", "q": "Call Mom", "scr": '[SCREEN: com.sec.android.app.launcher]\n[1] Button "Phone"', "hist": None, "exp_tool": "OPEN_APP"},
    {"cat": "multistep", "q": "Call Mom", "scr": '[SCREEN: com.samsung.android.dialer]\n[1] TextView "Mom (Mobile)"\n[2] Button "Call"', "hist": 'Step 1: OPEN_APP("Phone")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Add contact Sarah", "scr": '[SCREEN: com.samsung.android.app.contacts]\n[1] Button "Add Contact"', "hist": 'Step 1: OPEN_APP("Contacts")', "exp_tool": "CLICK"},
    {"cat": "multistep", "q": "Add contact Sarah", "scr": '[SCREEN: com.samsung.android.app.contacts]\n[1] EditText "Name" [Input]', "hist": 'Step 1: OPEN_APP("Contacts") -> Step 2: CLICK("Add Contact")', "exp_tool": "TYPE"},
    {"cat": "multistep", "q": "Add contact Sarah", "scr": '[SCREEN: com.samsung.android.app.contacts]\n[1] Button "Save Contact"', "hist": 'Step 1: OPEN_APP(...) -> Step 2: CLICK(...) -> Step 3: TYPE("Name", "Sarah")', "exp_tool": "CLICK"},

    # --- E. DOCUMENT Q&A & FAITHFULNESS (15 PROMPTS) ---
    {"cat": "doc_qa", "q": "<document name=\"nda.pdf\">\nConfidentiality obligations last 3 years from effective date. Governing law is Delaware.\n</document>\n\nHow long do the confidentiality obligations last in this NDA?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"specs.txt\">\nDevice requirements: 6GB RAM, Snapdragon 8 Gen 2 or Exynos 1380, Android 14+.\n</document>\n\nWhat is the minimum RAM required according to the specs?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"meeting.md\">\nAction item 1: John to push v1.2 release. Action item 2: Dave to configure load balancer.\n</document>\n\nWho is assigned to configure the load balancer?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"receipt.txt\">\nTotal paid: $45.20. Payment method: Apple Pay. Return window: 14 days.\n</document>\n\nWhat was the total amount paid on the receipt?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"policy.txt\">\nFull refund granted within 30 days. Digital gift cards are strictly non-refundable.\n</document>\n\nCan digital gift cards be refunded under this policy?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"api.json\">\nEndpoints: GET /api/v1/users (auth required), POST /api/v1/auth/login (public).\n</document>\n\nIs authentication required for the /api/v1/users endpoint?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"prescription.txt\">\nAmoxicillin 500mg: Take 1 capsule three times daily with food for 7 days.\n</document>\n\nHow often should the medication be taken?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"log.txt\">\nERROR [10:45:12] Database connection timed out after 30000ms. Host: 10.0.0.5\n</document>\n\nWhat caused the error in the log file?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"lease.pdf\">\nMonthly rent: $2,400 due on the 1st of each month. Late fee of $100 applies after the 5th.\n</document>\n\nWhen does the late fee apply under the lease?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"flight.txt\">\nFlight EK203 departs Dubai DXB at 14:15, arrives New York JFK at 19:45 local time.\n</document>\n\nWhat is the scheduled arrival time in New York?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"recipe.txt\">\nBake at 375F (190C) for 25 minutes until golden brown.\n</document>\n\nWhat is the recommended oven temperature and baking time?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"code.py\">\ndef calculate_tax(subtotal, rate=0.08):\n    return subtotal * rate\n</document>\n\nWhat is the default tax rate in the python function?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"warranty.pdf\">\nLimited 2-year warranty covers manufacturing defects. Accidental water damage is excluded.\n</document>\n\nDoes the warranty cover accidental water damage?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"invite.txt\">\nEvent: Annual Tech Summit 2026. Date: October 18. Venue: Moscone Center, Hall B.\n</document>\n\nWhere is the Tech Summit being hosted?", "scr": None, "hist": None, "exp_tool": None},
    {"cat": "doc_qa", "q": "<document name=\"shipping.txt\">\nTracking number: TRK-982341. Carrier: FedEx Express. Estimated delivery: Thursday Nov 19.\n</document>\n\nWhat is the tracking number and carrier?", "scr": None, "hist": None, "exp_tool": None}
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== BENCHMARKING YORI ACROSS {len(BENCHMARK_PROMPTS)} REAL-TIME SCENARIOS ON {device} ===", flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    print(f"Loading weights from exported .eer binary: {EER_PATH}...", flush=True)
    sd = v.load_state_dict_from_eer(EER_PATH)
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float16)
    model.load_state_dict(sd, strict=False)
    model.eval()

    total_prompts = len(BENCHMARK_PROMPTS)
    tag_valid_count = 0
    action_leak_count = 0
    tool_syntax_valid_count = 0
    tool_match_count = 0
    id_correct_count = 0

    cat_counts = {}
    cat_tag_ok = {}

    chat_total = 0
    tool_total = 0
    id_total = 0

    t_start = time.time()
    total_tokens_generated = 0

    print("\n" + "=" * 90, flush=True)
    print(f"  {'#':<4} | {'Category':<11} | {'Prompt':<40} | {'TagOK':<6} | {'Leak':<6} | {'Result Action / Summary':<20}")
    print("=" * 90, flush=True)

    for idx, item in enumerate(BENCHMARK_PROMPTS, 1):
        cat = item["cat"]
        is_tool_prompt = cat in ["tool", "multistep"]

        if cat not in cat_counts:
            cat_counts[cat] = 0
            cat_tag_ok[cat] = 0
        cat_counts[cat] += 1

        prompt = SYSTEM_CONTRACT
        if item["hist"]:
            prompt += f"<history>\n{item['hist']}\n</history>\n"
        prompt += f"<user>{item['q']}</user>\n"
        if item["scr"]:
            prompt += f"<screen>\n{item['scr']}\n</screen>\n"
        prompt += "<thought>\n"

        ids = tokenizer.encode(prompt).ids
        curr = torch.tensor([ids], dtype=torch.long, device=device)
        gen = []

        max_gen_toks = 85 if is_tool_prompt else 120

        with torch.no_grad():
            for _ in range(max_gen_toks):
                logits = model(curr)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[0, -1, :].clone()

                # Selective presence penalty (exempt digits, spaces, newlines)
                for tok_id in set(gen[-32:]):
                    if tok_id in [198, 220, 13] or (15 <= tok_id <= 25):
                        continue
                    next_logits[tok_id] -= 0.55

                if is_tool_prompt and len(gen) > 12 and "<action>" in tokenizer.decode(gen):
                    next_tok = next_logits.argmax().item()
                else:
                    probs = F.softmax(next_logits / 0.65, dim=-1)
                    next_tok = torch.multinomial(probs, 1).item()

                if next_tok in [128001, 128009]:
                    break
                gen.append(next_tok)
                curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)
                dec = tokenizer.decode(gen)

                if is_tool_prompt and "</action>" in dec:
                    break
                elif not is_tool_prompt and "</thought>" in dec:
                    break

        total_tokens_generated += len(gen)
        full_output = "<thought>\n" + tokenizer.decode(gen).strip()

        # Tag Validity Check
        has_thought_close = "</thought>" in full_output
        has_action_open = "<action>" in full_output
        has_action_close = "</action>" in full_output
        tag_valid = has_thought_close and (not has_action_open or has_action_close)
        if tag_valid:
            tag_valid_count += 1
            cat_tag_ok[cat] += 1

        # Action Leakage Check (Non-action categories must NEVER have <action>)
        is_leak = False
        if not is_tool_prompt:
            chat_total += 1
            if has_action_open:
                is_leak = True
                action_leak_count += 1

        # Identity Check
        if cat == "identity":
            id_total += 1
            low = full_output.lower()
            if "embeddra" in low and ("maria" in low or "yori" in low):
                id_correct_count += 1

        # Tool Check
        result_desc = ""
        if is_tool_prompt:
            tool_total += 1
            act_match = re.search(r"<action>(.*?)</action>", full_output, re.DOTALL)
            if act_match:
                act_str = act_match.group(1).strip()
                func_name = act_str.split("(")[0].strip()
                if func_name in VALID_TOOLS and act_str.endswith(")"):
                    tool_syntax_valid_count += 1
                    exp = item["exp_tool"]
                    if exp:
                        if func_name == exp or (exp in ["OPEN_APP", "CLICK"] and func_name in ["OPEN_APP", "CLICK"]):
                            tool_match_count += 1
                result_desc = act_str[:22]
            else:
                result_desc = "(No action tag)"
        else:
            # First line of thought response
            first_line = full_output.replace("<thought>", "").strip().split("\n")[0]
            result_desc = first_line[:22]

        q_disp = item["q"].replace("\n", " ")[:38]
        leak_str = "LEAK!" if is_leak else "No"
        tag_str = "PASS" if tag_valid else "FAIL"

        if idx % 5 == 0 or idx == 1 or idx == total_prompts:
            print(f"  {idx:3d}  | {cat:<11} | \"{q_disp:<38}\" | {tag_str:<6} | {leak_str:<6} | {result_desc}", flush=True)

    elapsed = time.time() - t_start
    tok_per_sec = total_tokens_generated / max(1e-3, elapsed)

    print("\n" + "=" * 90, flush=True)
    print("  100-PROMPT BENCHMARK REPORT CARD", flush=True)
    print("=" * 90, flush=True)
    print(f"  Total Prompts Evaluated:          {total_prompts}")
    print(f"  Total Output Tokens Generated:    {total_tokens_generated} tokens")
    print(f"  Total Benchmark Time:             {elapsed:.1f}s ({tok_per_sec:.1f} tokens/sec on CUDA)")
    print("-" * 90)
    print(f"  🏆 Overall Tag Validity Rate:      {tag_valid_count}/{total_prompts} ({tag_valid_count/total_prompts*100:.1f}%)")
    print(f"  🏆 Action Leakage Rate:            {action_leak_count}/{chat_total} ({action_leak_count/max(1, chat_total)*100:.1f}%) [Target: 0%]")
    print(f"  🏆 Tool Syntax Validity Rate:      {tool_syntax_valid_count}/{tool_total} ({tool_syntax_valid_count/max(1, tool_total)*100:.1f}%) [Target: 100%]")
    print(f"  🎯 Tool Intent Match Accuracy:     {tool_match_count}/{tool_total} ({tool_match_count/max(1, tool_total)*100:.1f}%)")
    print(f"  🎯 Identity Contract Accuracy:     {id_correct_count}/{id_total} ({id_correct_count/max(1, id_total)*100:.1f}%)")
    print("-" * 90)
    print("  Category Tag Validity Breakdown:")
    for c, cnt in cat_counts.items():
        ok = cat_tag_ok[c]
        print(f"    * {c:<16}: {ok:2d}/{cnt:2d} ({ok/cnt*100:.1f}%)")
    print("=" * 90, flush=True)

if __name__ == "__main__":
    main()
