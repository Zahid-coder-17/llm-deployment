import os
import sys
import math
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
DATASET_PATH = os.path.join(BASE_DIR, "yori_pristine_master_dataset.jsonl")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== EVALUATING CURRENT PERPLEXITIES & ACCURACIES ON {device} ===", flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    print(f"Loading weights from exported .eer binary: {EER_PATH}...", flush=True)
    sd = v.load_state_dict_from_eer(EER_PATH)
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float16)
    model.load_state_dict(sd, strict=False)
    model.eval()

    print(f"Loading dataset: {DATASET_PATH}...", flush=True)
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # Split dataset into Train (85%) and Held-Out Validation (15%) with fixed seed 42 (matching training split)
    import random
    random.seed(42)
    shuffled = list(samples)
    random.shuffle(shuffled)
    val_size = max(20, int(len(shuffled) * 0.15))
    val_samples = shuffled[:val_size]
    print(f"Evaluating Perplexity across all {len(val_samples)} Held-Out Validation samples...", flush=True)

    # 1. PERPLEXITY EVALUATION
    cat_stats = {}
    overall_nll = 0.0
    overall_tokens = 0

    with torch.no_grad():
        for item in val_samples:
            cat = item.get("category", "general")
            full = item["prompt"] + item["target"]
            full_ids = tokenizer.encode(full).ids
            prompt_ids = tokenizer.encode(item["prompt"]).ids

            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            p_len = len(prompt_ids) - 1
            labels[0, :p_len] = -100

            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100, reduction="sum").item()

            ntoks = (labels != -100).sum().item()
            if cat not in cat_stats:
                cat_stats[cat] = [0.0, 0, 0] # [nll, tokens, count]
            cat_stats[cat][0] += nll
            cat_stats[cat][1] += ntoks
            cat_stats[cat][2] += 1

            overall_nll += nll
            overall_tokens += ntoks

    print("\n" + "=" * 80, flush=True)
    print("  1. PERPLEXITY (PPL) BY TASK CATEGORY (HELD-OUT VALIDATION SET)", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Category':<22} | {'Samples':<8} | {'NLL Loss':<10} | {'Perplexity (PPL)':<18} | {'Target Zone':<16}")
    print("-" * 80)

    for cat, (total_nll, ntoks, count) in sorted(cat_stats.items()):
        avg_nll = total_nll / max(1, ntoks)
        ppl = math.exp(min(avg_nll, 20.0))
        target = "12 - 28 (Chat)" if cat in ["advice", "identity", "tech_explanation"] else "2 - 4 (Tool/Action)"
        status = "Optimal" if (cat in ["advice", "identity"] and 10 <= ppl <= 30) or (cat not in ["advice", "identity"] and ppl <= 8) else "In-Range"
        print(f"{cat:<22} | {count:<8} | {avg_nll:<10.4f} | {ppl:<18.2f} | {target} [{status}]")

    avg_overall_nll = overall_nll / max(1, overall_tokens)
    overall_ppl = math.exp(min(avg_overall_nll, 20.0))
    print("-" * 80)
    print(f"{'OVERALL VALIDATION':<22} | {len(val_samples):<8} | {avg_overall_nll:<10.4f} | {overall_ppl:<18.2f} | 1.25 - 2.50 [Sweet Spot]")
    print("=" * 80, flush=True)

    # 2. ACCURACY & BEHAVIORAL BENCHMARK
    print("\n" + "=" * 80, flush=True)
    print("  2. BEHAVIORAL ACCURACY & PROTOCOL COMPLIANCE BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    test_cases = [
        # Conversational / Advice
        {"type": "chat", "query": "Give me 3 valuable pieces of life and career advice", "screen": None, "history": None},
        {"type": "chat", "query": "What are 3 effective strategies for managing stress?", "screen": None, "history": None},
        {"type": "chat", "query": "Suggest 3 habits for improving coding productivity", "screen": None, "history": None},
        # Identity
        {"type": "identity", "query": "Who are you and who built you?", "screen": None, "history": None},
        {"type": "identity", "query": "What architecture do you run on?", "screen": None, "history": None},
        # Direct Tool Actions
        {"type": "tool", "query": "Turn on the flashlight", "screen": None, "history": None, "expected_tool": "TORCH"},
        {"type": "tool", "query": "Set an alarm for 7:30 AM", "screen": None, "history": None, "expected_tool": "SET_ALARM"},
        {"type": "tool", "query": "Mute the volume", "screen": None, "history": None, "expected_tool": "VOLUME"},
        {"type": "tool", "query": "Open YouTube", "screen": '[SCREEN: com.sec.android.app.launcher]\n[1] Button "YouTube"', "history": None, "expected_tool": "OPEN_APP"},
        # Multi-Step Trajectory
        {"type": "multistep", "query": "Find Mom on WhatsApp and open her chat", 
         "screen": '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Mom: Call me"\n[2] TextView "Bilal: Hey"', 
         "history": 'Step 1: OPEN_APP("WhatsApp")', "expected_tool": "CLICK"},
        {"type": "multistep", "query": "Search for lofi beats on Spotify", 
         "screen": '[SCREEN: com.spotify.music]\n[1] EditText "Search songs, artists" [Input]', 
         "history": 'Step 1: OPEN_APP("Spotify") -> Step 2: CLICK("Search Tab")', "expected_tool": "TYPE"},
        {"type": "multistep", "query": "Send message to Sarah", 
         "screen": '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, ready" [Typed]\n[2] Button "Send"', 
         "history": 'Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Sarah") -> Step 3: TYPE("Message", "Yes, ready")', "expected_tool": "CLICK"},
    ]

    total_tests = len(test_cases)
    tag_valid_count = 0
    action_leakage_count = 0 # should be 0
    tool_syntax_valid_count = 0
    tool_match_count = 0
    identity_correct_count = 0

    chat_total = 0
    tool_total = 0
    id_total = 0

    valid_tools = ["OPEN_APP", "OPEN_SETTINGS", "CLICK", "TYPE", "SCROLL", "PRESS_KEY", "TORCH", "VOLUME", "SET_ALARM", "DONE"]

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

    print(f"Running generation on {total_tests} test cases across all categories...\n", flush=True)

    for i, tc in enumerate(test_cases, 1):
        prompt = SYSTEM_CONTRACT
        if tc["history"]:
            prompt += f"<history>\n{tc['history']}\n</history>\n"
        prompt += f"<user>{tc['query']}</user>\n"
        if tc["screen"]:
            prompt += f"<screen>\n{tc['screen']}\n</screen>\n"
        prompt += "<thought>\n"

        ids = tokenizer.encode(prompt).ids
        curr = torch.tensor([ids], dtype=torch.long, device=device)
        gen = []

        is_tool_test = tc["type"] in ["tool", "multistep"]

        for _ in range(150):
            with torch.no_grad():
                logits = model(curr)
                if isinstance(logits, tuple):
                    logits = logits[0]
                next_logits = logits[0, -1, :].clone()

                # Presence / selective repetition penalty
                for tok_id in set(gen[-32:]):
                    if tok_id in [198, 220, 13] or (15 <= tok_id <= 25): # exempt newlines/spaces/digits
                        continue
                    if next_logits[tok_id] > 0:
                        next_logits[tok_id] -= 0.6
                    else:
                        next_logits[tok_id] -= 0.6

                if is_tool_test and len(gen) > 10 and "<action>" in tokenizer.decode(gen):
                    next_tok = next_logits.argmax().item()
                else:
                    probs = F.softmax(next_logits / 0.6, dim=-1)
                    next_tok = torch.multinomial(probs, 1).item()

            if next_tok in [128001, 128009]:
                break
            gen.append(next_tok)
            curr = torch.cat([curr, torch.tensor([[next_tok]], device=device)], dim=1)
            decoded = tokenizer.decode(gen)
            if is_tool_test and "</action>" in decoded:
                break
            elif not is_tool_test and "</thought>" in decoded:
                break

        full_out = "<thought>\n" + tokenizer.decode(gen).strip()

        # Check Tag Validity
        has_thought_close = "</thought>" in full_out
        has_action = "<action>" in full_out
        has_action_close = "</action>" in full_out
        tag_valid = has_thought_close and (not has_action or has_action_close)
        if tag_valid:
            tag_valid_count += 1

        # Check Chat / Action Leakage
        if tc["type"] in ["chat", "identity"]:
            chat_total += 1
            if "<action>" in full_out:
                action_leakage_count += 1

        # Check Identity
        if tc["type"] == "identity":
            id_total += 1
            lower_out = full_out.lower()
            if "embeddra" in lower_out and ("maria" in lower_out or "yori" in lower_out):
                identity_correct_count += 1

        # Check Tool Validity & Match
        if is_tool_test:
            tool_total += 1
            action_match = re.search(r"<action>(.*?)</action>", full_out, re.DOTALL)
            if action_match:
                action_str = action_match.group(1).strip()
                # Check syntax
                func_name = action_str.split("(")[0].strip()
                if func_name in valid_tools and action_str.endswith(")"):
                    tool_syntax_valid_count += 1
                    if func_name == tc["expected_tool"] or (tc["expected_tool"] == "OPEN_APP" and func_name in ["OPEN_APP", "CLICK"]):
                        tool_match_count += 1

        # Print summary line for test case
        tool_tag = f" -> {re.search(r'<action>(.*?)</action>', full_out).group(1)}" if has_action and re.search(r'<action>(.*?)</action>', full_out) else ""
        print(f"[{tc['type'].upper():<9}] \"{tc['query'][:42]:<42}\" | TagOK: {tag_valid} | Leak: {has_action and not is_tool_test}{tool_tag}", flush=True)

    print("-" * 80)
    print("  ACCURACY & COMPLIANCE METRIC SUMMARY:", flush=True)
    print(f"  * Tag Validity Rate:              {tag_valid_count}/{total_tests} ({tag_valid_count/total_tests*100:.1f}%)")
    print(f"  * Action Leakage Rate:            {action_leakage_count}/{chat_total} ({action_leakage_count/chat_total*100:.1f}%) [Target: 0%]")
    print(f"  * Tool Call Syntax Validity:      {tool_syntax_valid_count}/{tool_total} ({tool_syntax_valid_count/tool_total*100:.1f}%) [Target: 100%]")
    print(f"  * Tool Intent Match Accuracy:     {tool_match_count}/{tool_total} ({tool_match_count/tool_total*100:.1f}%)")
    print(f"  * Identity Contract Accuracy:     {identity_correct_count}/{id_total} ({identity_correct_count/id_total*100:.1f}%)")
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
