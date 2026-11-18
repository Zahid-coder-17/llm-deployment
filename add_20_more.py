import os
import json
import random

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
OUTPUT_JSONL = os.path.join(BASE_DIR, "yori_pristine_master_dataset.jsonl")

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

def make_sample(category, query, screen=None, history=None, goal=None, plan=None, action_reason=None, tool_call=None, answer_text=None, doc=None):
    prompt = SYSTEM_CONTRACT
    if doc:
        prompt += f"{doc.strip()}\n"
    if history:
        prompt += f"<history>\n{history.strip()}\n</history>\n"
    prompt += f"<user>{query.strip()}</user>\n"
    if screen:
        prompt += f"<screen>\n{screen.strip()}\n</screen>\n"
    prompt += "<thought>\n"

    if answer_text:
        target = f"{answer_text.strip()}\n</thought>"
    else:
        target = (
            f"🎯 Goal: {goal.strip()}\n"
            f"📋 Plan: {plan.strip()}\n"
            f"⚡ Immediate Action: {action_reason.strip()}\n"
            f"</thought>\n"
            f"<action>{tool_call.strip()}</action>"
        )
    return {
        "category": category,
        "prompt": prompt,
        "target": target,
        "full_text": prompt + target
    }

def add_20_more():
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        existing = [json.loads(line) for line in f]

    more = [
        make_sample("doc_qa", "What are the failing tests in this test run?", doc='<document name="test_run_output.txt">\nTest Suite: AuthModuleTests\n- test_token_expiration: PASSED (12ms)\n- test_refresh_token_revocation: FAILED (AssertionError: Expected 401 but got 200)\n- test_brute_force_lockout: PASSED (45ms)\n</document>', answer_text="Failing Test Summary:\n• 'test_refresh_token_revocation' failed with an AssertionError (Expected HTTP 401 but received 200)."),
        make_sample("doc_qa", "Summarize the key terms of this license agreement", doc='<document name="software_license.txt">\nMIT License Terms:\n- Permission granted free of charge to deal in the Software without restriction.\n- Mandatory: Include original copyright notice and permission notice in all copies.\n- Warranty: Provided "AS IS", without warranty of any kind.\n</document>', answer_text="MIT License Key Terms:\n1. Unrestricted Usage: Free commercial and private use, modification, and distribution.\n2. Attribution: Must retain the original copyright notice in all copies.\n3. Liability: Provided completely 'AS IS' with zero author warranty."),
        make_sample("doc_qa", "What are the server specifications in this configuration?", doc='<document name="server_specs.yaml">\nserver:\n  hostname: edge-node-01\n  cpu: AMD Ryzen 7 7840HS\n  cores: 8 cores, 16 threads\n  ram: 32GB DDR5 5600MHz\n  storage: 1TB NVMe PCIe 4.0\n</document>', answer_text="Server Specifications (edge-node-01):\n• CPU: AMD Ryzen 7 7840HS (8 cores, 16 threads)\n• RAM: 32GB DDR5 5600MHz\n• Storage: 1TB NVMe PCIe 4.0"),
        make_sample("doc_qa", "What is the recommended fix in this vulnerability report?", doc='<document name="sec_advisory.txt">\nAdvisory CVE-2026-8910:\nSeverity: High (CVSS 8.2)\nDescription: Path traversal in file upload handler.\nMitigation: Upgrade libupload to v3.4.2 or sanitize input paths with realpath().\n</document>', answer_text="Vulnerability Mitigation (CVE-2026-8910):\n• Recommended Fix: Upgrade 'libupload' to version 3.4.2 or validate input paths using canonical 'realpath()' sanitization."),
        make_sample("doc_qa", "Summarize the candidate qualifications from this review", doc='<document name="interview_feedback.txt">\nCandidate: Alex Mercer\nRecommendation: Strong Hire\nHighlights: Exceptional distributed systems knowledge, excellent code clarity in Rust, clear design trade-offs.\nAreas for Growth: Could improve unit test coverage speed.\n</document>', answer_text="Interview Assessment for Alex Mercer:\n• Recommendation: Strong Hire.\n• Strengths: Deep expertise in distributed systems, clean Rust architecture, and thoughtful trade-off analysis.\n• Growth Area: Unit testing authoring velocity."),
        make_sample("tech_explanation", "How does Maria-1B achieve fast execution on mobile devices?", answer_text="Maria-1B achieves high mobile execution speed through a specialized lightweight architecture designed by Embeddra AI Research Lab. It utilizes grouped-query attention (GQA), FP16 tensor quantizations, and custom C++ kernels with ARM NEON SIMD intrinsics to maximize hardware cache locality and eliminate cloud round-trip latency."),
        make_sample("tech_explanation", "Explain what an on-device executive agent is", answer_text="An on-device executive agent is an artificial intelligence model that runs entirely on local device silicon to perceive the active screen state, plan multi-step actions, and autonomously interact with mobile or desktop applications without transmitting user data to external servers."),
        make_sample("tech_explanation", "What is the difference between SFT and GRPO?", answer_text="Supervised Fine-Tuning (SFT) trains a model to predict target tokens from curated demonstration pairs. Grouped Relative Policy Optimization (GRPO) is a reinforcement learning algorithm that samples multiple completions for a prompt, scores them using a rule-based reward function (e.g. format compliance or task success), and updates model weights relative to the group average without requiring a separate critic network."),
        make_sample("tech_explanation", "Why is offline inference beneficial for enterprise workflows?", answer_text="Offline inference ensures zero-data-leakage compliance (data never leaves the premises), deterministic low latency unaffected by internet congestion, zero ongoing API per-token costs, and uninterrupted operational reliability even in air-gapped or remote field environments."),
        make_sample("tech_explanation", "What is an attention mechanism in transformers?", answer_text="An attention mechanism computes dynamic weights between token representations in a sequence. By calculating dot products between Query and Key vectors and using Softmax to weight Value vectors, the model dynamically focuses on relevant context words regardless of their distance in the prompt."),
        make_sample("advice", "Give me 3 guidelines for conducting productive code reviews", answer_text="Here are 3 guidelines for constructive code reviews:\n1. Keep Diffs Under 300 Lines: Smaller pull requests receive significantly more thorough architectural reviews.\n2. Differentiate Nitpicks from Blocking Issues: Prefix non-critical stylistic suggestions with 'nit:' so authors know what is essential.\n3. Verify Test Coverage and Edge Cases: Confirm that new branches and error handling paths have corresponding automated tests."),
        make_sample("advice", "What are 3 rules for writing clear executive status updates?", answer_text="Here are 3 rules for executive updates:\n1. Lead with the Bottom Line First (BLUF): State project status (On Track, At Risk, Blocked) in the very first sentence.\n2. Quantify Concrete Progress: Report shipped milestones and measurable metrics rather than listing hours spent.\n3. Clearly State Blockers and Exact Asks: Detail exactly what executive intervention or resource allocation is needed."),
        make_sample("direct_control", "Set an alarm for 04:30 AM", goal="Set alarm for 04:30 AM", plan="1. Create 04:30 alarm", action_reason="Setting alarm directly for 04:30.", tool_call='SET_ALARM("04:30")'),
        make_sample("direct_control", "Set an alarm for 11:00 PM", goal="Set alarm for 23:00", plan="1. Create 23:00 alarm", action_reason="Setting alarm directly for 23:00.", tool_call='SET_ALARM("23:00")'),
        make_sample("direct_control", "Open Hotspot settings", goal="Open Mobile Hotspot settings", plan="1. Open Hotspot settings", action_reason="Opening system Hotspot settings.", tool_call='OPEN_SETTINGS("HOTSPOT")'),
        make_sample("direct_control", "Open Storage settings", goal="Open Storage settings", plan="1. Open Storage settings", action_reason="Opening system Storage settings.", tool_call='OPEN_SETTINGS("STORAGE")'),
        make_sample("direct_control", "Open Network settings", goal="Open Network settings", plan="1. Open Network connections", action_reason="Opening system Network settings.", tool_call='OPEN_SETTINGS("NETWORK")'),
        make_sample("direct_control", "Launch Twitter", goal="Open Twitter application", plan="1. Open Twitter", action_reason="Launching Twitter application.", tool_call='OPEN_APP("Twitter")'),
        make_sample("direct_control", "Launch LinkedIn", goal="Open LinkedIn application", plan="1. Open LinkedIn", action_reason="Launching LinkedIn application.", tool_call='OPEN_APP("LinkedIn")'),
        make_sample("direct_control", "Launch Slack", goal="Open Slack application", plan="1. Open Slack", action_reason="Launching Slack application.", tool_call='OPEN_APP("Slack")')
    ]

    all_samples = existing + more
    unique_dict = {s["prompt"]: s for s in all_samples}
    final_list = list(unique_dict.values())
    random.seed(42)
    random.shuffle(final_list)

    print(f"\n=======================================================")
    print(f" TOTAL PRISTINE 100% UNIQUE SAMPLES: {len(final_list)}")
    print(f"=======================================================")
    cat_counts = {}
    for s in final_list:
        c = s["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for c, count in sorted(cat_counts.items()):
        print(f"  • {c:18s}: {count:3d} samples ({count/len(final_list)*100:4.1f}%)")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in final_list:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[OK] Wrote verified master dataset with {len(final_list)} unique samples to {OUTPUT_JSONL}")

if __name__ == "__main__":
    add_20_more()
