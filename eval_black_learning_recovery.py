import os
import sys
import time
import re
import types
import json
import torch
from tokenizers import Tokenizer
from typing import Dict, List, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))

import verify_eer_smoke as v
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

SYSTEM_CONTRACT = (
    "<system>You are YORI, an on-device executive agent.\n"
    "- For conversation, questions, advice, or chit-chat: answer directly inside <thought> with NO <action>.\n"
    "- For device operations and app workflows: inspect the visible <screen> elements, reason step-by-step in <thought>:\n"
    "  🎯 Goal: [Clarified user objective]\n"
    "  📋 Plan: [Numbered steps to achieve goal]\n"
    "  ⚡ Immediate Action: [Why this specific action is chosen based on visible screen]\n"
    "- Then emit ONE action from:\n"
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

def parse_thought_blocks(text: str) -> Dict[str, str]:
    res = {"goal": "", "plan": "", "immediate_action": "", "action": ""}
    if "<action>" in text:
        res["action"] = text.split("<action>")[1].split("</action>")[0].strip()
    
    thought = text.split("</thought>")[0].replace("<thought>", "").strip()
    
    goal_m = re.search(r"🎯\s*Goal:\s*(.*?)(?=\n📋|\n⚡|$)", thought, re.DOTALL)
    if goal_m:
        res["goal"] = goal_m.group(1).strip()
        
    plan_m = re.search(r"📋\s*Plan:\s*(.*?)(?=\n⚡|$)", thought, re.DOTALL)
    if plan_m:
        res["plan"] = plan_m.group(1).strip()
        
    action_m = re.search(r"⚡\s*Immediate Action:\s*(.*?)$", thought, re.DOTALL)
    if action_m:
        res["immediate_action"] = action_m.group(1).strip()
        
    return res

class BlackLearningBenchmark:
    def __init__(self, eer_path: str = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eer_file = eer_path or os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
        tok_file = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
        
        print(f"[*] Loading Black Learning Evaluator on {self.device} from {eer_file}...")
        self.tokenizer = Tokenizer.from_file(tok_file)
        sd = v.load_state_dict_from_eer(eer_file)
        cfg = MariaConfig()
        self.model = MariaForCausalLM(cfg).to(device=self.device, dtype=torch.float16)
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()

    def generate(self, prompt: str, max_tokens: int = 160, repetition_penalty: float = 1.15):
        ids = self.tokenizer.encode(prompt).ids
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)
        past = None
        curr = inp
        pos = 0
        gen_tokens = []
        t0 = time.time()
        
        with torch.no_grad():
            for _ in range(max_tokens):
                logits, past = self.model(curr, position_offset=pos, past_key_values=past, use_cache=True)
                
                if repetition_penalty != 1.0 and len(gen_tokens) > 0:
                    for prev_t in set(gen_tokens[-30:]):
                        if logits[0, -1, prev_t] > 0:
                            logits[0, -1, prev_t] /= repetition_penalty
                        else:
                            logits[0, -1, prev_t] *= repetition_penalty

                t = logits[0, -1, :].argmax().item()
                gen_tokens.append(t)
                pos += curr.shape[1]
                curr = torch.tensor([[t]], dtype=torch.long, device=self.device)
                dec = self.tokenizer.decode(gen_tokens)
                if "</action>" in dec or (t == 128001):
                    break
        dt = time.time() - t0
        return self.tokenizer.decode(gen_tokens), dt

    def run_suite(self):
        # Test scenarios where the model is blinded by unexpected obstacles
        scenarios = [
            {
                "id": "BL-01",
                "name": "Unexpected Cookie Wall in Web Automation",
                "query": "Open chrome, navigate to indeed jobs, and apply for AI Engineer position",
                "screen": '[SCREEN: com.android.chrome]\n[1] TextView "Cookie Preferences"\n[2] TextView "We use cookies to enhance your experience"\n[3] Button "Accept all cookies"\n[4] Button "Reject all"',
                "expected_resolution": ["CLICK(\"Accept all cookies\")", "CLICK(\"Reject all\")"],
                "goal_keywords": ["indeed", "apply", "ai engineer", "job"],
                "obstacle_type": "Cookie Overlay Wall"
            },
            {
                "id": "BL-02",
                "name": "System Permission Interception in Photo Workflow",
                "query": "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp",
                "screen": '[SCREEN: com.google.android.permissioncontroller]\n[1] TextView "Allow Gallery to access photos and videos on this device?"\n[2] Button "Allow"\n[3] Button "Don\'t allow"',
                "expected_resolution": ["CLICK(\"Allow\")", "CLICK(\"Don't allow\")"],
                "goal_keywords": ["gallery", "photo", "camera", "whatsapp", "share"],
                "obstacle_type": "Runtime Permission Modal"
            },
            {
                "id": "BL-03",
                "name": "Negative Environment Feedback (Unclickable Element Recovery)",
                "query": "Open chatgpt and write a message about HyperGQE",
                "screen": '[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[5] View "Ask ChatGPT"\n[6] View "Attachment"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]',
                "expected_resolution": ["TYPE(\"Ask ChatGPT\", \"Explain HyperGQE architecture\")", "TYPE(\"Ask ChatGPT\",", "CLICK(\"Ask ChatGPT\")"],
                "forbidden_actions": ["CLICK(\"Menu\")"],
                "goal_keywords": ["chatgpt", "hypergqe", "message"],
                "obstacle_type": "Static Element Unresponsiveness"
            },
            {
                "id": "BL-04",
                "name": "Off-Screen Viewport Element (Requires Scroll)",
                "query": "Complete checkout and place order",
                "screen": '[SCREEN: com.amazon.mShop.android.shopping]\n[1] TextView "Cart (1 item)"\n[2] TextView "Wireless Headphones - $79.99"\n[3] TextView "Shipping Address: 123 Main St"\n[4] TextView "Payment: Visa ending 4021"',
                "expected_resolution": ["SCROLL(\"DOWN\")"],
                "goal_keywords": ["checkout", "order", "place order"],
                "obstacle_type": "Missing Viewport Target"
            },
            {
                "id": "BL-05",
                "name": "App Version Update Interstitial Dialog",
                "query": "Send message on WhatsApp to Sister Maryam",
                "screen": '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Update WhatsApp"\n[2] TextView "A new version of WhatsApp is available."\n[3] Button "Later"\n[4] Button "Update"',
                "expected_resolution": ["CLICK(\"Later\")"],
                "goal_keywords": ["whatsapp", "maryam", "message"],
                "obstacle_type": "Version Interstitial Dialog"
            }
        ]

        print("\n" + "=" * 80)
        print("🧠 BLACK LEARNING AUTOMATED SELF-EVALUATION BENCHMARK (BL-ORB)")
        print("    Evaluating Model Resilience When Blinded by Unexpected Obstacles")
        print("=" * 80)

        results = []
        total_recovery_time = 0.0
        resolved_count = 0
        goal_retention_count = 0
        adaptation_count = 0
        zero_repeat_count = 0

        for sc in scenarios:
            print(f"\n[Scenario {sc['id']}] {sc['name']}")
            print(f"🚧 Obstacle Type: {sc['obstacle_type']}")
            prompt = f"{SYSTEM_CONTRACT}<user>{sc['query']}</user>\n<screen>\n{sc['screen']}\n</screen>\n<thought>\n"
            
            output, latency = self.generate(prompt)
            blocks = parse_thought_blocks(output)
            action = blocks["action"]
            
            # Metric 1: Obstacle Resolution
            resolved = any(exp in action for exp in sc["expected_resolution"])
            if resolved:
                resolved_count += 1
                
            # Metric 2: Goal Anchor Retention
            goal_lower = blocks["goal"].lower()
            retained = any(kw in goal_lower for kw in sc["goal_keywords"])
            if retained:
                goal_retention_count += 1

            # Metric 3: Plan Dynamic Adaptation
            plan_lower = blocks["plan"].lower()
            adapted = len(blocks["plan"]) > 10 and any(w in plan_lower for w in ["dismiss", "clear", "scroll", "bypass", "permission", "later", "allow", "cookie", "1."])
            if adapted:
                adaptation_count += 1

            # Metric 4: Zero Repeat Failure
            forbidden = sc.get("forbidden_actions", [])
            no_repeat = not any(f in action for f in forbidden)
            if no_repeat:
                zero_repeat_count += 1

            total_recovery_time += latency

            print(f"🎯 Locked Goal:      {blocks['goal']}")
            print(f"📋 Dynamic Plan:     {blocks['plan']}")
            print(f"⚡ Recovery Action:  {action} (Latency: {latency:.2f}s)")
            status_badge = "✓ RESOLVED" if resolved else "✗ FAILED"
            print(f"STATUS: {status_badge} | Goal Retention: {'PASS' if retained else 'FAIL'} | Adaptation: {'PASS' if adapted else 'FAIL'}")

            results.append({
                "id": sc["id"],
                "name": sc["name"],
                "latency_s": latency,
                "resolved": resolved,
                "goal_retained": retained,
                "plan_adapted": adapted,
                "action": action
            })

        n = len(scenarios)
        orr = (resolved_count / n) * 100.0
        gari = (goal_retention_count / n) * 100.0
        pdas = (adaptation_count / n) * 100.0
        zrf = (zero_repeat_count / n) * 100.0
        avg_rec_time = total_recovery_time / n

        print("\n" + "=" * 80)
        print("📊 BLACK LEARNING RECOVERY METRICS SUMMARY")
        print("=" * 80)
        print(f"1. Obstacle Resolution Rate (ORR):             {orr:.1f}% ({resolved_count}/{n})")
        print(f"2. Goal Anchor Retention Index (GARI):         {gari:.1f}% ({goal_retention_count}/{n})")
        print(f"3. Plan Dynamic Adaptation Score (PDAS):       {pdas:.1f}% ({adaptation_count}/{n})")
        print(f"4. Zero-Repeat Failure Rate (ZRF):             {zrf:.1f}% ({zero_repeat_count}/{n})")
        print(f"5. Average Recovery Latency (T_recover):       {avg_rec_time:.3f} seconds / 1.0 turns")
        print("=" * 80)

        # Save benchmark artifact
        out_json = os.path.join(BASE_DIR, "black_learning_benchmark_results.json")
        with open(out_json, "w") as f:
            json.dump({
                "metrics": {
                    "orr_percent": orr,
                    "gari_percent": gari,
                    "pdas_percent": pdas,
                    "zrf_percent": zrf,
                    "avg_recovery_latency_sec": avg_rec_time
                },
                "scenarios": results
            }, f, indent=2)
        print(f"[OK] Benchmark report saved to {out_json}")

if __name__ == "__main__":
    bench = BlackLearningBenchmark()
    bench.run_suite()
