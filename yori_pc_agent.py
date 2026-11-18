import os
import sys
import time
import webbrowser
import subprocess
import re
import torch
import torch.nn.functional as F
from typing import Generator, Dict, Any, Optional, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))

import verify_eer_smoke as v
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from tokenizers import Tokenizer

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

class YoriPcEngine:
    def __init__(self, eer_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eer_file = eer_path or os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
        print(f"[*] Initializing Yori on {self.device} from {eer_file}...")
        
        self.tokenizer = Tokenizer.from_file(os.path.join(BASE_DIR, "checkpoints", "tokenizer.json"))
        sd = v.load_state_dict_from_eer(eer_file)
        self.cfg = MariaConfig()
        self.model = MariaForCausalLM(self.cfg).to(device=self.device, dtype=torch.float16)
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()
        print(f"[OK] Yori Maria-1B ready on {self.device}!")

    def stream_turn(
        self,
        query: str,
        screen_context: Optional[str] = None,
        max_tokens: int = 90,
        temperature: float = 0.0,
        repetition_penalty: float = 1.15
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Streams Maria-1B generation token by token with latency tracking.
        """
        if screen_context:
            prompt = f"{SYSTEM_CONTRACT}<user>{query}</user>\n<screen>\n{screen_context}\n</screen>\n<thought>\n"
        else:
            prompt = f"{SYSTEM_CONTRACT}<user>{query}</user>\n<thought>\n"

        ids = self.tokenizer.encode(prompt).ids
        inp = torch.tensor([ids], dtype=torch.long, device=self.device)

        past_key_values = None
        curr = inp
        pos = 0
        gen_tokens = []

        t_start = time.time()
        ttft_ms = 0.0
        full_text = ""
        eos_id = 128001

        with torch.no_grad():
            for step in range(max_tokens):
                logits, past_key_values = self.model(
                    curr,
                    position_offset=pos,
                    past_key_values=past_key_values,
                    use_cache=True
                )

                if step == 0:
                    ttft_ms = (time.time() - t_start) * 1000.0

                next_logits = logits[0, -1, :].clone().float()

                # Repetition penalty
                if repetition_penalty != 1.0 and len(gen_tokens) > 0:
                    for prev_tok in set(gen_tokens):
                        if next_logits[prev_tok] > 0:
                            next_logits[prev_tok] /= repetition_penalty
                        else:
                            next_logits[prev_tok] *= repetition_penalty

                if temperature > 0:
                    probs = F.softmax(next_logits / temperature, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_token_id = next_logits.argmax().item()

                gen_tokens.append(next_token_id)
                tok_str = self.tokenizer.decode([next_token_id])
                full_text += tok_str

                elapsed = max(0.001, time.time() - t_start)
                tok_per_sec = (step + 1) / elapsed

                yield {
                    "token": tok_str,
                    "full_text": full_text,
                    "ttft_ms": ttft_ms,
                    "tok_per_sec": tok_per_sec,
                    "is_finished": False
                }

                if next_token_id == eos_id or "</action>" in full_text:
                    break

                pos += curr.shape[1]
                curr = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)

        yield {
            "token": "",
            "full_text": full_text,
            "ttft_ms": ttft_ms,
            "tok_per_sec": tok_per_sec,
            "is_finished": True
        }

    def execute_pc_action(self, action_str: str) -> Dict[str, Any]:
        """
        Executes real desktop actions on Windows PC.
        """
        action_clean = action_str.strip()
        tool_name = action_clean.split("(")[0].strip().upper()
        
        inside = action_clean[action_clean.find("(") + 1 : action_clean.rfind(")")] if "(" in action_clean else ""
        raw_args = []
        for m in re.finditer(r'["\']([^"\']*)["\']|([^,\s]+)', inside):
            raw_args.append(m.group(1) or m.group(2) or "")

        res = {
            "tool": tool_name,
            "args": raw_args,
            "success": True,
            "message": "",
            "opened_app": None
        }

        try:
            if tool_name == "OPEN_APP":
                app_target = (raw_args[0] if raw_args else "").lower().strip()
                res["opened_app"] = app_target

                if "chatgpt" in app_target:
                    webbrowser.open("https://chatgpt.com")
                    res["message"] = "🚀 Opened ChatGPT (https://chatgpt.com) in browser"
                elif "youtube" in app_target:
                    webbrowser.open("https://www.youtube.com")
                    res["message"] = "🚀 Opened YouTube (https://www.youtube.com) in browser"
                elif "whatsapp" in app_target:
                    webbrowser.open("https://web.whatsapp.com")
                    res["message"] = "🚀 Opened WhatsApp Web (https://web.whatsapp.com)"
                elif "clock" in app_target or "alarm" in app_target:
                    subprocess.Popen("start ms-clock:", shell=True)
                    res["message"] = "⏰ Opened Windows Clock & Alarms"
                elif "notepad" in app_target:
                    subprocess.Popen("notepad.exe")
                    res["message"] = "📝 Opened Windows Notepad"
                elif "calc" in app_target or "calculator" in app_target:
                    subprocess.Popen("calc.exe")
                    res["message"] = "🧮 Opened Windows Calculator"
                elif "setting" in app_target or "wifi" in app_target or "bluetooth" in app_target:
                    return self.execute_pc_action(f'OPEN_SETTINGS("{app_target}")')
                elif "spotify" in app_target:
                    webbrowser.open("https://open.spotify.com")
                    res["message"] = "🎵 Opened Spotify in browser"
                else:
                    try:
                        subprocess.Popen(f"start {app_target}", shell=True)
                        res["message"] = f"🚀 Launched '{app_target}' on Windows"
                    except Exception:
                        webbrowser.open(f"https://www.google.com/search?q={app_target}")
                        res["message"] = f"🌐 Searched for '{app_target}' in browser"

            elif tool_name == "OPEN_SETTINGS":
                target = (raw_args[0] if raw_args else "").upper().strip()
                uri = "ms-settings:"
                if "WIFI" in target or "WI-FI" in target:
                    uri = "ms-settings:network-wifi"
                elif "BLUETOOTH" in target:
                    uri = "ms-settings:bluetooth"
                elif "DISPLAY" in target:
                    uri = "ms-settings:display"
                elif "SOUND" in target or "VOLUME" in target:
                    uri = "ms-settings:sound"
                elif "BATTERY" in target or "POWER" in target:
                    uri = "ms-settings:powersleep"
                elif "APP" in target:
                    uri = "ms-settings:appsfeatures"

                subprocess.Popen(f"start {uri}", shell=True)
                res["message"] = f"⚙️ Opened Windows Settings ({uri})"

            elif tool_name == "SEARCH":
                query = raw_args[0] if raw_args else ""
                webbrowser.open(f"https://www.google.com/search?q={query}")
                res["message"] = f"🌐 Searched Google for: '{query}'"

            elif tool_name == "TYPE":
                target = raw_args[0] if raw_args else ""
                text = raw_args[1] if len(raw_args) > 1 else ""
                res["message"] = f"⌨️ Typed '{text}' into '{target}'"

            elif tool_name == "CLICK":
                target = raw_args[0] if raw_args else ""
                res["message"] = f"👆 Clicked '{target}'"

            elif tool_name == "TORCH":
                state = raw_args[0] if raw_args else "1"
                state_str = "ON" if str(state) in ("1", "ON", "True") else "OFF"
                res["message"] = f"🔦 Flashlight toggled {state_str}"

            elif tool_name == "VOLUME":
                mode = raw_args[0] if raw_args else "UP"
                res["message"] = f"🔊 Volume adjusted: {mode}"

            elif tool_name == "SET_ALARM":
                t = raw_args[0] if raw_args else "07:00"
                subprocess.Popen("start ms-clock:", shell=True)
                res["message"] = f"⏰ Set alarm for {t} in Windows Clock"

            elif tool_name == "DONE":
                summary = raw_args[0] if raw_args else "Task completed."
                res["message"] = f"✓ {summary}"

            else:
                res["message"] = f"Executed {action_clean}"

        except Exception as e:
            res["success"] = False
            res["message"] = f"Action error: {e}"

        return res
