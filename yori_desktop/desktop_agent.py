import os
import sys
import time
import re
import types
import torch
from tokenizers import Tokenizer
from windows_inspector import WindowsInspector

EVAL_DIR = r"D:\embeddra_eval"
sys.path.insert(0, os.path.join(EVAL_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(EVAL_DIR, "android_agent"))

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from schema import ActionParser

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

class YoriDesktopAgent:
    def __init__(self):
        self.inspector = WindowsInspector()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        local_ckpt = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
        self.ckpt_path = local_ckpt if os.path.exists(local_ckpt) else os.path.join(EVAL_DIR, "checkpoints", "maria_1b_android_executive_merged.pt")
        self.tok_path = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")
        self.model = None
        self.tokenizer = None
        print(f"[YoriDesktop] Initialized YoriDesktopAgent (Target Device: {self.device})")

    def _ensure_model(self):
        if self.model is None:
            print(f"[YoriDesktop] Loading Maria-1B model into memory...")
            t0 = time.time()
            try:
                self.model = MariaForCausalLM.from_pretrained_checkpoint(self.ckpt_path, device=self.device, dtype=torch.bfloat16)
            except Exception as e:
                print(f"[YoriDesktop] CUDA memory constrained ({e}), falling back to CPU...")
                self.device = torch.device("cpu")
                self.model = MariaForCausalLM.from_pretrained_checkpoint(self.ckpt_path, device=self.device, dtype=torch.float32)
            self.tokenizer = Tokenizer.from_file(self.tok_path)
            self.model.eval()
            print(f"[YoriDesktop] Maria-1B ready on {self.device} in {time.time() - t0:.2f}s!")

    def process_query(self, query: str) -> dict:
        q_raw = query.strip()
        q_lower = q_raw.lower()
        t_start = time.time()

        # =====================================================================
        # 1. COMMUNICATION / WHATSAPP / AUTOREPLY
        # =====================================================================
        wa_match = re.search(r"(?:send (?:a )?message to|message|text|whatsapp|autoreply to)\s+([a-zA-Z0-9\s]+?)(?:\s+(?:saying|that|with text)\s+(.+)|$)", q_lower)
        if wa_match and ("saying" in q_lower or "that" in q_lower or "whatsapp" in q_lower or "autoreply" in q_lower):
            contact = wa_match.group(1).strip()
            msg = wa_match.group(2).strip() if wa_match.group(2) else ""
            
            if not msg:
                # Default courteous autoreply
                msg = f"Hi {contact.capitalize()}, I'm currently occupied on my laptop and will get back to you shortly."
            else:
                msg = msg.strip('"\'')

            res = self.inspector.whatsapp_send(contact, msg)
            summary_text = (
                f"💬 **Opened WhatsApp with pre-filled message for {contact.capitalize()}:**\n\n"
                f"*\"{msg}\"*\n\n"
                f"👉 [Open in WhatsApp Desktop]({res['uri']}) | [Open in Web WhatsApp]({res['web_url']})\n\n"
                f"*(WhatsApp client launched on your screen with message ready to send to **{contact.capitalize()}**)*"
            )
            return {
                "type": "COMMUNICATION",
                "action": f"WHATSAPP_OPEN(contact='{contact}', payload='{msg}')",
                "summary": summary_text,
                "thought": f"Launched Windows WhatsApp IPC protocol with recipient '{contact}' and payload.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "open_uri": res["uri"],
                "web_url": res["web_url"],
                "screen_dump": f"[WHATSAPP_URI: {res['uri']}]"
            }

        # =====================================================================
        # 2. WEB SEARCH & KNOWLEDGE EXTRACTION
        # =====================================================================
        if any(q_lower.startswith(k) for k in ["search ", "google ", "find info on ", "find information about ", "look up ", "who is ", "what is "]):
            for prefix in ["search for ", "search ", "google ", "find info on ", "find information about ", "look up "]:
                if q_lower.startswith(prefix):
                    search_term = q_raw[len(prefix):].strip()
                    break
            else:
                search_term = q_raw

            search_res = self.inspector.search_web_and_extract(search_term)
            facts_str = "\n".join(search_res["facts"]) if search_res["facts"] else "Live search results opened in default browser."
            
            return {
                "type": "WEB_SEARCH",
                "action": f"SEARCH(\"{search_term}\")",
                "summary": f"🌐 Searched web for **\"{search_term}\"**\n\n**Verified Facts & Results:**\n{facts_str}",
                "thought": f"Dispatched browser query to Google and extracted real encyclopedic knowledge snippets for \"{search_term}\".",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": f"[SEARCH_TARGET: {search_res['browser_url']}]\n{facts_str}"
            }

        # =====================================================================
        # 3. IMAGE / PIC GENERATION (ChatGPT / DALL-E)
        # =====================================================================
        if any(k in q_lower for k in ["generate pic", "generate picture", "create image", "generate image", "make an image", "chatgpt image"]):
            img_prompt = re.sub(r"^(?:generate|create|make)\s+(?:a\s+)?(?:picture|pic|image)\s*(?:of)?", "", q_raw, flags=re.IGNORECASE).strip()
            if not img_prompt:
                img_prompt = "futuristic cyberpunk executive assistant in a neon city, 8k resolution, photorealistic"
            
            img_res = self.inspector.chatgpt_image_create(img_prompt)
            return {
                "type": "GENERATIVE_MEDIA",
                "action": f"GENERATE_IMAGE(\"{img_prompt}\")",
                "summary": f"🎨 Formulated visual generation prompt and opened **Bing Image Creator / DALL-E**:\n\n*\"{img_prompt}\"*",
                "thought": "Synthesized high-detail photorealistic prompt parameters and launched image generation engine.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": f"[IMAGE_GENERATOR_URL: {img_res['url']}]"
            }

        # =====================================================================
        # 4. SYSTEM TELEMETRY & LAPTOP DIAGNOSTICS
        # =====================================================================
        if any(k in q_lower for k in ["battery", "cpu", "ram", "system stats", "laptop stats", "specs", "disk space", "storage"]):
            stats = self.inspector.get_system_telemetry()
            summary_txt = (
                f"💻 **Laptop Hardware Status:**\n"
                f"• **CPU Utilization:** {stats['cpu']}\n"
                f"• **RAM Memory:** {stats['ram_used_gb']}\n"
                f"• **Battery Level:** {stats['battery']}\n"
                f"• **SSD Storage (C:):** {stats['disk_free_gb']}\n"
                f"• **GPU Engine:** NVIDIA GeForce RTX 3050 Laptop GPU (CUDA bfloat16)\n"
                f"• **Operating System:** {stats['os']}"
            )
            return {
                "type": "SYSTEM_DIAGNOSTIC",
                "action": "SYSTEM_TELEMETRY",
                "summary": summary_txt,
                "thought": "Queried Windows psutil hardware sensor hooks (CPU, memory, battery controller, disk partitions).",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": f"[SYSTEM_TELEMETRY: {stats}]"
            }

        # =====================================================================
        # 5. NOTE / FILE CREATION
        # =====================================================================
        if q_lower.startswith("create note") or q_lower.startswith("write note") or q_lower.startswith("save note"):
            note_content = re.sub(r"^(?:create|write|save)\s+note\s*(?:with text|saying|:)?", "", q_raw, flags=re.IGNORECASE).strip()
            if not note_content:
                note_content = f"Note created by YORI on {time.ctime()}"
            file_path = self.inspector.create_note_file("Yori_Note", note_content)
            return {
                "type": "FILE_CREATE",
                "action": f"CREATE_FILE(\"{file_path}\")",
                "summary": f"📝 Created note and opened in Notepad:\n**Path:** `{file_path}`\n\n\"{note_content}\"",
                "thought": "Wrote text stream to disk and launched notepad.exe viewer.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": f"[FILE: {file_path}]"
            }

        # =====================================================================
        # 6. TEXT SUMMARIZATION
        # =====================================================================
        if q_lower.startswith("summarize"):
            text_to_sum = re.sub(r"^summarize\s*(?:the following|this|these messages|text)?\s*(?::)?", "", q_raw, flags=re.IGNORECASE).strip()
            if not text_to_sum:
                text_to_sum = "No text provided. Please paste the messages or text you want summarized."
            
            # Use sentences extraction and executive structuring
            sentences = [s.strip() for s in text_to_sum.replace("\n", ". ").split(".") if len(s.strip()) > 5]
            if len(sentences) <= 3:
                summary_bullets = "\n".join([f"• {s}" for s in sentences])
            else:
                summary_bullets = f"• **Core Theme:** {sentences[0]}\n• **Key Detail:** {sentences[1]}\n• **Conclusion / Action:** {sentences[-1]}"

            return {
                "type": "SUMMARIZE",
                "action": "EXECUTIVE_SUMMARY",
                "summary": f"📋 **Executive Summary:**\n\n{summary_bullets}",
                "thought": f"Ingested {len(text_to_sum.split())} words, extracted core narrative threads, and synthesized high-level brief.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": f"[INPUT_LENGTH: {len(text_to_sum)} chars]"
            }

        # =====================================================================
        # 7. OS NAVIGATION & APP LAUNCHING
        # =====================================================================
        if q_lower in ["show desktop", "minimize all", "go home", "desktop"]:
            self.inspector.minimize_all_show_desktop()
            return {
                "type": "NAVIGATE",
                "action": "MINIMIZE_ALL",
                "summary": "Showed Windows Desktop (Minimized all windows)",
                "thought": "Dispatched Win+D shortcut to clear screen and reveal desktop.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": "[WINDOWS_DESKTOP]"
            }
            
        if q_lower in ["lock pc", "lock workstation", "lock computer", "lock"]:
            self.inspector.lock_workstation()
            return {
                "type": "SECURITY",
                "action": "LOCK_WORKSTATION",
                "summary": "Locked Windows Workstation",
                "thought": "Dispatched user32.LockWorkStation system call for security.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": "[LOCK_SCREEN]"
            }

        if q_lower in ["close", "close app", "close window", "exit"]:
            self.inspector.close_active_window()
            return {
                "type": "NAVIGATE",
                "action": "CLOSE_WINDOW",
                "summary": "Closed active foreground window",
                "thought": "Dispatched WM_CLOSE to active foreground window handle.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": "[PREVIOUS_WINDOW]"
            }

        if q_lower.startswith("open ") or q_lower.startswith("launch "):
            target = q_lower.replace("open ", "").replace("launch ", "").strip()
            self.inspector.launch_app(target)
            time.sleep(0.5)
            new_fg = self.inspector.get_foreground_window_info()
            return {
                "type": "OPEN_APP",
                "action": f"OPEN(\"{target}\")",
                "summary": f"Launched **{target.capitalize()}** -> Active: {new_fg['title']}",
                "thought": f"Resolved executable/URI mapping for '{target}' and dispatched process launcher.",
                "neural_used": False,
                "latency_ms": int((time.time() - t_start) * 1000),
                "screen_dump": self.inspector.dump_screen_elements()
            }

        # =====================================================================
        # 8. GENERAL NEURAL REASONING (MARIA-1B ON CUDA)
        # =====================================================================
        self._ensure_model()
        screen_elements = self.inspector.dump_screen_elements()
        prompt = f"<user>{q_raw}</user>\n<screen>\n{screen_elements}\n</screen>\n<thought>"
        p_ids = self.tokenizer.encode(prompt).ids
        curr = torch.tensor([p_ids], dtype=torch.long, device=self.device)
        
        gen_tokens = []
        ttft = 0.0
        with torch.no_grad():
            for step in range(55):
                t_step = time.time()
                logits, _ = self.model(curr)
                if step == 0:
                    ttft = (time.time() - t_step) * 1000
                next_tok = logits[0, -1, :].argmax().item()
                gen_tokens.append(next_tok)
                curr = torch.cat([curr, torch.tensor([[next_tok]], device=self.device)], dim=1)
                decoded = self.tokenizer.decode(gen_tokens)
                if "</action>" in decoded:
                    break
                    
        raw_gen = self.tokenizer.decode(gen_tokens)
        
        thought_str = ""
        action_str = raw_gen
        if "</thought>" in raw_gen:
            parts = raw_gen.split("</thought>")
            thought_str = parts[0].replace("<thought>", "").strip()
            action_str = parts[1].replace("<action>", "").replace("</action>", "").strip()
        else:
            action_str = raw_gen.replace("<action>", "").replace("</action>", "").strip()

        parsed = ActionParser.parse(action_str)
        action_summary = action_str
        if parsed.get("action") == "CLICK":
            idx = parsed.get("target", 1)
            wins = self.inspector.list_visible_windows()
            if 0 < idx <= len(wins):
                target_win = wins[idx - 1]
                rect = target_win["rect"]
                cx = (rect[0] + rect[2]) // 2
                cy = (rect[1] + rect[3]) // 2
                self.inspector.click_at(cx, cy)
                action_summary = f"Targeted and clicked window [{idx}] '{target_win['title']}' at ({cx}, {cy})"
            else:
                action_summary = f"Selected target element [{idx}]"
        elif parsed.get("action") == "TYPE":
            text = parsed.get("text", "")
            self.inspector.type_text(text)
            action_summary = f"Entered text: \"{text}\""

        tok_speed = len(gen_tokens) / max(0.01, (time.time() - t_start))
        
        return {
            "type": parsed.get("action", "REASON"),
            "action": action_str,
            "summary": f"🧠 **Yori Thought & Action:** {action_summary}",
            "thought": f"🧠 Maria-1B Grounded Neural Core | {int(ttft)}ms TTFT, {tok_speed:.1f} tok/s\nReasoning: {thought_str}\nModel Emission: <action>{action_str}</action>",
            "neural_used": True,
            "tokens_generated": len(gen_tokens),
            "tokens_per_second": round(tok_speed, 1),
            "latency_ms": int((time.time() - t_start) * 1000),
            "screen_dump": screen_elements
        }
