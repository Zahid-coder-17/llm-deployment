"""
EMBEDDRA AI: MASTER UNIVERSAL ANDROID EXECUTIVE AGENT TRAINING
Pure Neural Generalist Agent on Maria-1B Base Weights (FP16/ARM NEON)
Zero rule-based heuristics. Zero hardcoded bypasses. Pure dynamic LLM perception.

Capabilities:
1. Dynamically distinguish Conversational Chat (advice, questions, chit-chat) -> emits pure <thought> without <action>.
2. Dynamically execute System Operations (Wi-Fi, Bluetooth, Alarms, Torch, Volume, Battery, Settings).
3. Dynamically perform Multi-Step UI Actions across ALL apps (ChatGPT, YouTube, WhatsApp, Spotify, Maps, Notes, Clock).
4. Dynamically recover from environment feedback (unchanged screen -> select alternate visible element).
"""
import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import json
import math
import time
import types
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.append(os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.append(os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

BASE_CKPT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
OUT_LORA = os.path.join(BASE_DIR, "lora_master_universal_v3.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")

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

def build_universal_training_samples():
    random.seed(42)
    samples = []

    # -------------------------------------------------------------
    # CATEGORY A: Pure Conversational Chat (NO <action> EMITTED)
    # -------------------------------------------------------------
    chat_pairs = [
        ("3 advices pls", "Here are 3 valuable pieces of advice:\n1. Focus on high-impact priorities early in the day.\n2. Build consistency through small, sustainable daily habits.\n3. Reflect regularly and iterate continuously based on feedback."),
        ("give me 3 advices", "Here are 3 practical pieces of advice:\n1. Protect your deep focus time from distractions.\n2. Prioritize quality sleep and daily physical activity.\n3. Keep learning and master skills by actively building."),
        ("3 advice for life", "Here are 3 core principles for life:\n1. Take ownership of your time and decisions.\n2. Cultivate deep relationships with people who inspire you.\n3. Stay curious and never stop learning."),
        ("tell me 3 tips for productivity", "1. Single-task on your most important project first.\n2. Work in focused 25-minute intervals.\n3. Plan your top 3 goals the night before."),
        ("who are you?", "I'm Yori, a private on-device AI executive assistant created by Embeddra AI. I run 100% locally on your phone to chat and control your device."),
        ("who made you?", "I was built by Embeddra AI using the Maria-1B on-device neural architecture running directly on ARM NEON."),
        ("what is your name?", "My name is Yori. I am your autonomous on-device AI assistant."),
        ("what is edge ai?", "Edge AI refers to running artificial intelligence models directly on local hardware like your phone or laptop rather than relying on remote cloud servers."),
        ("what is hypergqe?", "HyperGQE is an advanced graph-based query embedding architecture designed for high-efficiency structural reasoning and neural query evaluation."),
        ("tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
        ("how are you?", "I'm running fast and smoothly directly on your phone's processor! How can I assist you today?"),
        ("hello", "Hello! I'm Yori. What task would you like me to execute on your phone today?"),
        ("hi", "Hi there! I'm ready to help you navigate your apps or answer any questions."),
        ("hey yori", "Hey! What would you like me to do on your device?"),
        ("what can you do?", "I can answer questions, explain concepts, and autonomously control your phone—like opening apps, searching YouTube, sending WhatsApp messages, setting alarms, and toggling device settings."),
        ("thanks", "You're very welcome! Let me know if you need anything else on your device."),
        ("thank you yori", "My pleasure! Feel free to ask anytime."),
        ("explain quantum computing in simple terms", "Quantum computing uses the strange properties of quantum physics—like superposition and entanglement—to process complex calculations much faster than traditional computers."),
        ("why is the sky blue?", "The sky looks blue because molecules in Earth's atmosphere scatter sunlight in all directions, and blue light has shorter, smaller waves that scatter more than other colors."),
        ("what time is it in tokyo when it is noon in london?", "Tokyo is 9 hours ahead of UTC/GMT (London standard time), so when it is 12:00 PM in London, it is 9:00 PM in Tokyo."),
    ]
    for q, a in chat_pairs:
        for _ in range(3):
            samples.append({
                "prompt": f"{SYSTEM_PROMPT}<user>{q}</user>\n<thought>",
                "target": f"{a}</thought>"
            })

    # -------------------------------------------------------------
    # CATEGORY B: System & Settings Operations
    # -------------------------------------------------------------
    settings_data = [
        ("Open Wi-Fi Settings", "User wants to open Wi-Fi settings.", 'OPEN_SETTINGS("WIFI")'),
        ("open wifi settings", "User wants to configure Wi-Fi.", 'OPEN_SETTINGS("WIFI")'),
        ("check wifi", "User wants to inspect Wi-Fi connections.", 'OPEN_SETTINGS("WIFI")'),
        ("turn on wifi", "User wants to manage Wi-Fi settings.", 'OPEN_SETTINGS("WIFI")'),
        ("Open Bluetooth Settings", "User wants to open Bluetooth settings.", 'OPEN_SETTINGS("BLUETOOTH")'),
        ("open bluetooth", "User wants to manage Bluetooth devices.", 'OPEN_SETTINGS("BLUETOOTH")'),
        ("open display settings", "User wants to adjust display settings.", 'OPEN_SETTINGS("DISPLAY")'),
        ("display brightness", "User wants to change display settings.", 'OPEN_SETTINGS("DISPLAY")'),
        ("Open Sound Settings", "User wants to open sound and volume settings.", 'OPEN_SETTINGS("SOUND")'),
        ("sound settings", "User wants to configure audio.", 'OPEN_SETTINGS("SOUND")'),
        ("check battery settings", "User wants to view battery usage.", 'OPEN_SETTINGS("BATTERY")'),
        ("battery info", "User wants to inspect device battery.", 'BATTERY()'),
        ("check battery", "User wants to inspect battery level.", 'BATTERY()'),
        ("Turn on flashlight", "User wants to turn on the flashlight.", 'TORCH(1)'),
        ("turn on torch", "User wants to enable torch.", 'TORCH(1)'),
        ("Turn off flashlight", "User wants to turn off the flashlight.", 'TORCH(0)'),
        ("torch off", "User wants to disable torch.", 'TORCH(0)'),
        ("Volume up", "User wants to increase media volume.", 'VOLUME("UP")'),
        ("increase volume", "User wants louder audio.", 'VOLUME("UP")'),
        ("Volume down", "User wants to decrease media volume.", 'VOLUME("DOWN")'),
        ("lower volume", "User wants quieter audio.", 'VOLUME("DOWN")'),
        ("Mute audio", "User wants to mute device audio.", 'VOLUME("MUTE")'),
        ("mute phone", "User wants to mute volume.", 'VOLUME("MUTE")'),
        ("Check my alarms on this phone", "User wants to view device alarms. I will open the Clock app.", 'OPEN_APP("Clock")'),
        ("check alarms", "User wants to view alarms. I will open Clock.", 'OPEN_APP("Clock")'),
        ("show alarms", "User wants to see active alarms.", 'OPEN_APP("Clock")'),
        ("see my alarms", "User wants to check alarms in Clock.", 'OPEN_APP("Clock")'),
        ("Set alarm for 07:00", "User wants to set an alarm for 7:00 AM.", 'SET_ALARM("07:00")'),
        ("set alarm for 6:30", "User wants to set an alarm for 6:30 AM.", 'SET_ALARM("06:30")'),
        ("Take a screenshot", "User wants to capture a screenshot.", 'SCREENSHOT()'),
        ("take screenshot", "User wants to capture the screen.", 'SCREENSHOT()'),
        ("capture screen", "User wants to capture screenshot.", 'SCREENSHOT()'),
        ("Go to home screen", "User wants to return to the Home screen.", 'GLOBAL_HOME()'),
        ("go home", "User wants to navigate home.", 'GLOBAL_HOME()'),
        ("open recents", "User wants to view recent apps.", 'GLOBAL_RECENTS()'),
        ("open notifications", "User wants to open notifications.", 'GLOBAL_NOTIFICATIONS()'),
    ]
    for q, t, a in settings_data:
        for _ in range(2):
            samples.append({
                "prompt": f"{SYSTEM_PROMPT}<user>{q}</user>\n<thought>",
                "target": f"{t}</thought>\n<action>{a}</action>"
            })

    # -------------------------------------------------------------
    # CATEGORY C: Multi-Step Universal App Workflows (All Major Apps)
    # -------------------------------------------------------------
    # 1. Generic App Launch Grounding (Any "Open <App>")
    app_launch_pairs = [
        ("open whatsapp", "WhatsApp"),
        ("launch whatsapp", "WhatsApp"),
        ("open youtube", "YouTube"),
        ("launch youtube", "YouTube"),
        ("open chatgpt", "ChatGPT"),
        ("launch chatgpt", "ChatGPT"),
        ("open spotify", "Spotify"),
        ("open maps", "Google Maps"),
        ("open chrome", "Chrome"),
        ("open clock", "Clock"),
        ("open notes", "Samsung Notes"),
        ("open settings", "Settings"),
    ]
    for q, app in app_launch_pairs:
        for _ in range(3):
            samples.append({
                "prompt": f"{SYSTEM_PROMPT}<user>{q}</user>\n<thought>",
                "target": f'User wants to open {app}. I will launch the app.</thought>\n<action>OPEN_APP("{app}")</action>'
            })

    # 2. ChatGPT Multi-Step Workflows (Full Real-Screen Grounding)
    cg_goals = [
        ("Open chatgpt and write a message about HyperGQE", "Explain the HyperGQE architecture"),
        ("open chatgpt and ask about HyperGQE", "Explain HyperGQE query embedding"),
        ("launch chatgpt and ask how hypergqe works", "How does HyperGQE neural query evaluation work?"),
        ("open chatgpt and write a message about quantum computing", "Explain quantum computing principles"),
    ]
    for user_g, query_text in cg_goals:
        for _ in range(25):
            # Step 1: User asks from Yori/Home screen
            samples.append({
                "prompt": f"{SYSTEM_PROMPT}<user>{user_g}</user>\n<thought>",
                "target": f'User wants to use ChatGPT to ask about this topic. I will launch ChatGPT first.</thought>\n<action>OPEN_APP("ChatGPT")</action>'
            })
            # Step 2 (Real Screen with "Ask ChatGPT"):
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[4] View "Temporary chat"\n[5] View "Ask ChatGPT"\n[8] View "Attachment"\n[10] View "Dictation"\n[12] View "Start a voice conversation"\n</screen>\n<thought>',
                "target": f'ChatGPT is open. I see the visible \'Ask ChatGPT\' text input area on screen. I will type the prompt.</thought>\n<action>TYPE("Ask ChatGPT", "{query_text}")</action>'
            })
            # Step 3: Text is typed, send button visible
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.openai.chatgpt]\n[5] EditText "{query_text}"\n[6] Button "Send message"\n</screen>\n<thought>',
                "target": f'The query is typed into the input field. I will tap Send message to submit it to ChatGPT.</thought>\n<action>CLICK("Send message")</action>'
            })
            # Step 4: ChatGPT response received
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.openai.chatgpt]\n[4] TextView "HyperGQE is an advanced graph query embedding architecture designed by DeepMind..."\n</screen>\n<thought>',
                "target": f'ChatGPT has received and answered the prompt. Task complete.</thought>\n<action>DONE("ChatGPT prompt submitted successfully")</action>'
            })

    # 3. WhatsApp Messaging Multi-Step
    wa_goals = [
        ("Send Message to Sister Maryam Bill Gates saying \"Hi ,I'm yori an ai agent built on Maria\"", "Sister Maryam", "Hi ,I'm yori an ai agent built on Maria"),
        ("Send WhatsApp to Bilal saying \"Meeting at 4pm\"", "Bilal", "Meeting at 4pm"),
        ("text Mom on WhatsApp saying \"I'm on my way\"", "Mom", "I'm on my way"),
    ]
    for user_g, contact, msg in wa_goals:
        for _ in range(8):
            # Step 1: Open WhatsApp
            samples.append({
                "prompt": f"{SYSTEM_PROMPT}<user>{user_g}</user>\n<thought>",
                "target": f'User wants to send a WhatsApp message to {contact}. I must open WhatsApp first.</thought>\n<action>OPEN_APP("WhatsApp")</action>'
            })
            # Step 2: Contact visible in list
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] Button "Search…"\n[2] TextView "{contact}"\n[3] TextView "Recent message"\n</screen>\n<thought>',
                "target": f'I see {contact} in the chat list. I will tap the contact.</thought>\n<action>CLICK("{contact}")</action>'
            })
            # Step 2b: Search for contact if not immediately top
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] Button "Search…"\n[2] TextView "Other Contact"\n</screen>\n<thought>',
                "target": f'I will tap Search to locate {contact}.</thought>\n<action>CLICK("Search…")</action>'
            })
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] EditText "Search…"\n</screen>\n<thought>',
                "target": f'I will type {contact} into the search field.</thought>\n<action>TYPE("Search…", "{contact}")</action>'
            })
            # Step 3: Inside chat, type message
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] TextView "{contact}"\n[2] EditText "Message"\n[3] Button "Voice message"\n</screen>\n<thought>',
                "target": f'In chat with {contact}. I will enter the requested message.</thought>\n<action>TYPE("Message", "{msg}")</action>'
            })
            # Step 4: Tap Send
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] EditText "{msg}"\n[2] Button "Send"\n</screen>\n<thought>',
                "target": f'The message is ready. I will tap Send to deliver it.</thought>\n<action>CLICK("Send")</action>'
            })
            # Step 5: Delivered
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.whatsapp.w4b]\n[1] TextView "{msg}"\n[2] ImageView "Delivered"\n</screen>\n<thought>',
                "target": f'The message has been sent successfully to {contact}.</thought>\n<action>DONE("Message delivered to {contact}")</action>'
            })

    # 4. YouTube Multi-Step Workflows
    yt_goals = [
        ("find Agentic AI lecture and play it from youtube", "Agentic AI lecture", "What is Agentic AI and How Does it Work?"),
        ("open youtube and play quantum computing video", "quantum computing", "Quantum Computing in 10 Minutes"),
        ("play hans zimmer music on youtube", "hans zimmer", "Hans Zimmer Live Concert"),
    ]
    for user_g, query, vid_title in yt_goals:
        for _ in range(8):
            # Step 1: Open YouTube
            samples.append({
                "prompt": f"{SYSTEM_PROMPT}<user>{user_g}</user>\n<thought>",
                "target": f'User wants to watch a video on YouTube. I will launch YouTube first.</thought>\n<action>OPEN_APP("YouTube")</action>'
            })
            # Step 2: Search button
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] ImageView "YouTube"\n[2] Button "Search"\n[3] Button "Cast"\n</screen>\n<thought>',
                "target": f'YouTube is open. I see the Search button. I will tap Search.</thought>\n<action>CLICK("Search")</action>'
            })
            # Step 3: Type search
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] EditText "Search YouTube"\n</screen>\n<thought>',
                "target": f'The search bar is active. I will type the query.</thought>\n<action>TYPE("Search YouTube", "{query}")</action>'
            })
            # Step 4: Search results
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] ViewGroup "{vid_title}"\n[2] ViewGroup "Alternative Video"\n</screen>\n<thought>',
                "target": f'I see relevant video results. I will tap to play \'{vid_title}\'.</thought>\n<action>CLICK("{vid_title}")</action>'
            })
            # Step 5: Playing
            samples.append({
                "prompt": f'{SYSTEM_PROMPT}<user>{user_g}</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] Button "Pause video"\n[2] TextView "{vid_title}"\n</screen>\n<thought>',
                "target": f'The video is currently playing. Task complete.</thought>\n<action>DONE("Playing {vid_title}")</action>'
            })

    # 5. Settings Configuration Workflows (Wi-Fi, Bluetooth)
    for _ in range(8):
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>turn on wifi</user>\n<screen>\n[SCREEN: com.android.settings]\n[1] TextView "Connections"\n[2] Switch "Wi-Fi (OFF)"\n[3] TextView "Bluetooth"\n</screen>\n<thought>',
            "target": f'I see the Wi-Fi toggle in the OFF state. I will tap to turn it on.</thought>\n<action>CLICK("Wi-Fi (OFF)")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>turn on wifi</user>\n<screen>\n[SCREEN: com.android.settings]\n[1] TextView "Connections"\n[2] Switch "Wi-Fi (ON)"\n</screen>\n<thought>',
            "target": f'Wi-Fi is now turned ON. Task complete.</thought>\n<action>DONE("Wi-Fi enabled")</action>'
        })

    # 6. Spotify Multi-Step
    sp_goal = "Play Hans Zimmer on Spotify"
    for _ in range(8):
        samples.append({
            "prompt": f"{SYSTEM_PROMPT}<user>{sp_goal}</user>\n<thought>",
            "target": f'User wants to listen to Hans Zimmer on Spotify. I will open Spotify.</thought>\n<action>OPEN_APP("Spotify")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{sp_goal}</user>\n<screen>\n[SCREEN: com.spotify.music]\n[1] Button "Search"\n[2] Button "Home"\n</screen>\n<thought>',
            "target": f'Spotify is open. I will tap Search.</thought>\n<action>CLICK("Search")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{sp_goal}</user>\n<screen>\n[SCREEN: com.spotify.music]\n[1] EditText "What do you want to listen to?"\n</screen>\n<thought>',
            "target": f'I will type Hans Zimmer into the search bar.</thought>\n<action>TYPE("What do you want to listen to?", "Hans Zimmer")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{sp_goal}</user>\n<screen>\n[SCREEN: com.spotify.music]\n[1] TextView "Hans Zimmer"\n[2] Button "Play"\n</screen>\n<thought>',
            "target": f'I see Hans Zimmer. I will tap Play.</thought>\n<action>CLICK("Play")</action>'
        })

    # 7. Samsung Notes Multi-Step
    note_goal = "Take a note about meeting at 3pm"
    for _ in range(8):
        samples.append({
            "prompt": f"{SYSTEM_PROMPT}<user>{note_goal}</user>\n<thought>",
            "target": f'User wants to create a note. I will open Samsung Notes.</thought>\n<action>OPEN_APP("Samsung Notes")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{note_goal}</user>\n<screen>\n[SCREEN: com.samsung.android.app.notes]\n[1] Button "Create note"\n</screen>\n<thought>',
            "target": f'I will tap Create note.</thought>\n<action>CLICK("Create note")</action>'
        })
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{note_goal}</user>\n<screen>\n[SCREEN: com.samsung.android.app.notes]\n[1] EditText "Title"\n[2] EditText "Note"\n</screen>\n<thought>',
            "target": f'I will enter the meeting details into the note.</thought>\n<action>TYPE("Note", "Meeting at 3pm")</action>'
        })

    # 8. Closed-Loop Negative Feedback Handling
    for _ in range(8):
        rec_goal = "Open Wi-Fi Settings"
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{rec_goal}</user>\n<screen>\n[SCREEN: com.embeddra.edge]\n[1] TextView "YORI"\n[ENVIRONMENT FEEDBACK: Previous action \'OPEN_APP("Wi-Fi Settings")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]\n</screen>\n<thought>',
            "target": f'The app was not found. I will open Wi-Fi settings directly using system settings.</thought>\n<action>OPEN_SETTINGS("WIFI")</action>'
        })
        rec_goal2 = "Turn on flashlight"
        samples.append({
            "prompt": f'{SYSTEM_PROMPT}<user>{rec_goal2}</user>\n<screen>\n[SCREEN: com.embeddra.edge]\n[1] TextView "YORI"\n[ENVIRONMENT FEEDBACK: Previous action \'OPEN_APP("Flashlight")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]\n</screen>\n<thought>',
            "target": f'I will toggle the hardware flashlight directly.</thought>\n<action>TORCH(1)</action>'
        })

    print(f"Total training samples synthesized: {len(samples)}")
    return samples

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 32, lora_alpha: int = 64):
        super().__init__()
        self.original_linear = original_linear
        for p in self.original_linear.parameters():
            p.requires_grad = False
        self.scaling = lora_alpha / r
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, original_linear.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(original_linear.out_features, r, dtype=dtype))
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original_linear(x)
        lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return orig + lora

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING MASTER UNIVERSAL AGENT ON {device} ===", flush=True)

    print(f"Loading base weights from {BASE_CKPT}...", flush=True)
    tokenizer = Tokenizer.from_file(TOK_PATH)
    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)

    for p in model.parameters():
        p.requires_grad = False

    # Attach fresh LoRA (rank 32, alpha 64) to all attention and FFN projections
    for block in model.blocks:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=32, lora_alpha=64).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=32, lora_alpha=64).to(device))

    print("Initialized fresh LoRA adapters (zero initialization, delta=0 at step 0).", flush=True)

    train_samples = build_universal_training_samples()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    MAX_STEPS = 350
    WARMUP_STEPS = 10
    PEAK_LR = 3.5e-5
    MIN_LR = 2.0e-6
    GRAD_ACCUM = 2

    optimizer = torch.optim.AdamW(trainable_params, lr=PEAK_LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        prog = float(step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return (MIN_LR / PEAK_LR) + (1.0 - (MIN_LR / PEAK_LR)) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Tokenize samples
    tokenized_samples = []
    for item in train_samples:
        p_ids = tokenizer.encode(item["prompt"]).ids
        t_ids = tokenizer.encode(item["target"]).ids + [128001]
        full_ids = p_ids + t_ids
        labels = [-100] * len(p_ids) + t_ids
        tokenized_samples.append((full_ids, labels))

    print(f"Starting training: {MAX_STEPS} steps with GradAccum={GRAD_ACCUM} (effective batch {GRAD_ACCUM}), {len(tokenized_samples)} samples...", flush=True)

    idx_pool = list(range(len(tokenized_samples)))
    random.shuffle(idx_pool)
    ptr = 0
    t0 = time.time()

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(GRAD_ACCUM):
            if ptr >= len(idx_pool):
                random.shuffle(idx_pool)
                ptr = 0
            full_ids, labels = tokenized_samples[idx_pool[ptr]]
            ptr += 1

            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)

            logits, _ = model(x)
            # Crucial: cast logits to float for numerical stability over 128k vocab
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100) / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item() * GRAD_ACCUM

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        avg_loss = step_loss / GRAD_ACCUM
        if step % 10 == 0 or step == 1 or step == MAX_STEPS:
            print(f"Step {step:3d}/{MAX_STEPS} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Elapsed: {time.time() - t0:.1f}s", flush=True)

    print("Training finished! Saving LoRA weights...", flush=True)
    lora_save_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_save_dict, OUT_LORA)

    print("Merging LoRA weights back into base model in FP32...", flush=True)
    model.eval()
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

    print(f"Saving merged FP16 PyTorch model to {OUT_PT}...", flush=True)
    clean_sd = {k: v.cpu() for k, v in model.state_dict().items() if "original_linear." not in k and "lora_" not in k}
    fixed_sd = {}
    for k, v in model.state_dict().items():
        if "lora_" in k: continue
        clean_key = k.replace(".original_linear.", ".")
        fixed_sd[clean_key] = v.cpu()
    torch.save(fixed_sd, OUT_PT)

    print(f"Serializing to EER binary format: {OUT_EER}...", flush=True)
    serialize_checkpoint_to_eer(OUT_PT, OUT_EER)
    print(f"Successfully generated {OUT_EER}!", flush=True)

if __name__ == "__main__":
    main()
