import os
import json
import random
import requests

CACHE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
ALPACA_PATH = os.path.join(CACHE_DIR, "alpaca_data_cleaned.json")
DOLLY_PATH = os.path.join(CACHE_DIR, "databricks-dolly-15k.jsonl")

UI_SOURCES = [
    r"D:\embeddra_eval\android_agent\train_reasoning_agent.jsonl",
    r"D:\embeddra_eval\android_agent\train_autonomous_agent.jsonl",
    r"D:\embeddra_eval\android_agent\train_idempotent_agent.jsonl",
]

OUT_TRAIN = os.path.join(CACHE_DIR, "yori_unified_train.jsonl")
OUT_VAL = os.path.join(CACHE_DIR, "yori_unified_val.jsonl")

def generate_expanded_tool_calling():
    examples = []

    def add(user_variants, thought, action):
        for u in user_variants:
            examples.append({
                "category": "tool_call",
                "prompt": f"<user>{u}</user>\n<thought>",
                "target": f"{thought}</thought>\n<action>{action}</action>"
            })

    # 1. Flashlight / Torch
    add([
        "Turn on flashlight", "Flashlight on", "It's dark in here, turn on the light",
        "Switch on my phone torch", "Enable torchlight", "Can you turn the flashlight on?",
        "Light up the room please", "Turn on the back flash", "Activate flashlight",
        "Need light, switch on torch", "Open torch", "Turn the flash on"
    ], "User wants to enable the device flashlight.", 'TORCH("ON")')

    add([
        "Turn off flashlight", "Flashlight off", "Switch off the torch",
        "Disable flashlight", "Kill the torch light", "Turn off the phone light please",
        "Shut off flashlight", "Deactivate torch", "Stop flashlight"
    ], "User wants to disable the device flashlight.", 'TORCH("OFF")')

    # 2. Volume & Audio
    add([
        "Volume up", "Increase volume", "Turn up the sound", "Make it louder",
        "Raise the media volume", "Too quiet, boost the volume", "Turn volume higher",
        "Sound up please", "Louder please"
    ], "User wants to increase media volume.", 'VOLUME("UP")')

    add([
        "Volume down", "Decrease volume", "Turn down the sound", "Make it quieter",
        "Lower the volume please", "Too loud, drop the volume", "Turn volume lower",
        "Reduce sound", "Quieter please"
    ], "User wants to decrease media volume.", 'VOLUME("DOWN")')

    add([
        "Mute phone", "Mute volume", "Silence the audio", "Turn off sound",
        "Mute all media", "Silence phone audio", "Put audio on mute"
    ], "User wants to mute device audio.", 'VOLUME("MUTE")')

    add([
        "Unmute phone", "Unmute volume", "Turn sound back on", "Restore audio",
        "Bring sound back", "Take phone off mute"
    ], "User wants to unmute device audio.", 'VOLUME("UNMUTE")')

    # 3. Media Playback
    add(["Pause music", "Stop the song", "Pause playback", "Pause audio", "Hold the music", "Pause the track"],
        "User wants to pause media playback.", 'MEDIA("PAUSE")')
    add(["Play music", "Resume song", "Continue playing", "Resume music", "Start music again", "Unpause track"],
        "User wants to resume media playback.", 'MEDIA("PLAY")')
    add(["Next song", "Skip this track", "Play next track", "Skip song", "Forward to next song", "Next track please"],
        "User wants to skip to the next track.", 'MEDIA("NEXT")')
    add(["Previous song", "Go back to last song", "Previous track", "Replay last song", "Back one track"],
        "User wants to play the previous track.", 'MEDIA("PREV")')

    # 4. Battery & Telemetry
    add([
        "Check battery", "How much battery is left?", "What is my battery percentage?",
        "Check phone battery and RAM", "How much juice is left?", "Is my phone charging?",
        "Show battery level", "Check device battery status", "How much power left?"
    ], "User wants live battery and memory telemetry.", 'BATTERY()')

    # 5. System Actions & Navigation
    add(["Take a screenshot", "Screenshot this", "Capture my screen", "Snap a pic of the screen", "Save screen image", "Take screen capture"],
        "User wants to capture a screenshot.", 'SCREENSHOT()')
    add(["Lock the phone", "Lock screen", "Turn off screen and lock", "Lock my device", "Sleep and lock phone"],
        "User wants to lock the device screen.", 'LOCK_SCREEN()')
    add(["Go home", "Go to home screen", "Close this app and go home", "Back to launcher", "Show home screen"],
        "User wants to navigate to the Home screen.", 'GLOBAL_HOME()')
    add(["Go back", "Back please", "Return to previous screen", "Navigate back one step"],
        "User wants to navigate Back.", 'GLOBAL_BACK()')
    add(["Open recent apps", "Show app switcher", "Show recents", "View open apps"],
        "User wants to view recent applications.", 'GLOBAL_RECENTS()')
    add(["Open notifications", "Pull down notification shade", "Check notifications", "Show my alerts"],
        "User wants to open the notifications panel.", 'GLOBAL_NOTIFICATIONS()')
    add(["Open quick settings", "Show quick settings tiles", "Pull down control center"],
        "User wants to open Quick Settings.", 'GLOBAL_QUICK_SETTINGS()')
    add(["Scroll down", "Move down the page", "Swipe down", "Page down"],
        "User wants to scroll down the current screen.", 'SCROLL_DOWN()')
    add(["Scroll up", "Move up the page", "Swipe up", "Page up"],
        "User wants to scroll up the current screen.", 'SCROLL_UP()')

    # 6. Calls, WhatsApp, SMS
    contacts = ["Bilal", "Mom", "Dad", "Sarah", "Alex", "John", "Zahid", "Tech Support", "Doctor", "Manager", "Ayesha", "Omar", "Ali", "Emma"]
    messages = [
        "Hey, I will be there in 10 minutes",
        "Running a bit late, see you soon",
        "Hey, I'm Yori an AI agent by Embeddra AI",
        "Can we talk later? In a meeting right now",
        "Reached home safely",
        "Please check your email",
        "Happy birthday!",
        "Call me when free",
        "Send me the location please",
        "On my way now"
    ]
    for c in contacts:
        add([
            f"Call {c}", f"Dial {c}", f"Ring up {c} please",
            f"Make a phone call to {c}", f"Phone {c}"
        ], f"User wants to place a phone call to {c}.", f'CALL("{c}")')

        for m in random.sample(messages, 5):
            add([
                f"Send a WhatsApp message to {c} saying {m}",
                f"Message {c} on WhatsApp: {m}",
                f"Tell {c} on WhatsApp that {m}",
                f"WhatsApp {c} saying {m}"
            ], f"User wants to send a WhatsApp message to {c}.", f'WHATSAPP("{c}", "{m}")')

            add([
                f"Send SMS to {c} saying {m}",
                f"Text {c} via SMS: {m}"
            ], f"User wants to send an SMS to {c}.", f'SMS("{c}", "{m}")')

    # 7. Alarms & Timers
    times = [
        ("05:30", "5:30 AM"), ("06:00", "6 AM"), ("06:30", "6:30 AM"), ("07:00", "7 AM"),
        ("07:45", "7:45 AM"), ("08:15", "8:15 AM"), ("09:30", "9:30 AM"), ("14:00", "2 PM"),
        ("18:00", "6 PM"), ("21:30", "9:30 PM"), ("22:00", "10 PM")
    ]
    for t24, t_human in times:
        add([
            f"Set an alarm for {t_human}",
            f"Wake me up at {t_human}",
            f"Create an alarm at {t_human}",
            f"Alarm for {t_human} please"
        ], f"User wants to set an alarm for {t24}.", f'SET_ALARM("{t24}")')

    timers = [
        ("30", "30 seconds"), ("45", "45 seconds"), ("60", "1 minute"),
        ("120", "2 minutes"), ("300", "5 minutes"), ("600", "10 minutes"),
        ("900", "15 minutes"), ("1800", "30 minutes"), ("3600", "1 hour")
    ]
    for secs, t_str in timers:
        add([
            f"Set a timer for {t_str}", f"Start a {t_str} timer",
            f"Timer for {t_str} please", f"Count down {t_str}"
        ], f"User wants a countdown timer for {t_str}.", f'SET_TIMER("{secs}")')

    # 8. Apps & Settings & Web Search
    apps = [
        "WhatsApp", "YouTube", "Chrome", "Spotify", "Instagram", "Camera",
        "Calculator", "Settings", "Clock", "Gmail", "Maps", "Gallery", "Notes", "Calendar"
    ]
    for app in apps:
        add([f"Open {app}", f"Launch {app}", f"Start {app} app", f"Can you open {app}?", f"Switch to {app}"],
            f"User wants to launch the {app} application.", f'OPEN_APP("{app}")')

    settings_targets = [
        ("Wi-Fi", "WIFI"), ("Bluetooth", "BLUETOOTH"), ("Flight mode", "FLIGHT_MODE"),
        ("Display", "DISPLAY"), ("Sound", "SOUND"), ("Location", "LOCATION"),
        ("Storage", "STORAGE"), ("Battery", "BATTERY"), ("Security", "SECURITY")
    ]
    for label, key in settings_targets:
        add([f"Open {label} settings", f"Go to {label} settings", f"Show {label} options", f"Check {label} in settings"],
            f"User wants to open {label} settings.", f'OPEN_SETTINGS("{key}")')

    queries = [
        "quantum computing breakthroughs", "weather in Tokyo today", "who won the cricket world cup",
        "how to make espresso at home", "latest AI robotics news", "Python asyncio tutorial",
        "best restaurants near me", "capital of Australia", "stock price of Nvidia",
        "how does photosynthesis work", "upcoming SpaceX launch"
    ]
    for q in queries:
        add([f"Search {q}", f"Google {q}", f"Look up {q} on the web", f"Search Google for {q}"],
            f"User wants to search Google for '{q}'.", f'SEARCH("{q}")')

    return examples

def main():
    random.seed(42)

    # 1. Chat (~1,600 train, ~120 val -> ~45% of mix)
    with open(ALPACA_PATH, "r", encoding="utf-8") as f:
        alpaca = json.load(f)
    chat_pool = []
    for row in alpaca:
        instr = row["instruction"].strip()
        inp = row["input"].strip()
        out = row["output"].strip()
        if not instr or not out or len(out) < 20 or len(out) > 380:
            continue
        user_msg = f"{instr}\n{inp}".strip() if inp else instr
        chat_pool.append({
            "category": "chat",
            "prompt": f"<user>{user_msg}</user>\n<thought>",
            "target": f"{out}</thought>"
        })

    if os.path.exists(DOLLY_PATH):
        with open(DOLLY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row = json.loads(line)
                if row.get("category") not in ["open_qa", "general_qa", "brainstorming", "summarization", "creative_writing"]:
                    continue
                instr = row["instruction"].strip()
                ctx = row.get("context", "").strip()
                resp = row["response"].strip()
                if not instr or not resp or len(resp) < 20 or len(resp) > 380:
                    continue
                user_msg = f"{instr}\nContext: {ctx[:300]}" if ctx else instr
                chat_pool.append({
                    "category": "chat",
                    "prompt": f"<user>{user_msg}</user>\n<thought>",
                    "target": f"{resp}</thought>"
                })

    random.shuffle(chat_pool)
    chat_train = chat_pool[:1500]
    chat_val = chat_pool[1500:1620]

    # Add 100 YORI identity & negative boundary examples
    yori_pairs = [
        ("Hi", "Hey! I'm Yori, your local AI assistant by Embeddra AI. How can I help you today?"),
        ("Hello Yori", "Hello! I'm running 100% locally on your device. Ask me anything or tell me what to control on your phone!"),
        ("Who are you?", "I'm Yori, an offline executive AI assistant built by Embeddra AI to chat locally and control your phone."),
        ("Who built you?", "I was created by Embeddra AI to run natively on smartphones and laptops without cloud servers."),
        ("What can you do?", "I can chat offline, answer questions, summarize text, write messages, and control your phone—opening apps, toggling flashlight or volume, making calls, setting alarms, and navigating screens."),
        ("Explain what a flashlight is", "A flashlight is a portable hand-held electric light powered by batteries, usually using an LED bulb and reflector."),
        ("How do smartphone batteries work?", "Smartphone batteries are lithium-ion cells that store chemical energy and move lithium ions between electrodes to power your device."),
        ("What is WhatsApp?", "WhatsApp is a popular cross-platform messaging and voice calling app that uses end-to-end encryption."),
    ]
    for _ in range(12):
        for u, a in yori_pairs:
            chat_train.append({"category": "chat", "prompt": f"<user>{u}</user>\n<thought>", "target": f"{a}</thought>"})

    # 2. Tool Calling (~1,250 train, ~140 val -> ~34% of mix)
    tool_pool = generate_expanded_tool_calling()
    random.shuffle(tool_pool)
    split_t = int(len(tool_pool) * 0.90)
    tool_train = tool_pool[:split_t]
    tool_val = tool_pool[split_t:]

    # 3. Diversified UI Grounding (~800 train, ~80 val -> ~21% of mix)
    ui_pool = []
    for src in UI_SOURCES:
        if not os.path.exists(src): continue
        with open(src, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 450 or not line.strip(): break
                traj = json.loads(line)
                intent = traj["user_intent"]
                turns = traj.get("turns", [])
                if turns:
                    turn = random.choice(turns)
                    thought = turn.get("thought", f"Analyzing elements to achieve: {intent}")
                    ui_pool.append({
                        "category": "ui_grounding",
                        "prompt": f"<user>{intent}</user>\n<screen>\n{turn['screen']}\n</screen>\n<thought>",
                        "target": f"{thought}</thought>\n<action>{turn['action']}</action>"
                    })
    random.shuffle(ui_pool)
    ui_train = ui_pool[:800]
    ui_val = ui_pool[800:880]

    train_all = chat_train + tool_train + ui_train
    val_all = chat_val + tool_val + ui_val
    random.shuffle(train_all)

    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in train_all:
            f.write(json.dumps(ex) + "\n")
    with open(OUT_VAL, "w", encoding="utf-8") as f:
        for ex in val_all:
            f.write(json.dumps(ex) + "\n")

    n_tr = len(train_all)
    print("=" * 70)
    print("REBALANCED UNIFIED DATASET READY")
    print(f"  Train Total:  {n_tr}")
    print(f"    chat:         {len(chat_train):4d} ({100*len(chat_train)/n_tr:.1f}%)")
    print(f"    tool_call:    {len(tool_train):4d} ({100*len(tool_train)/n_tr:.1f}%)")
    print(f"    ui_grounding: {len(ui_train):4d} ({100*len(ui_train)/n_tr:.1f}%)")
    print(f"  Val Total:    {len(val_all)} (chat={len(chat_val)}, tool={len(tool_val)}, ui={len(ui_val)})")
    print("=" * 70)

if __name__ == "__main__":
    main()
