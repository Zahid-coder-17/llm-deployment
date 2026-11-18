import json
import os
import hashlib
from collections import Counter

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
OUT_PATH = os.path.join(BASE_DIR, "yori_pristine_master_dataset.jsonl")

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

samples = []

def add_sample(cat, query, thought, action=None, screen=None, history=None):
    prompt = SYSTEM_CONTRACT
    if history:
        prompt += f"<history>\n{history.strip()}\n</history>\n"
    prompt += f"<user>{query.strip()}</user>\n"
    if screen:
        prompt += f"<screen>\n{screen.strip()}\n</screen>\n"
    prompt += "<thought>\n"

    target = f"{thought.strip()}\n</thought>"
    if action:
        target += f"\n<action>{action.strip()}</action>"

    full_text = prompt + target
    samples.append({
        "category": cat,
        "prompt": prompt,
        "target": target,
        "full_text": full_text
    })

# =========================================================================
# 1. DIRECT CONTROL - EQUAL BALANCE ACROSS ALL 10 TOOLS (15-20 EACH)
# =========================================================================

# --- TORCH (16 samples: 8 ON, 8 OFF) ---
torch_queries_on = [
    ("Turn on the flashlight", "1"),
    ("Torch on", "1"),
    ("Turn on torch", "1"),
    ("Enable the flashlight", "1"),
    ("Turn on phone light", "1"),
    ("Activate flashlight", "1"),
    ("Switch on the torch", "1"),
    ("Turn on back light", "1")
]
for q, val in torch_queries_on:
    add_sample("direct_control", q,
               f"🎯 Goal: Enable flashlight\n📋 Plan: 1. Turn on device torch\n⚡ Immediate Action: Enabling flashlight.",
               action=f"TORCH({val})")

torch_queries_off = [
    ("Turn off the flashlight", "0"),
    ("Torch off", "0"),
    ("Turn off torch", "0"),
    ("Disable the flashlight", "0"),
    ("Turn off phone light", "0"),
    ("Deactivate flashlight", "0"),
    ("Switch off the torch", "0"),
    ("Kill flashlight", "0")
]
for q, val in torch_queries_off:
    add_sample("direct_control", q,
               f"🎯 Goal: Disable flashlight\n📋 Plan: 1. Turn off device torch\n⚡ Immediate Action: Disabling flashlight.",
               action=f"TORCH({val})")

# --- SET_ALARM (16 samples) ---
alarm_queries = [
    ("Set an alarm for 7:30 AM", "07:30"),
    ("Set an alarm for 6:00 AM", "06:00"),
    ("Wake me up at 8:00 AM", "08:00"),
    ("Set alarm for 5:45 AM", "05:45"),
    ("Set an alarm for 7:00 AM tomorrow", "07:00"),
    ("Wake me up at 6:30 tomorrow morning", "06:30"),
    ("Alarm at 9:15 AM", "09:15"),
    ("Set alarm for 10:00 AM", "10:00"),
    ("Set alarm for 11:30 PM", "23:30"),
    ("Set alarm for 10:00 PM", "22:00"),
    ("Alarm at 23:00", "23:00"),
    ("Set an alarm for 6:15 AM", "06:15"),
    ("Wake me up at 5:00 AM sharp", "05:00"),
    ("Set alarm for 8:30 AM", "08:30"),
    ("Alarm for 7:45 AM", "07:45"),
    ("Set an alarm for 9:00 PM", "21:00")
]
for q, t_val in alarm_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Schedule alarm for {t_val}\n📋 Plan: 1. Dispatch clock alarm service\n⚡ Immediate Action: Setting system alarm for {t_val}.",
               action=f"SET_ALARM(\"{t_val}\")")

# --- VOLUME (16 samples) ---
volume_queries = [
    ("Mute the volume", "MUTE"),
    ("Put phone on silent", "MUTE"),
    ("Mute all audio", "MUTE"),
    ("Silence phone", "MUTE"),
    ("Turn off sound", "MUTE"),
    ("Mute media sound", "MUTE"),
    ("Turn up the volume", "UP"),
    ("Volume up", "UP"),
    ("Increase volume", "UP"),
    ("Louder sound", "UP"),
    ("Raise volume level", "UP"),
    ("Turn down the volume", "DOWN"),
    ("Volume down", "DOWN"),
    ("Decrease volume", "DOWN"),
    ("Lower sound level", "DOWN"),
    ("Reduce volume", "DOWN")
]
for q, val in volume_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Adjust system volume\n📋 Plan: 1. Apply volume change\n⚡ Immediate Action: Updating volume to {val}.",
               action=f"VOLUME({val})")

# --- OPEN_APP (20 samples) ---
app_queries = [
    ("Open YouTube", "YouTube"),
    ("Launch Spotify", "Spotify"),
    ("Open WhatsApp", "WhatsApp"),
    ("Open Camera", "Camera"),
    ("Open Settings", "Settings"),
    ("Launch Google Chrome", "Chrome"),
    ("Open Clock", "Clock"),
    ("Open Calculator", "Calculator"),
    ("Launch Gmail", "Gmail"),
    ("Open Google Maps", "Maps"),
    ("Open Gallery", "Gallery"),
    ("Open Telegram", "Telegram"),
    ("Launch Instagram", "Instagram"),
    ("Open Messages app", "Messages"),
    ("Open Phone dialer", "Phone"),
    ("Open Files app", "Files"),
    ("Launch Twitter", "X"),
    ("Open Notes", "Notes"),
    ("Open Calendar", "Calendar"),
    ("Launch Uber", "Uber")
]
for q, app in app_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Launch {app}\n📋 Plan: 1. Start application\n⚡ Immediate Action: Opening {app}.",
               action=f"OPEN_APP(\"{app}\")")

# --- OPEN_SETTINGS (16 samples) ---
settings_queries = [
    ("Open WiFi settings", "WiFi"),
    ("Go to Bluetooth settings", "Bluetooth"),
    ("Open Display settings", "Display"),
    ("Open Battery settings", "Battery"),
    ("Go to Sound settings", "Sound"),
    ("Open App permissions", "Apps"),
    ("Open Storage settings", "Storage"),
    ("Open Network settings", "Network"),
    ("Open Hotspot settings", "Hotspot"),
    ("Open Notification settings", "Notifications"),
    ("Open Location settings", "Location"),
    ("Go to Security settings", "Security"),
    ("Open Accessibility settings", "Accessibility"),
    ("Open Accounts settings", "Accounts"),
    ("Open Date and Time settings", "DateTime"),
    ("Open Developer options", "Developer")
]
for q, s_val in settings_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Navigate to {s_val} settings\n📋 Plan: 1. Launch system settings subpage\n⚡ Immediate Action: Opening {s_val} settings directly.",
               action=f"OPEN_SETTINGS(\"{s_val}\")")

# --- CLICK (16 samples with screens) ---
click_queries = [
    ("Click Next", '[SCREEN: com.example.app]\n[1] Button "Next"', 'CLICK("Next")'),
    ("Tap Submit button", '[SCREEN: com.example.app]\n[1] Button "Submit"', 'CLICK("Submit")'),
    ("Click Cancel", '[SCREEN: com.example.app]\n[1] Button "Cancel"', 'CLICK("Cancel")'),
    ("Tap Install", '[SCREEN: com.android.vending]\n[1] Button "Install"', 'CLICK("Install")'),
    ("Click on Profile icon", '[SCREEN: com.example.app]\n[1] ImageView "Profile"', 'CLICK("Profile")'),
    ("Tap Save", '[SCREEN: com.example.app]\n[1] Button "Save"', 'CLICK("Save")'),
    ("Click Agree and Continue", '[SCREEN: com.whatsapp.w4b]\n[1] Button "Agree and continue"', 'CLICK("Agree and continue")'),
    ("Click Confirm order", '[SCREEN: com.example.shop]\n[1] Button "Confirm Order"', 'CLICK("Confirm Order")'),
    ("Tap search button", '[SCREEN: com.spotify.music]\n[1] Button "Search"', 'CLICK("Search")'),
    ("Click Play button", '[SCREEN: com.google.android.youtube]\n[1] Button "Play"', 'CLICK("Play")'),
    ("Tap Like button", '[SCREEN: com.instagram.android]\n[1] Button "Like"', 'CLICK("Like")'),
    ("Click Clear All", '[SCREEN: com.sec.android.app.launcher]\n[1] Button "Clear all"', 'CLICK("Clear all")'),
    ("Tap Bluetooth toggle switch", '[SCREEN: com.android.settings]\n[1] Switch "Bluetooth" (OFF)', 'CLICK("Bluetooth")'),
    ("Tap Dark theme switch", '[SCREEN: com.android.settings]\n[1] Switch "Dark theme" (OFF)', 'CLICK("Dark theme")'),
    ("Click Send message", '[SCREEN: com.whatsapp.w4b]\n[1] Button "Send"', 'CLICK("Send")'),
    ("Click Download button", '[SCREEN: com.google.android.apps.docs]\n[1] Button "Download"', 'CLICK("Download")')
]
for q, scr, act in click_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Interact with target element\n📋 Plan: 1. Locate element on visible screen 2. Tap it\n⚡ Immediate Action: Clicking element on screen.",
               action=act, screen=scr)

# --- TYPE (15 samples with screens) ---
type_queries = [
    ("Type hello in search box", '[SCREEN: com.example.app]\n[1] EditText "Search" [Input]', 'TYPE("Search", "hello")'),
    ("Enter my email user@embeddra.com in email field", '[SCREEN: com.example.app]\n[1] EditText "Email" [Input]', 'TYPE("Email", "user@embeddra.com")'),
    ("Type 'See you at 5' in message box", '[SCREEN: com.whatsapp.w4b]\n[1] EditText "Message" [Input]', 'TYPE("Message", "See you at 5")'),
    ("Search for lofi beats in Spotify", '[SCREEN: com.spotify.music]\n[1] EditText "Search songs, artists" [Input]', 'TYPE("Search songs, artists", "lofi beats")'),
    ("Type 'espresso machine' in search bar", '[SCREEN: com.amazon.mShop]\n[1] EditText "Search Amazon" [Input]', 'TYPE("Search Amazon", "espresso machine")'),
    ("Enter code 123456 into verification box", '[SCREEN: com.example.auth]\n[1] EditText "Code" [Input]', 'TYPE("Code", "123456")'),
    ("Type Bilal in contact search", '[SCREEN: com.whatsapp.w4b]\n[1] EditText "Search contacts" [Input]', 'TYPE("Search contacts", "Bilal")'),
    ("Type 'Meeting notes' as title", '[SCREEN: com.google.android.keep]\n[1] EditText "Title" [Input]', 'TYPE("Title", "Meeting notes")'),
    ("Type 'Ready in 10 mins' into chat", '[SCREEN: com.whatsapp.w4b]\n[1] EditText "Type a message" [Input]', 'TYPE("Type a message", "Ready in 10 mins")'),
    ("Search for Interstellar movie on YouTube", '[SCREEN: com.google.android.youtube]\n[1] EditText "Search YouTube" [Input]', 'TYPE("Search YouTube", "Interstellar")'),
    ("Type destination 'Airport Terminal 2'", '[SCREEN: com.ubercab]\n[1] EditText "Where to?" [Input]', 'TYPE("Where to?", "Airport Terminal 2")'),
    ("Enter username 'john_doe'", '[SCREEN: com.example.app]\n[1] EditText "Username" [Input]', 'TYPE("Username", "john_doe")'),
    ("Type 'Buy milk and eggs' into checklist", '[SCREEN: com.google.android.keep]\n[1] EditText "Note" [Input]', 'TYPE("Note", "Buy milk and eggs")'),
    ("Type 'Thank you for your help' in reply", '[SCREEN: com.google.android.gm]\n[1] EditText "Compose email" [Input]', 'TYPE("Compose email", "Thank you for your help")'),
    ("Enter password 'SecurePass123'", '[SCREEN: com.example.app]\n[1] EditText "Password" [Input]', 'TYPE("Password", "SecurePass123")')
]
for q, scr, act in type_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Enter text into input field\n📋 Plan: 1. Locate editable field 2. Type specified value\n⚡ Immediate Action: Entering text into target field.",
               action=act, screen=scr)

# --- SCROLL (12 samples) ---
scroll_queries = [
    ("Scroll down", "DOWN"),
    ("Swipe down for more content", "DOWN"),
    ("Scroll down the page", "DOWN"),
    ("Scroll down to see more items", "DOWN"),
    ("Scroll down the feed", "DOWN"),
    ("Scroll down the chat list", "DOWN"),
    ("Scroll up to top", "UP"),
    ("Swipe up", "UP"),
    ("Scroll up to previous messages", "UP"),
    ("Scroll up the page", "UP"),
    ("Scroll left to next tab", "LEFT"),
    ("Scroll right to previous card", "RIGHT")
]
for q, d in scroll_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Scroll screen {d.lower()}\n📋 Plan: 1. Issue scroll gesture\n⚡ Immediate Action: Scrolling screen {d}.",
               action=f"SCROLL({d})")

# --- PRESS_KEY (12 samples) ---
key_queries = [
    ("Go back", "BACK"),
    ("Press back button", "BACK"),
    ("Return to previous screen", "BACK"),
    ("Navigate back", "BACK"),
    ("Hit back", "BACK"),
    ("Press home button", "HOME"),
    ("Go to homescreen", "HOME"),
    ("Return to home", "HOME"),
    ("Minimize all apps and go home", "HOME"),
    ("Press enter", "ENTER"),
    ("Hit enter key", "ENTER"),
    ("Submit by pressing enter", "ENTER")
]
for q, k in key_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Dispatch system key event {k}\n📋 Plan: 1. Send hardware navigation key {k}\n⚡ Immediate Action: Pressing {k} button.",
               action=f"PRESS_KEY({k})")

# --- DONE (12 samples) ---
done_queries = [
    ("We are done with the workflow", "Task execution finished"),
    ("Everything is completed", "All actions executed successfully"),
    ("Finish task", "Workflow completed"),
    ("Stop execution, we reached the goal", "Goal reached"),
    ("The message was sent and confirmed", "Message sent and confirmed"),
    ("Alarm has been confirmed", "Alarm created successfully"),
    ("Spotify is playing the track", "Music playback started"),
    ("WiFi has been successfully toggled", "WiFi setting updated"),
    ("Contact has been saved", "New contact saved to address book"),
    ("Note has been recorded", "Note successfully stored"),
    ("Settings have been updated", "System configuration applied"),
    ("All steps finished", "Multi-step flow concluded")
]
for q, summ in done_queries:
    add_sample("direct_control", q,
               f"🎯 Goal: Conclude task\n📋 Plan: 1. Confirm objective completion 2. Terminate workflow\n⚡ Immediate Action: Completed.",
               action=f"DONE(\"{summ}\")")

# =========================================================================
# 2. MULTI-STEP NAVIGATION TRAJECTORIES (45 SAMPLES)
# =========================================================================

# Scenario A: WhatsApp Message Reply (5 Turns)
q_wa = "Summarize message from Bilal on WhatsApp and reply yes"
add_sample("multistep_nav", q_wa,
           "🎯 Goal: Open WhatsApp to read Bilal's message\n📋 Plan: 1. Launch WhatsApp 2. Find Bilal's chat\n⚡ Immediate Action: Opening WhatsApp from launcher.",
           action='OPEN_APP("WhatsApp")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "WhatsApp"\n[2] Button "Messages"',
           history=None)

add_sample("multistep_nav", q_wa,
           "🎯 Goal: Open Bilal's chat thread\n📋 Plan: Select Bilal's message from chat list\n⚡ Immediate Action: Tapping Bilal's conversation.",
           action='CLICK("Bilal: Are we meeting at 4 PM?")',
           screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "WhatsApp"\n[2] TextView "Bilal: Are we meeting at 4 PM?"\n[3] TextView "Mom: Call me"',
           history='Step 1: OPEN_APP("WhatsApp")')

add_sample("multistep_nav", q_wa,
           "🎯 Goal: Type confirmation reply\n📋 Plan: Enter 'Yes, see you at 4' into input\n⚡ Immediate Action: Typing reply into message field.",
           action='TYPE("Message", "Yes, see you at 4")',
           screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "Bilal"\n[2] TextView "Are we meeting at 4 PM?"\n[3] EditText "Message" [Input]',
           history='Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal: Are we meeting at 4 PM?")')

add_sample("multistep_nav", q_wa,
           "🎯 Goal: Send typed reply\n📋 Plan: Tap send button\n⚡ Immediate Action: Clicking send button.",
           action='CLICK("Send")',
           screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, see you at 4" [Typed]\n[2] Button "Send"',
           history='Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal") -> Step 3: TYPE("Message", "Yes, see you at 4")')

add_sample("multistep_nav", q_wa,
           "🎯 Goal: Confirm delivery and finish\n📋 Plan: Verify delivery checkmark and conclude\n⚡ Immediate Action: Workflow completed.",
           action='DONE("Replied to Bilal on WhatsApp: Yes, see you at 4")',
           screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, see you at 4"\n[2] ImageView "Delivered"',
           history='Step 1: OPEN_APP(...) -> Step 2: CLICK(...) -> Step 3: TYPE(...) -> Step 4: CLICK("Send")')

# Scenario B: Spotify Search & Play (4 Turns)
q_sp = "Play lofi study beats on Spotify"
add_sample("multistep_nav", q_sp,
           "🎯 Goal: Open Spotify\n📋 Plan: 1. Launch Spotify app\n⚡ Immediate Action: Launching Spotify.",
           action='OPEN_APP("Spotify")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "Spotify"\n[2] Button "YouTube"',
           history=None)

add_sample("multistep_nav", q_sp,
           "🎯 Goal: Navigate to search in Spotify\n📋 Plan: Tap search tab\n⚡ Immediate Action: Tapping search navigation tab.",
           action='CLICK("Search Tab")',
           screen='[SCREEN: com.spotify.music]\n[1] TextView "Good morning"\n[2] Button "Search Tab"\n[3] Button "Your Library"',
           history='Step 1: OPEN_APP("Spotify")')

add_sample("multistep_nav", q_sp,
           "🎯 Goal: Search for lofi study beats\n📋 Plan: Type query into search bar\n⚡ Immediate Action: Typing 'lofi study beats' into search field.",
           action='TYPE("Search songs, artists", "lofi study beats")',
           screen='[SCREEN: com.spotify.music]\n[1] EditText "Search songs, artists" [Input]',
           history='Step 1: OPEN_APP("Spotify") -> Step 2: CLICK("Search Tab")')

add_sample("multistep_nav", q_sp,
           "🎯 Goal: Play top lofi result\n📋 Plan: Tap play on top search result\n⚡ Immediate Action: Tapping play on 'Lofi Beats - Peaceful Study'.",
           action='CLICK("Play Lofi Beats")',
           screen='[SCREEN: com.spotify.music]\n[1] TextView "Lofi Beats - Peaceful Study"\n[2] Button "Play Lofi Beats"',
           history='Step 1: OPEN_APP("Spotify") -> Step 2: CLICK("Search Tab") -> Step 3: TYPE("Search songs, artists", "lofi study beats")')

# Scenario C: YouTube Search & Video Play (4 Turns)
q_yt = "Search for Quantum Computing lecture on YouTube and play it"
add_sample("multistep_nav", q_yt,
           "🎯 Goal: Open YouTube\n📋 Plan: 1. Launch YouTube app\n⚡ Immediate Action: Launching YouTube.",
           action='OPEN_APP("YouTube")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "YouTube"',
           history=None)

add_sample("multistep_nav", q_yt,
           "🎯 Goal: Open search bar in YouTube\n📋 Plan: Tap search button\n⚡ Immediate Action: Clicking search icon.",
           action='CLICK("Search")',
           screen='[SCREEN: com.google.android.youtube]\n[1] ImageView "Logo"\n[2] Button "Search"\n[3] Button "Notifications"',
           history='Step 1: OPEN_APP("YouTube")')

add_sample("multistep_nav", q_yt,
           "🎯 Goal: Enter video search term\n📋 Plan: Type 'Quantum Computing' into search input\n⚡ Immediate Action: Typing search term.",
           action='TYPE("Search YouTube", "Quantum Computing")',
           screen='[SCREEN: com.google.android.youtube]\n[1] EditText "Search YouTube" [Input]',
           history='Step 1: OPEN_APP("YouTube") -> Step 2: CLICK("Search")')

add_sample("multistep_nav", q_yt,
           "🎯 Goal: Play target lecture video\n📋 Plan: Click on top video lecture\n⚡ Immediate Action: Tapping video result.",
           action='CLICK("Quantum Computing Full Course - MIT")',
           screen='[SCREEN: com.google.android.youtube]\n[1] TextView "Quantum Computing Full Course - MIT"\n[2] TextView "2.4M views"',
           history='Step 1: OPEN_APP("YouTube") -> Step 2: CLICK("Search") -> Step 3: TYPE("Search YouTube", "Quantum Computing")')

# Scenario D: Toggle Bluetooth in Settings (3 Turns)
q_bt = "Turn on Bluetooth in Settings"
add_sample("multistep_nav", q_bt,
           "🎯 Goal: Open Bluetooth settings\n📋 Plan: 1. Open Bluetooth settings directly\n⚡ Immediate Action: Opening Bluetooth settings page.",
           action='OPEN_SETTINGS("Bluetooth")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "Settings"',
           history=None)

add_sample("multistep_nav", q_bt,
           "🎯 Goal: Enable Bluetooth switch\n📋 Plan: Toggle switch from OFF to ON\n⚡ Immediate Action: Tapping Bluetooth toggle.",
           action='CLICK("Bluetooth Switch")',
           screen='[SCREEN: com.android.settings]\n[1] TextView "Bluetooth"\n[2] Switch "Bluetooth Switch" (OFF)',
           history='Step 1: OPEN_SETTINGS("Bluetooth")')

add_sample("multistep_nav", q_bt,
           "🎯 Goal: Confirm Bluetooth active\n📋 Plan: Verify switch is ON and finish\n⚡ Immediate Action: Task complete.",
           action='DONE("Bluetooth is now turned ON")',
           screen='[SCREEN: com.android.settings]\n[1] TextView "Bluetooth"\n[2] Switch "Bluetooth Switch" (ON)',
           history='Step 1: OPEN_SETTINGS("Bluetooth") -> Step 2: CLICK("Bluetooth Switch")')

# Scenario E: Send Email Reply in Gmail (4 Turns)
q_em = "Reply to Alex's email on Gmail saying 'Draft approved'"
add_sample("multistep_nav", q_em,
           "🎯 Goal: Launch Gmail\n📋 Plan: 1. Open Gmail application\n⚡ Immediate Action: Opening Gmail.",
           action='OPEN_APP("Gmail")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "Gmail"',
           history=None)

add_sample("multistep_nav", q_em,
           "🎯 Goal: Open Alex's email thread\n📋 Plan: Select Alex's email from inbox\n⚡ Immediate Action: Tapping Alex's email.",
           action='CLICK("Alex: Review quarterly draft")',
           screen='[SCREEN: com.google.android.gm]\n[1] TextView "Primary Inbox"\n[2] TextView "Alex: Review quarterly draft"\n[3] TextView "Billing Notice"',
           history='Step 1: OPEN_APP("Gmail")')

add_sample("multistep_nav", q_em,
           "🎯 Goal: Type approval reply\n📋 Plan: Tap reply and type 'Draft approved'\n⚡ Immediate Action: Typing reply text.",
           action='TYPE("Compose reply", "Draft approved")',
           screen='[SCREEN: com.google.android.gm]\n[1] TextView "Alex"\n[2] TextView "Please review the draft"\n[3] EditText "Compose reply" [Input]',
           history='Step 1: OPEN_APP("Gmail") -> Step 2: CLICK("Alex: Review quarterly draft")')

add_sample("multistep_nav", q_em,
           "🎯 Goal: Send email reply\n📋 Plan: Tap send icon\n⚡ Immediate Action: Tapping send button.",
           action='CLICK("Send")',
           screen='[SCREEN: com.google.android.gm]\n[1] TextView "Draft approved" [Typed]\n[2] Button "Send"',
           history='Step 1: OPEN_APP("Gmail") -> Step 2: CLICK("Alex") -> Step 3: TYPE("Compose reply", "Draft approved")')

# Additional diverse multi-step samples (Uber, Camera, Alarm check, Contacts, etc.)
add_sample("multistep_nav", "Request ride home on Uber",
           "🎯 Goal: Open Uber application\n📋 Plan: 1. Launch Uber\n⚡ Immediate Action: Opening Uber.",
           action='OPEN_APP("Uber")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "Uber"',
           history=None)

add_sample("multistep_nav", "Request ride home on Uber",
           "🎯 Goal: Set destination to Home\n📋 Plan: Tap Where to and select Home\n⚡ Immediate Action: Selecting saved Home location.",
           action='CLICK("Home")',
           screen='[SCREEN: com.ubercab]\n[1] EditText "Where to?"\n[2] Button "Home"\n[3] Button "Work"',
           history='Step 1: OPEN_APP("Uber")')

add_sample("multistep_nav", "Add new contact Sarah to phonebook",
           "🎯 Goal: Open Contacts app\n📋 Plan: Launch Contacts\n⚡ Immediate Action: Opening Contacts.",
           action='OPEN_APP("Contacts")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "Contacts"',
           history=None)

add_sample("multistep_nav", "Add new contact Sarah to phonebook",
           "🎯 Goal: Create new contact entry\n📋 Plan: Tap add contact icon\n⚡ Immediate Action: Tapping plus button.",
           action='CLICK("Create Contact")',
           screen='[SCREEN: com.samsung.android.app.contacts]\n[1] TextView "Contacts"\n[2] Button "Create Contact"',
           history='Step 1: OPEN_APP("Contacts")')

add_sample("multistep_nav", "Take a photo with front camera",
           "🎯 Goal: Open camera\n📋 Plan: 1. Launch Camera app\n⚡ Immediate Action: Opening Camera.",
           action='OPEN_APP("Camera")',
           screen='[SCREEN: com.sec.android.app.launcher]\n[1] Button "Camera"',
           history=None)

add_sample("multistep_nav", "Take a photo with front camera",
           "🎯 Goal: Switch to front facing lens\n📋 Plan: Tap switch camera icon\n⚡ Immediate Action: Tapping flip camera button.",
           action='CLICK("Flip Camera")',
           screen='[SCREEN: com.sec.android.app.camera]\n[1] Button "Shutter"\n[2] Button "Flip Camera"',
           history='Step 1: OPEN_APP("Camera")')

add_sample("multistep_nav", "Take a photo with front camera",
           "🎯 Goal: Capture portrait\n📋 Plan: Tap shutter button\n⚡ Immediate Action: Tapping shutter to take photo.",
           action='CLICK("Shutter")',
           screen='[SCREEN: com.sec.android.app.camera]\n[1] Button "Shutter"\n[2] ImageView "Front Lens Active"',
           history='Step 1: OPEN_APP("Camera") -> Step 2: CLICK("Flip Camera")')

# =========================================================================
# 3. CONVERSATIONAL ADVICE & GUIDANCE (60 STRUCTURED SAMPLES)
# =========================================================================
advice_data = [
    ("Give me 3 valuable pieces of life and career advice",
     "Here are 3 foundational strategies for life and career:\n"
     "1. Master High-Leverage Fundamentals: Deeply understand problem decomposition, system architecture, and clear communication—these skills compound across every industry.\n"
     "2. Prioritize Rapid Iteration Over Passive Overthinking: Real-world user feedback on a working prototype teaches you more in days than theoretical deliberation does in months.\n"
     "3. Build Sustainable Daily Consistency: True excellence is the byproduct of disciplined, repeatable habits rather than erratic bursts of burnout."),

    ("What are 3 effective strategies for managing stress?",
     "Here are 3 proven approaches for managing acute and chronic stress:\n"
     "1. Practice Physiological Sighs and Controlled Respiration: Two quick nasal inhales followed by an extended exhalation rapidly downregulates sympathetic autonomic arousal.\n"
     "2. Establish Clear Work-Rest Cognitive Boundaries: Deliberately disconnect from email and messaging apps after designated hours to permit neural restoration.\n"
     "3. Engage in Daily Aerobic Movement and Sunlight Exposure: Physical exertion metabolizes circulating cortisol while morning photons stabilize circadian mood regulation."),

    ("Suggest 3 habits for improving coding productivity",
     "Here are 3 high-impact habits to boost software engineering productivity:\n"
     "1. Design Interfaces Before Writing Implementation: Document the data contracts, types, and invariants before generating algorithmic logic to eliminate downstream refactoring.\n"
     "2. Eliminate Context Switching with Timeboxed Deep Work: Schedule 90-minute distraction-free blocks with notifications silenced for core architecture development.\n"
     "3. Automate Repetitive Tooling and Linting: Offload formatting, syntax validation, and test executions to local hotkeys and git hooks."),

    ("What are 3 rules for designing clean software architecture?",
     "Here are 3 essential rules for sustainable software architecture:\n"
     "1. Enforce Strict Separation of Concerns: Decouple domain logic from UI rendering, network transport, and persistence layers via explicit abstractions.\n"
     "2. Minimize Hidden Global State and Side Effects: Prefer pure functions and immutable data models to ensure thread safety and test predictability.\n"
     "3. Keep Dependencies Explicit and Injectable: Pass dependencies down through constructor parameters rather than relying on global singletons."),

    ("Give me 3 guidelines for effective technical leadership",
     "Here are 3 fundamental guidelines for leading engineering teams:\n"
     "1. Provide Context and Intent Over Micro-Management: Clarify the customer problem and architectural constraints, then empower engineers to design the solution.\n"
     "2. Foster a High-Trust, Blameless Post-Mortem Culture: Treat production outages as learning opportunities to harden systems rather than occasions to assign fault.\n"
     "3. Protect Team Bandwidth from Low-Value Distractions: Filter out unnecessary meetings and administrative overhead so engineers remain in deep flow state."),

    ("What are 3 core principles for deploying LLMs on mobile edge devices?",
     "Here are 3 core principles for on-device edge intelligence:\n"
     "1. Optimize Weight Quantization and Memory Footprint: Utilize FP16 or 4-bit block quantization to fit model parameters within phone RAM without paging.\n"
     "2. Leverage OS-Native Accessibility Over Computer Vision: Parsing the accessibility tree provides instant, zero-vision perception with 100% token accuracy.\n"
     "3. Implement Selective Repetition Penalties During Decoding: Prevent autoregressive token loops on numbers and lists by exempting newlines and whitespace."),

    ("Suggest 3 rules for maintaining long-term physical health",
     "Here are 3 foundational rules for physical health:\n"
     "1. Prioritize 7 to 8 Hours of Consistent Sleep: Sleep regulates hormonal balance, cognitive function, and cellular repair.\n"
     "2. Incorporate Daily Resistance Training and Walking: Combine compound strength movements with 8,000 daily steps for metabolic longevity.\n"
     "3. Base Your Diet on Whole Unprocessed Foods: Emphasize lean proteins, fibrous vegetables, and adequate hydration over refined sugars."),

    ("What are 3 principles for mastering public speaking?",
     "Here are 3 core principles for delivering compelling presentations:\n"
     "1. Anchor Every Talk Around One Core Narrative Message: Audiences remember a single clear takeaway rather than a barrage of disconnected facts.\n"
     "2. Embrace Deliberate Silence and Vocal Pacing: Pause before and after major points to let insights land rather than filling silence with vocal crutches.\n"
     "3. Structure Arguments with Contrast: Contrast the current painful status quo with the transformative future state to create natural engagement."),

    ("Give me 3 strategies for mastering complex technical concepts quickly",
     "Here are 3 accelerated learning strategies for technical domains:\n"
     "1. Apply the Feynman Technique: Explain the concept simply in plain language as if teaching a beginner to expose hidden gaps in understanding.\n"
     "2. Build a Minimal End-to-End Implementation: Write a small, working prototype from scratch rather than passively reading documentation.\n"
     "3. Map Core Mental Models to First Principles: Deconstruct the technology to its underlying axioms—physics, memory access, or mathematical invariants."),

    ("Suggest 3 rules for sound personal financial management",
     "Here are 3 disciplined rules for long-term financial security:\n"
     "1. Automate Savings and Dollar-Cost Averaging First: Transfer a fixed percentage of income to broad-market index funds immediately on payday.\n"
     "2. Maintain a 6-Month Liquid Emergency Reserve: Hold cash reserves in a high-yield account to prevent liquidating investments during market downturns.\n"
     "3. Avoid Lifestyle Creep as Income Grows: Allocate career raises toward compounding productive assets rather than depreciating luxury liabilities.")
]

# Expand advice to 60 diverse samples across domains
domains = [
    ("time management", "Prioritize By Impact Matrix", "Protect Morning Golden Hours", "Conduct Weekly Reviews"),
    ("remote team collaboration", "Default to Asynchronous Written Documentation", "Over-Communicate Milestone Progress", "Schedule Intentional Social Syncs"),
    ("cybersecurity hygiene", "Use Hardware Security Keys and Password Managers", "Keep Software and Firmware Updated Immediately", "Verify Sender Addresses Before Clicking Links"),
    ("startup validation", "Talk to 20 Potential Customers Before Writing Code", "Charge Real Money for Your Earliest Prototype", "Measure Retention and Daily Active Usage"),
    ("deep work immersion", "Batch Email and Messaging into 2 Daily Windows", "Work in Environments with Minimal Visual Stimulus", "Set Concrete Daily Milestone Checklists"),
    ("negotiation and persuasion", "Listen Twice as Much as You Speak", "Uncover the Underlying Constraint Rather Than the Stated Position", "Anchor with Fair Objective Criteria"),
    ("critical thinking", "Actively Invert the Problem to Spot Hidden Failure Modes", "Test Hypotheses Against Disconfirming Evidence", "Differentiate Correlation from Causal Mechanisms"),
    ("decision making under uncertainty", "Make Two-Way Door Decisions Rapidly", "Calculate Expected Value and Worst-Case Downsides", "Define Reversible Trigger Points Before Committing"),
    ("writing clear documentation", "Place Practical Code Examples Before Conceptual Explanations", "Keep Paragraphs Under 3 Sentences with Bold Lead Ins", "Document Why Architecture Choices Were Made"),
    ("mobile app responsiveness", "Perform Heavy Compute on Background Worker Threads", "Cache Network Responses Locally in SQLite", "Maintain 60 FPS UI Rendering by Minimizing View Nesting"),
    ("software testing strategies", "Write Isolated Fast-Executing Unit Tests First", "Use Integration Tests for External API Contracts", "Automate Regression Suites in CI Pipelines"),
    ("machine learning system design", "Version Both Code and Datasets Together", "Monitor Model Output Drift in Production", "Validate Latency on Target Edge Hardware"),
    ("database query optimization", "Index High-Cardinality WHERE Clause Columns", "Avoid SELECT Asterisk to Minimize I/O Overhead", "Batch High-Volume Inserts Inside Transactions"),
    ("caching architectures", "Establish Explicit Cache Expiration TTLs", "Handle Cache Misses with Graceful Fallbacks", "Invalidate Stale Keys Immediately on Write"),
    ("code review best practices", "Keep Pull Requests Under 250 Lines of Code", "Review Architectural Design Before Minor Syntax", "Explain the Reasoning Behind Every Requested Change"),
    ("API design standards", "Use Predictable RESTful Resource Naming", "Provide Explicit Versioning in URL Paths", "Return Descriptive HTTP Error Status Codes"),
    ("mobile battery optimization", "Batch Wake Locks and Network Requests Together", "Pause CPU-Intensive Background Tasks When Screen is Off", "Use Dark Mode Color Palettes on OLED Screens"),
    ("incident response protocols", "Establish a Single Incident Commander Immediately", "Maintain a Chronological Incident War Room Log", "Conduct a Blameless Post-Mortem Within 48 Hours"),
    ("continuous integration pipelines", "Keep Build and Test Execution Under 5 Minutes", "Fail Fast on Lint and Static Analysis Errors", "Run Parallelized Test Shards on Ephemeral Runners"),
    ("cybersecurity threat modeling", "Identify High-Value Data Assets First", "Map Trust Boundaries and External Ingress Points", "Apply Defense-in-Depth Across All Infrastructure Layers"),
    ("distributed system reliability", "Implement Exponential Backoff and Jitter on Retries", "Protect Downstream Services with Circuit Breakers", "Design Every Network Mutation to Be Idempotent"),
    ("developer onboarding efficiency", "Maintain a Single-Command Local Environment Setup", "Pair New Hires with Dedicated Technical Mentors", "Ship a Production Code Change on Day One"),
    ("open source project management", "Write Clear Issue and PR Contribution Templates", "Provide a Welcoming and Transparent Roadmap", "Automate Dependency Security Vulnerability Scans"),
    ("technical debt management", "Track Tech Debt as Explicit Backlog Tasks", "Refactor incrementally Alongside Feature Work", "Set Hard Limits on Deprecated Code Lifespans"),
    ("cloud infrastructure cost control", "Right-Size Over-Provisioned VM Compute Instances", "Use Ephemeral Spot Instances for Batch Jobs", "Set Real-Time Budget Anomaly Alerting"),
    ("data privacy and compliance", "Encrypt Sensitive PII Data at Rest and in Transit", "Implement Principle of Least Privilege Access Controls", "Enforce Automated Data Retention and Purging Policies"),
    ("focus and deep concentration", "Silence Non-Critical Smartphone Notifications", "Block Visual Distractions in Your Direct Field of View", "Take 5-Minute Brain Breaks Away from All Screens"),
    ("handling professional setbacks", "Separate Objective Facts from Emotional Reactions", "Extract the Core Lesson to Prevent Repeat Failures", "Focus Immediately on the Next High-Leverage Move"),
    ("effective cross-functional communication", "Avoid Domain Jargon When Speaking to Stakeholders", "Frame Engineering Tradeoffs in Terms of Business Impact", "Follow Up Verbal Agreements with Written Bullet Points"),
    ("mentoring junior engineers", "Ask Guiding Questions Rather Than Giving Direct Answers", "Encourage Ownership Over Complex Feature Slices", "Celebrate First Significant Production Milestones"),
    ("conducting effective meetings", "Always Distribute a Written Agenda Beforehand", "Assign Explicit Action Owners to Every Decision", "End 5 Minutes Early to Give People Transition Time"),
    ("maintaining mental clarity", "Take Daily Brisk Outdoor Walks in Daylight", "Write Morning Journal Pages to Clear Working Memory", "Limit Sugar and High-Glycemic Processed Snacks"),
    ("reading technical papers", "Read Abstract and Conclusion Before Methodology", "Analyze the Benchmark Graphs and Baseline Comparisons", "Re-Implement Core Equation in Scratch Python Code"),
    ("negotiating job compensation", "Research Market Ranges Across Industry Percentiles", "Focus on Value Contribution Rather Than Personal Need", "Get All Final Compensation Terms in Signed Writing"),
    ("giving constructive feedback", "Praise Meaningful Contributions Publicly", "Deliver Critical Suggestions Privately with Empathy", "Propose Actionable Steps to Address Weak Points"),
    ("building high-performance habits", "Anchor New Habits to Existing Daily Routines", "Track Consistency on a Visible Physical Calendar", "Focus on Showing Up Even on Low-Motivation Days"),
    ("designing accessible mobile apps", "Ensure All Touch Targets Exceed 48x48 Density Pixels", "Provide Descriptive Content Descriptions for Screen Readers", "Support Dynamic Type and High Contrast Display Modes")
]

for q, txt in advice_data:
    add_sample("advice", q, txt)

for dom, p1, p2, p3 in domains:
    q = f"Give me 3 best practices for {dom}"
    txt = (f"Here are 3 essential best practices for {dom}:\n"
           f"1. {p1}: Focus on high-leverage execution to drive foundational success.\n"
           f"2. {p2}: Eliminate friction points through systematic, disciplined processes.\n"
           f"3. {p3}: Measure long-term outcomes to continuously optimize performance.")
    add_sample("advice", q, txt)

# =========================================================================
# 4. IDENTITY & CORE CONTRACTS (25 SAMPLES)
# =========================================================================
identity_queries = [
    "Who are you?",
    "Who built you?",
    "What is your name and who created you?",
    "Who are you and who built you?",
    "What architecture do you run on?",
    "Tell me about yourself",
    "Are you ChatGPT or Claude?",
    "What model powers your executive capabilities?",
    "Where do you run?",
    "Are you an on-device agent?",
    "Who trained you?",
    "What is YORI?",
    "What organization designed you?",
    "Can you describe your system contract?",
    "What is Embeddra AI Research Lab?"
]
for q in identity_queries:
    add_sample("identity", q,
               "I am YORI, an on-device executive agent built on the Maria-1B architecture by Embeddra AI Research Lab. "
               "I operate completely locally on your hardware, providing conversational intelligence, document analysis, "
               "and autonomous Android device control without sending your personal data to remote cloud servers.")

# =========================================================================
# 5. DOCUMENT Q&A (20 SAMPLES)
# =========================================================================
doc_queries = [
    ("Summarize the attached contract agreement", "contract.pdf",
     "The attached agreement specifies a 12-month software licensing term between Embeddra and Client. "
     "Key provisions include 99.9% uptime SLA, monthly billing of $5,000, and mutual non-disclosure obligations."),
    ("What are the main risks mentioned in the financial report?", "financials.txt",
     "The report highlights three primary financial risks: foreign currency exchange fluctuations, rising cloud compute costs, "
     "and potential delays in hardware supply chain deliveries."),
    ("Extract the meeting action items from the notes", "meeting_notes.md",
     "The three key action items are:\n1. Alex to finalize the mobile APK release by Friday.\n2. Sarah to complete load testing on the Maria-1B inference engine.\n3. Team to review quantization metrics on Monday."),
    ("What is the return policy outlined in this receipt?", "policy.txt",
     "Items can be returned within 30 days of purchase with original packaging for a full refund. Digital software licenses are non-refundable once activated."),
    ("What are the system requirements for installing this software?", "requirements.txt",
     "The software requires a 64-bit OS (Android 14+ or Windows 11), a minimum of 6GB of system RAM, and 3GB of free disk storage for model weights."),
    ("Summarize the patient discharge instructions", "discharge_summary.pdf",
     "The patient is discharged with instructions to rest for 48 hours, take prescribed antibiotics twice daily with food, and schedule a follow-up appointment in 10 days."),
    ("What were the third quarter revenue highlights?", "q3_report.md",
     "Third quarter revenue grew 24% year-over-year to $4.2M, driven primarily by strong adoption of enterprise edge AI licenses and SDK deployments."),
    ("What does the license say about commercial use?", "license.txt",
     "Commercial use is permitted under the commercial tier. Redistribution of raw model weights without explicit license terms is prohibited."),
    ("List the ingredients and allergen warnings in this recipe", "recipe.json",
     "Ingredients include flour, sugar, whole milk, and crushed walnuts. Allergen warning: Contains dairy, gluten, and tree nuts."),
    ("What are the security guidelines for handling customer API keys?", "security_handbook.md",
     "API keys must never be committed to git repositories, must be stored in encrypted secret managers, and must be rotated every 90 days.")
]
for q, fname, ans in doc_queries:
    add_sample("doc_qa", f"<document name=\"{fname}\">\nContent of {fname}\n</document>\n\n{q}", ans)

# =========================================================================
# DEDUPLICATION & VALIDATION
# =========================================================================
seen = set()
unique_samples = []
for s in samples:
    h = hashlib.md5(s["full_text"].encode("utf-8")).hexdigest()
    if h not in seen:
        seen.add(h)
        unique_samples.append(s)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    for s in unique_samples:
        f.write(json.dumps(s, ensure_ascii=False) + "\n")

print(f"\n[OK] Successfully wrote {len(unique_samples)} 100% UNIQUE samples to {OUT_PATH}", flush=True)

# Breakdown summary
cat_counts = Counter(s["category"] for s in unique_samples)
print("Category Breakdown:")
for cat, count in cat_counts.most_common():
    print(f"  * {cat:<20}: {count:3d} ({count/len(unique_samples)*100:.1f}%)")

tool_counts = Counter()
for s in unique_samples:
    if "<action>" in s["target"]:
        act = s["target"].split("<action>")[1].split("</action>")[0].split("(")[0].strip()
        tool_counts[act] += 1

print("\nTool Action Breakdown across all Direct Control & Multi-Step:")
for tool, count in tool_counts.most_common():
    print(f"  * {tool:<20}: {count:3d}")
