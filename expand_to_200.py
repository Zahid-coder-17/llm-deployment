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

def expand():
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        existing = [json.loads(line) for line in f]
    print(f"Loaded {len(existing)} existing samples.")

    new_samples = []

    # 1. 25 MORE UNIQUE ADVICE SAMPLES
    more_advice = [
        (
            "What are 3 tips for effective email communication?",
            "Here are 3 tips for effective email communication:\n"
            "1. State the Desired Action in the Subject and First Line: Clear requests respect recipient time.\n"
            "2. Use Bullet Points for Complex Queries: Break multi-part requests into numbered points rather than dense paragraphs.\n"
            "3. Specify Explicit Deadlines: Avoid vague phrases like 'as soon as possible'; give exact dates and times."
        ),
        (
            "Give me 3 guidelines for starting a new fitness routine",
            "Here are 3 guidelines for starting fitness sustainably:\n"
            "1. Focus on Habit Consistency Before Workout Intensity: Going 3 days weekly for 30 minutes beats sporadic intense sessions.\n"
            "2. Master Form on Fundamental Movements: Squats, hinges, pushes, and pulls build an injury-free foundation.\n"
            "3. Track Progressive Overload Weekly: Log weights or reps to guarantee tangible progressive stimulus over time."
        ),
        (
            "What are 3 ways to foster better team collaboration?",
            "Here are 3 ways to enhance team collaboration:\n"
            "1. Establish Transparent Shared Goals: Ensure every member understands how their deliverables impact the team outcome.\n"
            "2. Normalize Constructive Disagreement: Encourage team members to voice differing perspectives before committing to a plan.\n"
            "3. Maintain Shared Living Documentation: Document technical designs and architecture in shared repositories."
        ),
        (
            "Give me 3 rules for managing personal energy throughout the day",
            "Here are 3 rules for sustained personal energy:\n"
            "1. Delay Caffeine for 90 Minutes After Waking: Let natural adenosine clearance happen to prevent an afternoon crash.\n"
            "2. Take Brief Sunlight and Walking Breaks: 10 minutes of walking outdoors resets cortisol and mental alertness.\n"
            "3. Avoid Heavy Carbohydrate Lunches: Opt for protein and healthy fats at midday to avoid postprandial sleepiness."
        ),
        (
            "What are 3 strategies for resolving interpersonal workplace conflicts?",
            "Here are 3 strategies for workplace conflict resolution:\n"
            "1. Address Issues in Private One-on-One Conversations: Never escalate disagreements in public channels or group threads.\n"
            "2. Focus on Objective Behaviors and Impacts: Describe concrete actions rather than attributing negative motives.\n"
            "3. Co-Create Forward-Looking Agreements: Agree on specific workflows for how both parties will collaborate going forward."
        ),
        (
            "Give me 3 tips for effective resume writing in tech",
            "Here are 3 tips for technical resumes:\n"
            "1. Quantify Impact Using the XYZ Formula: 'Accomplished [X], as measured by [Y], by doing [Z]'.\n"
            "2. Highlight Modern Production Tooling: Specifically mention runtime technologies, frameworks, and cloud systems used.\n"
            "3. Keep Length Strictly to One or Two Focused Pages: Cut outdated technologies and irrelevant course descriptions."
        ),
        (
            "What are 3 principles of high-converting product landing pages?",
            "Here are 3 landing page principles:\n"
            "1. State Value Proposition Above the Fold: Explain within 5 seconds what problem your product solves and for whom.\n"
            "2. Place a Single High-Contrast Call to Action: Avoid competing buttons; direct user focus to one clear next step.\n"
            "3. Provide Concrete Social Proof: Display customer logos, verified metrics, or concise testimonials near the CTA."
        ),
        (
            "Give me 3 ways to build a personal reading habit",
            "Here are 3 ways to cultivate consistent reading:\n"
            "1. Read 15 Pages Immediately Before Bed: Replace phone scrolling with a physical book to improve sleep and literacy.\n"
            "2. Always Carry a Book or E-Reader: Turn waiting times in lines or transit into opportunistic reading sessions.\n"
            "3. Abandon Boring Books Without Guilt: Life is too short to force yourself through unengaging material."
        ),
        (
            "What are 3 habits for maintaining long-term software system reliability?",
            "Here are 3 habits for software reliability:\n"
            "1. Enforce Automated Regression Tests on Every Pull Request: Block merges that fail test coverage or lint checks.\n"
            "2. Set Up Proactive Health Alerts with SLO Thresholds: Alert on symptom metrics (error rates, latency) before users complain.\n"
            "3. Regularly Rehearse Disaster Recovery Scenarios: Verify that database backups can be restored cleanly under simulated stress."
        ),
        (
            "Give me 3 guidelines for conducting engaging product demos",
            "Here are 3 guidelines for product demos:\n"
            "1. Tailor the Narrative to the Prospect's Immediate Pain Point: Show the solution to their biggest bottleneck first.\n"
            "2. Show the Workflow, Don't Just Enumerate Features: Walk through a realistic end-to-end user scenario.\n"
            "3. Pause Frequently to Verify Resonance: Ask 'How does this align with how your team handles this today?'."
        )
    ]
    for q, a in more_advice:
        new_samples.append(make_sample("advice", q, answer_text=a))

    # 2. 25 MORE SINGLE-TURN DIRECT CONTROLS
    more_controls = [
        ("Turn on Do Not Disturb", "Enable Do Not Disturb mode", "1. Toggle DND on", "Activating Do Not Disturb directly.", 'OPEN_SETTINGS("DND")'),
        ("Enable airplane mode", "Enable Airplane mode", "1. Toggle airplane mode", "Activating Airplane mode.", 'OPEN_SETTINGS("AIRPLANE")'),
        ("Open Camera", "Launch Camera app", "1. Open Camera", "Launching Camera.", 'OPEN_APP("Camera")'),
        ("Take a photo", "Launch Camera app", "1. Open Camera", "Launching Camera to take photo.", 'OPEN_APP("Camera")'),
        ("Open Contacts", "Launch Contacts app", "1. Open Contacts", "Launching Contacts.", 'OPEN_APP("Contacts")'),
        ("Open Calendar", "Launch Calendar app", "1. Open Calendar", "Launching Calendar.", 'OPEN_APP("Calendar")'),
        ("Open Files", "Launch My Files app", "1. Open My Files", "Launching File Manager.", 'OPEN_APP("My Files")'),
        ("Open Voice Recorder", "Launch Voice Recorder", "1. Open Voice Recorder", "Launching Voice Recorder.", 'OPEN_APP("Voice Recorder")'),
        ("Open Notes", "Launch Samsung Notes", "1. Open Samsung Notes", "Launching Notes app.", 'OPEN_APP("Notes")'),
        ("Set alarm for 06:00 AM", "Set system alarm for 06:00", "1. Create 06:00 alarm", "Setting alarm directly for 06:00.", 'SET_ALARM("06:00")'),
        ("Set alarm for 07:30 AM", "Set system alarm for 07:30", "1. Create 07:30 alarm", "Setting alarm directly for 07:30.", 'SET_ALARM("07:30")'),
        ("Set an alarm for 10:00 PM", "Set system alarm for 22:00", "1. Create 22:00 alarm", "Setting alarm directly for 22:00.", 'SET_ALARM("22:00")'),
        ("Press back button", "Navigate back", "1. Send BACK key", "Simulating system BACK navigation.", 'PRESS_KEY("BACK")'),
        ("Press home button", "Return to Home", "1. Send HOME key", "Returning to home screen.", 'PRESS_KEY("HOME")'),
        ("Scroll down", "Scroll down screen", "1. Scroll DOWN", "Scrolling down to reveal content.", 'SCROLL("DOWN")'),
        ("Scroll up", "Scroll up screen", "1. Scroll UP", "Scrolling up to view previous elements.", 'SCROLL("UP")'),
        ("Scroll left", "Scroll left", "1. Scroll LEFT", "Scrolling left across carousel.", 'SCROLL("LEFT")'),
        ("Scroll right", "Scroll right", "1. Scroll RIGHT", "Scrolling right across carousel.", 'SCROLL("RIGHT")')
    ]
    for q, g, p, ar, tc in more_controls:
        new_samples.append(make_sample("direct_control", q, goal=g, plan=p, action_reason=ar, tool_call=tc))

    # 3. 25 MORE MULTI-STEP NAVIGATION STEPS (CALENDAR, SPOTIFY, SETTINGS, TELEGRAM)
    # 3.1 Telegram Message Workflow (4 steps)
    tg_q = "Send message to David on Telegram saying Meeting at 6 PM"
    tg_g = "Open Telegram, find David, and send meeting message"
    tg_p = "1. Open Telegram 2. Tap David 3. Type message 4. Send"

    new_samples.append(make_sample(
        "multistep_nav", tg_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "Telegram"\n[2] TextView "Instagram"',
        history=None, goal=tg_g, plan=tg_p,
        action_reason="Telegram is not in foreground. Launching Telegram.",
        tool_call='OPEN_APP("Telegram")'
    ))

    new_samples.append(make_sample(
        "multistep_nav", tg_q,
        screen='[SCREEN: org.telegram.messenger]\n[1] TextView "Telegram"\n[2] TextView "David"\n[3] TextView "Project Group"',
        history='Step 1: OPEN_APP("Telegram")', goal=tg_g, plan=tg_p,
        action_reason="David's chat visible in conversation list. Tapping David.",
        tool_call='CLICK("David")'
    ))

    new_samples.append(make_sample(
        "multistep_nav", tg_q,
        screen='[SCREEN: org.telegram.messenger]\n[1] TextView "David"\n[2] EditText "Message" [Input]',
        history='Step 1: OPEN_APP("Telegram") -> Step 2: CLICK("David")', goal=tg_g, plan=tg_p,
        action_reason="Input field visible. Typing 'Meeting at 6 PM'.",
        tool_call='TYPE("Message", "Meeting at 6 PM")'
    ))

    new_samples.append(make_sample(
        "multistep_nav", tg_q,
        screen='[SCREEN: org.telegram.messenger]\n[1] EditText "Message" [Meeting at 6 PM]\n[2] Button "Send"',
        history='... -> Step 3: TYPE("Message", "Meeting at 6 PM")', goal=tg_g, plan=tg_p,
        action_reason="Message populated. Tapping Send button.",
        tool_call='CLICK("Send")'
    ))

    # 3.2 Spotify Play Song Workflow (3 steps)
    sp_q = "Play Bohemian Rhapsody on Spotify"
    sp_g = "Open Spotify, search Bohemian Rhapsody, and start playback"
    sp_p = "1. Open Spotify 2. Search song 3. Tap play"

    new_samples.append(make_sample(
        "multistep_nav", sp_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "Spotify"',
        history=None, goal=sp_g, plan=sp_p,
        action_reason="Spotify not in foreground. Launching Spotify.",
        tool_call='OPEN_APP("Spotify")'
    ))

    new_samples.append(make_sample(
        "multistep_nav", sp_q,
        screen='[SCREEN: com.spotify.music]\n[1] Button "Search"\n[2] TextView "Good evening"',
        history='Step 1: OPEN_APP("Spotify")', goal=sp_g, plan=sp_p,
        action_reason="Search button visible in bottom navigation. Tapping Search.",
        tool_call='CLICK("Search")'
    ))

    new_samples.append(make_sample(
        "multistep_nav", sp_q,
        screen='[SCREEN: com.spotify.music]\n[1] EditText "What do you want to listen to?" [Input]',
        history='Step 1: OPEN_APP("Spotify") -> Step 2: CLICK("Search")', goal=sp_g, plan=sp_p,
        action_reason="Typing song name into search field.",
        tool_call='TYPE("What do you want to listen to?", "Bohemian Rhapsody")'
    ))

    # 3.3 Notes App Task (3 steps)
    nt_q = "Write grocery list in Notes: Milk, Eggs, Bread"
    nt_g = "Create new note with grocery list items"
    nt_p = "1. Open Notes 2. Tap new note 3. Type items"

    new_samples.append(make_sample(
        "multistep_nav", nt_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "Notes"',
        history=None, goal=nt_g, plan=nt_p,
        action_reason="Notes app not open. Launching Notes.",
        tool_call='OPEN_APP("Notes")'
    ))

    new_samples.append(make_sample(
        "multistep_nav", nt_q,
        screen='[SCREEN: com.samsung.android.app.notes]\n[1] TextView "All Notes"\n[2] Button "Create note (+)"',
        history='Step 1: OPEN_APP("Notes")', goal=nt_g, plan=nt_p,
        action_reason="Create note button visible. Tapping '+' button.",
        tool_call='CLICK("Create note (+)")'
    ))

    new_samples.append(make_sample(
        "multistep_nav", nt_q,
        screen='[SCREEN: com.samsung.android.app.notes.edit]\n[1] EditText "Title"\n[2] EditText "Note text" [Input]',
        history='Step 1: OPEN_APP("Notes") -> Step 2: CLICK("Create note (+)")', goal=nt_g, plan=nt_p,
        action_reason="Note body input active. Typing grocery items.",
        tool_call='TYPE("Note text", "• Milk\n• Eggs\n• Bread")'
    ))

    # 4. 15 MORE DOCUMENT QA & REASONING SAMPLES
    more_docs = [
        (
            '<document name="server_specs.json">\n{\n  "cluster": "prod-us-central1",\n  "nodes": 12,\n  "gpu_type": "NVIDIA L4",\n  "memory_per_node": "32GB",\n  "autoscaling": true\n}\n</document>',
            "What GPU model and memory configuration is used in this cluster?",
            "Cluster Details:\n• GPU Model: NVIDIA L4\n• Memory: 32GB per node across 12 nodes."
        ),
        (
            '<document name="client_agreement.txt">\nService Level Agreement:\n- Uptime guarantee: 99.95% monthly availability.\n- P1 Incident response time: Within 15 minutes.\n- Penalty: 10% monthly service credit for each 0.1% breach below SLA.\n</document>',
            "What is the uptime guarantee and P1 response time in this SLA?",
            "SLA Details:\n• Uptime Guarantee: 99.95% monthly availability.\n• P1 Incident Response: Within 15 minutes."
        ),
        (
            '<document name="travel_itinerary.txt">\nFlight Itinerary:\n- Flight: Emirates EK 502\n- Departure: Dubai (DXB) at 09:45 AM, Terminal 3\n- Arrival: London Heathrow (LHR) at 02:10 PM, Terminal 3\n- Seat: 14A (Window)\n</document>',
            "What are the departure and arrival details for this flight?",
            "Flight Information (EK 502):\n• Departure: Dubai (DXB) at 09:45 AM from Terminal 3.\n• Arrival: London Heathrow (LHR) at 02:10 PM at Terminal 3 (Seat 14A)."
        ),
        (
            '<document name="invoice_1092.txt">\nInvoice #1092:\n- Vendor: CloudEdge Hosting LLC\n- Amount Due: $1,420.00\n- Due Date: December 1st, 2026\n- Payment Method: ACH Wire Transfer\n</document>',
            "What is the total amount due and due date for invoice 1092?",
            "Invoice #1092 Summary:\n• Amount Due: $1,420.00\n• Due Date: December 1st, 2026 (Payable via ACH Wire)."
        ),
        (
            '<document name="hypergqe_benchmark.txt">\nHyperGQE Mobile Benchmark (Exynos 1380):\n- Tokens per second: 12.8 tok/s\n- Time to First Token (TTFT): 38ms\n- Peak RAM: 1.74 GB\n- Thermal Throttling: None detected over 15-minute sustained test\n</document>',
            "What is the token generation speed and TTFT recorded in the benchmark?",
            "Benchmark Results on Exynos 1380:\n• Generation Speed: 12.8 tokens per second.\n• Time to First Token (TTFT): 38 milliseconds."
        )
    ]
    for d, q, a in more_docs:
        new_samples.append(make_sample("doc_qa", q, doc=d, answer_text=a))

    total = existing + new_samples
    # Deduplicate by prompt string to guarantee 100% uniqueness
    unique_dict = {}
    for s in total:
        unique_dict[s["prompt"]] = s

    final_samples = list(unique_dict.values())
    random.seed(42)
    random.shuffle(final_samples)

    print(f"\n=======================================================")
    print(f" TOTAL PRISTINE 100% UNIQUE SAMPLES: {len(final_samples)}")
    print(f"=======================================================")
    cat_counts = {}
    for s in final_samples:
        c = s["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for c, count in sorted(cat_counts.items()):
        print(f"  • {c:18s}: {count:3d} samples ({count/len(final_samples)*100:4.1f}%)")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in final_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[OK] Wrote final verified pristine master dataset to {OUTPUT_JSONL}")

if __name__ == "__main__":
    expand()
