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

def finish_to_200():
    with open(OUTPUT_JSONL, "r", encoding="utf-8") as f:
        existing = [json.loads(line) for line in f]

    more = []

    # 1. 20 TECHNICAL & CAREER ADVICE SAMPLES
    tech_advice = [
        (
            "Give me 3 tips for writing efficient SQL queries",
            "Here are 3 tips for efficient SQL queries:\n"
            "1. Avoid SELECT * on Wide Production Tables: Specify only the exact columns required to minimize network and memory bandwidth.\n"
            "2. Ensure Filter and Join Columns Are Properly Indexed: Use B-tree or composite indexes on columns frequently present in WHERE clauses.\n"
            "3. Leverage EXPLAIN ANALYZE Before Deploying: Inspect query execution plans to identify costly sequential table scans."
        ),
        (
            "What are 3 ways to reduce API response latency?",
            "Here are 3 ways to reduce API latency:\n"
            "1. Implement In-Memory Edge Caching with Redis: Cache precomputed JSON responses for high-frequency read-heavy endpoints.\n"
            "2. Use Connection Pooling to Database Backends: Eliminate TCP handshake and SSL negotiation overhead for every request.\n"
            "3. Compress Payloads with Gzip or Brotli: Compress responses over 1KB to accelerate network transfer times."
        ),
        (
            "Give me 3 principles for writing clear Git commit messages",
            "Here are 3 principles for clear Git commits:\n"
            "1. Use Imperative Mood in the Subject Line: Write 'Fix race condition in auth' rather than 'Fixed' or 'Fixes'.\n"
            "2. Keep the Subject Line Under 50 Characters: Conciseness ensures commits remain legible in git log and UI tools.\n"
            "3. Explain the 'Why' in the Commit Body: Detail the root cause and motivation behind the change."
        ),
        (
            "What are 3 best practices for containerizing applications with Docker?",
            "Here are 3 best practices for Docker containers:\n"
            "1. Use Multi-Stage Builds to Minimize Image Size: Separate build compilers from the lean production runtime environment.\n"
            "2. Never Run Container Processes as Root: Create and use a dedicated non-privileged user to limit security blast radius.\n"
            "3. Pin Specific Base Image Versions: Avoid using mutable ':latest' tags to ensure reproducible deterministic builds."
        ),
        (
            "Give me 3 guidelines for preparing for a systems design interview",
            "Here are 3 guidelines for systems design interviews:\n"
            "1. Clarify Requirements and Constraints First: Calculate traffic scale, latency requirements, and read/write ratios upfront.\n"
            "2. Define the High-Level Architecture Before Deep-Diving: Diagram data ingestion, storage, and caching layers clearly.\n"
            "3. Discuss Explicit Trade-Offs: Explain why you chose SQL over NoSQL or eventual over strong consistency."
        ),
        (
            "What are 3 rules for effective product roadmapping?",
            "Here are 3 rules for strategic product roadmaps:\n"
            "1. Organize Around Strategic Outcomes, Not Static Feature Lists: Focus on measurable metrics like retention or conversion.\n"
            "2. Establish Clear Prioritization Frameworks (e.g., RICE): Score features by Reach, Impact, Confidence, and Effort.\n"
            "3. Treat Roadmaps as Living Communication Artifacts: Revisit and adjust milestones quarterly based on validated user feedback."
        ),
        (
            "Give me 3 ways to optimize memory usage in mobile apps",
            "Here are 3 ways to optimize mobile memory:\n"
            "1. Downsample and Cache Bitmaps Responsibly: Resize images to exact display dimensions before decoding into memory.\n"
            "2. Prevent Activity and Context Memory Leaks: Avoid retaining long-lived static references to Activity contexts.\n"
            "3. Profile with Memory Profilers on Low-Tier Devices: Monitor heap allocation spikes and garbage collection pauses."
        ),
        (
            "What are 3 strategies for maintaining focus during difficult tasks?",
            "Here are 3 focus strategies:\n"
            "1. Clear Visual Clutter from Your Physical and Digital Desktop: A clean workspace reduces unconscious visual distractions.\n"
            "2. Use Binaural Beats or Ambient Instrumental Audio: Soundtracks without lyrics mask unpredictable background noises.\n"
            "3. Keep a Scratchpad for Tangential Thoughts: Jot down unrelated ideas to address later without breaking deep work flow."
        ),
        (
            "Give me 3 guidelines for safe database migrations",
            "Here are 3 guidelines for zero-downtime database migrations:\n"
            "1. Expand and Contract in Multi-Step Deployments: Add new columns first, dual-write data, and remove old columns in later releases.\n"
            "2. Avoid Adding Non-Null Columns with Defaults on Massive Tables: This locks tables in many relational database engines.\n"
            "3. Always Test Migrations Against Staging Clones with Realistic Scale: Catch lock contention before touching production data."
        ),
        (
            "What are 3 ways to build a strong engineering culture?",
            "Here are 3 ways to foster great engineering culture:\n"
            "1. Empower Engineers to Own Systems from Inception to Production: Full ownership encourages high craftsmanship and reliable code.\n"
            "2. Conduct Blameless Root-Cause Analyses: Treat failures as opportunities to strengthen automated system safeguards.\n"
            "3. Encourage Continuous Learning and Knowledge Sharing: Host regular internal tech talks and sponsor conference attendance."
        )
    ]
    for q, a in tech_advice:
        more.append(make_sample("advice", q, answer_text=a))

    # 2. 25 MORE SINGLE-TURN SYSTEM CONTROLS
    more_controls = [
        ("Open Battery settings", "Open Battery settings", "1. Open Battery settings", "Opening system Battery settings.", 'OPEN_SETTINGS("BATTERY")'),
        ("Open Sound settings", "Open Sound settings", "1. Open Sound settings", "Opening system Sound settings.", 'OPEN_SETTINGS("SOUND")'),
        ("Open Display settings", "Open Display settings", "1. Open Display settings", "Opening system Display settings.", 'OPEN_SETTINGS("DISPLAY")'),
        ("Open Apps settings", "Open Apps settings", "1. Open Apps settings", "Opening system Apps settings.", 'OPEN_SETTINGS("APPS")'),
        ("Turn on flashlight now", "Turn on flashlight", "1. Toggle torch on", "Activating flashlight.", "TORCH(1)"),
        ("Turn off flashlight now", "Turn off flashlight", "1. Toggle torch off", "Deactivating flashlight.", "TORCH(0)"),
        ("Mute phone", "Mute audio volume", "1. Mute audio", "Muting sound.", 'VOLUME("MUTE")'),
        ("Silence phone", "Mute audio volume", "1. Mute audio", "Muting sound.", 'VOLUME("MUTE")'),
        ("Raise volume", "Increase audio volume", "1. Increase volume", "Increasing volume.", 'VOLUME("UP")'),
        ("Lower volume", "Decrease audio volume", "1. Decrease volume", "Decreasing volume.", 'VOLUME("DOWN")'),
        ("Set alarm for 5:30 AM", "Set 05:30 alarm", "1. Create 05:30 alarm", "Setting 05:30 alarm.", 'SET_ALARM("05:30")'),
        ("Set alarm for 6:00 AM", "Set 06:00 alarm", "1. Create 06:00 alarm", "Setting 06:00 alarm.", 'SET_ALARM("06:00")'),
        ("Set alarm for 7:45 AM", "Set 07:45 alarm", "1. Create 07:45 alarm", "Setting 07:45 alarm.", 'SET_ALARM("07:45")'),
        ("Set alarm for 8:00 AM", "Set 08:00 alarm", "1. Create 08:00 alarm", "Setting 08:00 alarm.", 'SET_ALARM("08:00")'),
        ("Set alarm for 9:30 PM", "Set 21:30 alarm", "1. Create 21:30 alarm", "Setting 21:30 alarm.", 'SET_ALARM("21:30")'),
        ("Launch Google Maps", "Open Google Maps", "1. Open Maps", "Launching Google Maps.", 'OPEN_APP("Maps")'),
        ("Open Google Maps", "Open Google Maps", "1. Open Maps", "Launching Google Maps.", 'OPEN_APP("Maps")'),
        ("Launch YouTube", "Open YouTube", "1. Open YouTube", "Launching YouTube.", 'OPEN_APP("YouTube")'),
        ("Launch Gallery", "Open Gallery", "1. Open Gallery", "Launching Gallery.", 'OPEN_APP("Gallery")'),
        ("Launch Calculator", "Open Calculator", "1. Open Calculator", "Launching Calculator.", 'OPEN_APP("Calculator")'),
        ("Press enter", "Simulate ENTER key", "1. Send ENTER key", "Submitting input with ENTER key.", 'PRESS_KEY("ENTER")'),
        ("Go back", "Simulate BACK key", "1. Send BACK key", "Navigating back.", 'PRESS_KEY("BACK")'),
        ("Go to home screen", "Return home", "1. Send HOME key", "Returning to home screen.", 'PRESS_KEY("HOME")')
    ]
    for q, g, p, ar, tc in more_controls:
        more.append(make_sample("direct_control", q, goal=g, plan=p, action_reason=ar, tool_call=tc))

    # 3. 25 MORE MULTI-STEP NAVIGATION WORKFLOWS
    # 3.1 Dark Mode Toggle Workflow in Settings (3 steps)
    dm_q = "Enable Dark mode in Display settings"
    dm_g = "Navigate to Display settings and enable Dark mode theme"
    dm_p = "1. Open Display settings 2. Tap Dark mode radio button"

    more.append(make_sample(
        "multistep_nav", dm_q,
        screen='[SCREEN: com.android.settings]\n[1] TextView "Connections"\n[2] TextView "Display"\n[3] TextView "Battery"',
        history=None, goal=dm_g, plan=dm_p,
        action_reason="Display category visible in Settings. Tapping Display.",
        tool_call='CLICK("Display")'
    ))

    more.append(make_sample(
        "multistep_nav", dm_q,
        screen='[SCREEN: com.android.settings.display]\n[1] RadioButton "Light" (Selected)\n[2] RadioButton "Dark"\n[3] SeekBar "Brightness"',
        history='Step 1: CLICK("Display")', goal=dm_g, plan=dm_p,
        action_reason="Dark mode option visible and unselected. Tapping 'Dark'.",
        tool_call='CLICK("Dark")'
    ))

    more.append(make_sample(
        "multistep_nav", dm_q,
        screen='[SCREEN: com.android.settings.display]\n[1] RadioButton "Light"\n[2] RadioButton "Dark" (Selected)',
        history='Step 1: CLICK("Display") -> Step 2: CLICK("Dark")', goal=dm_g, plan=dm_p,
        action_reason="Dark mode is now confirmed selected. Task complete.",
        tool_call='DONE("Dark mode successfully enabled in system display settings")'
    ))

    # 3.2 Chrome Documentation Search (3 steps)
    ch_q = "Search for PyTorch documentation in Chrome"
    ch_g = "Open Chrome, search for PyTorch documentation"
    ch_p = "1. Open Chrome 2. Enter search URL 3. View results"

    more.append(make_sample(
        "multistep_nav", ch_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "Chrome"',
        history=None, goal=ch_g, plan=ch_p,
        action_reason="Chrome not in foreground. Launching Chrome.",
        tool_call='OPEN_APP("Chrome")'
    ))

    more.append(make_sample(
        "multistep_nav", ch_q,
        screen='[SCREEN: com.android.chrome]\n[1] EditText "Search or type URL" [Input]',
        history='Step 1: OPEN_APP("Chrome")', goal=ch_g, plan=ch_p,
        action_reason="Search address bar active. Typing search query.",
        tool_call='TYPE("Search or type URL", "PyTorch documentation")'
    ))

    more.append(make_sample(
        "multistep_nav", ch_q,
        screen='[SCREEN: com.android.chrome]\n[1] TextView "PyTorch Documentation - PyTorch 2.5"\n[2] TextView "docs.pytorch.org"',
        history='Step 1: OPEN_APP("Chrome") -> Step 2: TYPE("Search or type URL", "PyTorch documentation")', goal=ch_g, plan=ch_p,
        action_reason="PyTorch documentation search result visible. Tapping first link.",
        tool_call='CLICK("PyTorch Documentation - PyTorch 2.5")'
    ))

    # 3.3 Calculator Simple Math (3 steps)
    calc_q = "Calculate 150 times 8 in Calculator"
    calc_g = "Multiply 150 by 8 in Calculator app"
    calc_p = "1. Open Calculator 2. Enter expression 3. Tap equals"

    more.append(make_sample(
        "multistep_nav", calc_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "Calculator"',
        history=None, goal=calc_g, plan=calc_p,
        action_reason="Calculator not open. Launching Calculator.",
        tool_call='OPEN_APP("Calculator")'
    ))

    more.append(make_sample(
        "multistep_nav", calc_q,
        screen='[SCREEN: com.sec.android.app.popupcalculator]\n[1] TextView "0"\n[2] Button "1"\n[3] Button "5"\n[4] Button "0"\n[5] Button "×"\n[6] Button "8"',
        history='Step 1: OPEN_APP("Calculator")', goal=calc_g, plan=calc_p,
        action_reason="Typing math expression '150 * 8'.",
        tool_call='TYPE("0", "150*8")'
    ))

    more.append(make_sample(
        "multistep_nav", calc_q,
        screen='[SCREEN: com.sec.android.app.popupcalculator]\n[1] TextView "150×8"\n[2] Button "="',
        history='Step 1: OPEN_APP("Calculator") -> Step 2: TYPE("0", "150*8")', goal=calc_g, plan=calc_p,
        action_reason="Expression entered. Tapping equals button.",
        tool_call='CLICK("=")'
    ))

    all_data = existing + more
    unique = {}
    for s in all_data:
        unique[s["prompt"]] = s

    final = list(unique.values())
    random.seed(42)
    random.shuffle(final)

    print(f"\n=======================================================")
    print(f" TOTAL PRISTINE 100% UNIQUE SAMPLES: {len(final)}")
    print(f"=======================================================")
    cat_counts = {}
    for s in final:
        c = s["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for c, count in sorted(cat_counts.items()):
        print(f"  • {c:18s}: {count:3d} samples ({count/len(final)*100:4.1f}%)")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in final:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[OK] Successfully saved {len(final)} samples to {OUTPUT_JSONL}")

if __name__ == "__main__":
    finish_to_200()
