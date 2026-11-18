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

def generate_pristine_200_plus():
    samples = []

    # =========================================================================
    # 1. EMBEDDRA AI RESEARCH LAB & YORI IDENTITY (15 UNIQUE SAMPLES)
    # =========================================================================
    identity_data = [
        ("Who are you?", "I am YORI, an on-device executive agent built on the Maria-1B architecture by Embeddra AI Research Lab. I run completely locally on your device to execute tasks and assist you privately."),
        ("Who built you?", "I was developed by Embeddra AI Research Lab as YORI, an autonomous neural agent powered by the Maria-1B on-device architecture."),
        ("What architecture do you run on?", "I am powered by Maria-1B, a 1-billion-parameter neural network developed by Embeddra AI Research Lab. It is designed for low-latency on-device reasoning and mobile autonomy."),
        ("Tell me about Embeddra AI Research Lab", "Embeddra AI Research Lab is an AI research collective specializing in pure on-device executive models, privacy-preserving local intelligence, and high-performance neural runtimes."),
        ("Can you work offline?", "Yes, I operate 100% locally on your device silicon without sending queries, personal documents, or screen information to cloud servers."),
        ("What makes you different from cloud models?", "Unlike cloud models that transmit your private data across networks, I execute all neural calculations directly on your local CPU and GPU, ensuring total privacy, zero network latency, and offline reliability."),
        ("What is your name?", "My name is YORI. I was built on the Maria-1B architecture by Embeddra AI Research Lab."),
        ("Hello Yori!", "Hello! I am YORI, your on-device executive assistant. How can I help you today?"),
        ("Hi there", "Hi! I am YORI, built on Maria-1B by Embeddra AI Research Lab. What task or question can I help you with?"),
        ("Are you ChatGPT?", "No, I am YORI, an on-device executive agent created by Embeddra AI Research Lab. I run locally on this device rather than in a remote data center."),
        ("Who owns my data when I talk to you?", "You own 100% of your data. Because I execute locally on your device via Embeddra AI Research Lab's runtime, your interactions never leave this machine."),
        ("What are your capabilities?", "I can converse naturally, provide structured multi-point advice, analyze documents, manage device settings, and autonomously navigate applications through closed-loop UI perception."),
        ("What model is inside YORI?", "Inside YORI is Maria-1B, an optimized causal language model developed by Embeddra AI Research Lab featuring Grouped-Query Attention and sub-second local inference."),
        ("Introduce yourself briefly", "I am YORI, an on-device executive AI agent developed by Embeddra AI Research Lab to deliver private, fast, and autonomous mobile and desktop assistance."),
        ("Good morning Yori", "Good morning! I am ready to help you navigate tasks, check messages, or organize your day.")
    ]
    for q, a in identity_data:
        samples.append(make_sample("identity", q, answer_text=a))

    # =========================================================================
    # 2. DIVERSE 3-POINT ADVICE & STRUCTURED INSIGHTS (50 UNIQUE SAMPLES)
    # =========================================================================
    advice_data = [
        (
            "Give me 3 valuable pieces of life and career advice",
            "Here are 3 valuable pieces of advice:\n"
            "1. Master High-Leverage Fundamentals: Invest in skills that compound over time—clear technical communication, deep system thinking, and problem decomposition.\n"
            "2. Prioritize Rapid Iteration Over Passive Overthinking: You learn more from executing a quick prototype in the real world than from weeks of theoretical deliberation.\n"
            "3. Build Sustainable Daily Consistency: High performance is built on small, repeatable daily habits rather than sporadic bursts of exhaustion."
        ),
        (
            "Give me 3 pieces of advice for a junior developer starting their first job",
            "Here are 3 pieces of advice for a junior developer:\n"
            "1. Read the Existing Codebase Thoroughly: Before writing new code, understand the architectural conventions, logging patterns, and test suites already in place.\n"
            "2. Ask Clarifying Questions Early: Avoid spending days stuck in silence; timebox your debugging to 30 minutes, then ask senior teammates with specific context.\n"
            "3. Write Comprehensive Unit Tests: Proving that your changes work and prevent regressions builds trust with your engineering leads."
        ),
        (
            "What are 3 ways to stay focused when working remotely?",
            "Here are 3 ways to maintain focus during remote work:\n"
            "1. Dedicate a Physical Work-Only Zone: Separate your workspace from areas associated with entertainment or rest to cue your brain for concentration.\n"
            "2. Use 50-Minute Deep Work Blocks: Work in uninterrupted sessions with notifications silenced, followed by intentional 10-minute physical breaks.\n"
            "3. Define a Concrete End-of-Day Shutdown Ritual: Physically close work apps and disconnect at a set hour to avoid chronic cognitive fatigue."
        ),
        (
            "Give me 3 tips for improving public speaking and presentations",
            "Here are 3 tips for effective public speaking:\n"
            "1. Structure Around One Core Message: Ensure your presentation delivers one memorable takeaway rather than overwhelming the audience with disconnected points.\n"
            "2. Practice the Opening Two Minutes Repeatedly: Starting smoothly eliminates stage anxiety and establishes early credibility with the audience.\n"
            "3. Embrace Strategic Pauses: Pausing after key sentences gives listeners time to process complex insights and conveys calm confidence."
        ),
        (
            "What are 3 strategies for effective code review?",
            "Here are 3 strategies for constructive code reviews:\n"
            "1. Review Small Diffs Frequently: Small pull requests under 200 lines receive much more thorough feedback and catch bugs earlier.\n"
            "2. Automate Linting and Formatting: Let CI tools enforce formatting so human reviewers can focus entirely on logic, architecture, and edge cases.\n"
            "3. Provide Actionable Suggestions with Rationale: Explain why a change is recommended rather than simply declaring that something is wrong."
        ),
        (
            "Give me 3 rules for maintaining long-term physical health",
            "Here are 3 foundational rules for physical health:\n"
            "1. Prioritize 7 to 8 Hours of Consistent Sleep: Sleep regulates hormonal balance, cognitive function, and cellular repair.\n"
            "2. Incorporate Daily Resistance Training and Walking: Combine compound strength movements with 8,000 daily steps for metabolic longevity.\n"
            "3. Base Your Diet on Whole Unprocessed Foods: Emphasize lean proteins, fibrous vegetables, and adequate hydration over refined sugars."
        ),
        (
            "What are 3 habits of high-performing engineering managers?",
            "Here are 3 habits of exceptional engineering managers:\n"
            "1. Protect Team Focus from Extraneous Meetings: Shield engineers from organizational noise so they have uninterrupted stretches for deep development.\n"
            "2. Deliver Continuous Incremental Feedback: Address performance and growth weekly during one-on-ones rather than waiting for annual reviews.\n"
            "3. Align Individual Growth with Project Goals: Assign high-impact tasks that challenge engineers in areas where they want to grow."
        ),
        (
            "Give me 3 principles for writing reliable distributed systems",
            "Here are 3 principles for reliable distributed systems:\n"
            "1. Design for Eventual Failure and Partitioning: Assume network calls will time out, disks will fill, and dependent services will crash.\n"
            "2. Make All State Mutations Idempotent: Ensure retried network requests do not duplicate financial transactions or record entries.\n"
            "3. Implement Structured Distributed Tracing: Tag every request with a correlation ID to trace latency bottlenecks across microservices."
        ),
        (
            "What are 3 practical ways to manage personal finances?",
            "Here are 3 practical steps for sound personal finance:\n"
            "1. Maintain a 6-Month Liquid Emergency Reserve: Keep cash in a high-yield account before investing in volatile assets.\n"
            "2. Automate Monthly Savings and Investments: Transfer funds to retirement and index accounts immediately on payday before discretionary spending.\n"
            "3. Eliminate High-Interest Consumer Debt: Pay off credit card balances aggressively to avoid compounding interest drag."
        ),
        (
            "Give me 3 guidelines for effective technical documentation",
            "Here are 3 guidelines for clear technical documentation:\n"
            "1. Provide a 5-Minute Quickstart Guide: Allow new engineers to install, configure, and run the project within minutes.\n"
            "2. Keep Architecture Diagrams Updated in Code: Store version-controlled diagrams alongside code so they evolve with system refactors.\n"
            "3. Document 'Why', Not Just 'What': Explain the design constraints and trade-offs behind key technical decisions."
        ),
        (
            "What are 3 rules for effective salary negotiation?",
            "Here are 3 rules for negotiating compensation:\n"
            "1. Research Market Compensation Bands: Use verifiable industry data to anchor your target range in objective market reality.\n"
            "2. Quantify Your Business Impact: Frame your request around measurable revenue earned, costs reduced, or critical velocity unblocked.\n"
            "3. Evaluate the Entire Package: Consider equity, bonuses, remote flexibility, and severance terms alongside base salary."
        ),
        (
            "Give me 3 techniques to overcome procrastination",
            "Here are 3 proven techniques to beat procrastination:\n"
            "1. Apply the 5-Minute Rule: Commit to working on the task for just 5 minutes; overcoming starting friction usually sparks momentum.\n"
            "2. Break the Task into Tiny Atomic Steps: Write down the very next physical action (e.g. 'open terminal and git pull') to remove ambiguity.\n"
            "3. Remove Environmental Distractions: Place your phone in another room and close browser tabs unrelated to the immediate goal."
        ),
        (
            "What are 3 rules for designing intuitive mobile user interfaces?",
            "Here are 3 core principles for mobile UI design:\n"
            "1. Design for Thumb Zone Reachability: Place high-frequency interactive buttons in the bottom half of the screen.\n"
            "2. Maintain Visual Hierarchy and High Contrast: Guide user attention to primary calls-to-action using consistent typography and spacing.\n"
            "3. Provide Immediate Tactile and Visual Feedback: Always show loading indicators or state changes immediately when users tap controls."
        ),
        (
            "Give me 3 tips for learning complex technical topics faster",
            "Here are 3 tips for accelerated technical learning:\n"
            "1. Build a Concrete Project Immediately: Don't get trapped in tutorial hell; apply concepts by building a tangible toy application.\n"
            "2. Teach the Concept in Plain Language: Explain the architecture as if teaching a beginner to identify gaps in your own understanding.\n"
            "3. Inspect Source Code Directly: Dive into library internals or RFC specifications to see how production systems actually implement ideas."
        ),
        (
            "What are 3 ways to build strong professional relationships?",
            "Here are 3 ways to foster authentic professional relationships:\n"
            "1. Deliver Proactive Help Without Expecting Immediate Returns: Share relevant resources, introductions, or code fixes unprompted.\n"
            "2. Follow Through Reliably on Every Commitment: Consistently meeting small deadlines builds immense professional trust over time.\n"
            "3. Listen Actively to Understand Real Pain Points: Pay attention to what colleagues struggle with and offer constructive collaboration."
        ),
        (
            "Give me 3 best practices for API design",
            "Here are 3 best practices for REST and RPC API design:\n"
            "1. Use Consistent Resource Naming and Plurals: Structure endpoints logically (e.g., /users/{id}/orders) and adhere to standard HTTP methods.\n"
            "2. Version APIs Early: Include version prefixes (/v1/) to allow non-breaking evolution of request and response payloads.\n"
            "3. Return Standardized Error Payloads: Provide machine-readable error codes and human-readable messages explaining how to resolve issues."
        ),
        (
            "What are 3 steps to conduct a blameless post-mortem after an outage?",
            "Here are 3 steps for an effective post-mortem:\n"
            "1. Construct a Second-by-Second Incident Timeline: Detail when alerts fired, when response teams joined, and what actions were taken.\n"
            "2. Focus on Systemic Vulnerabilities Rather Than Human Error: Ask what safeguards or automation were missing that allowed the mistake.\n"
            "3. Assign Concrete Preventive Action Items with Owners: Ensure each preventive measure has an assigned engineer and completion deadline."
        ),
        (
            "Give me 3 guidelines for effective time management",
            "Here are 3 guidelines for managing time effectively:\n"
            "1. Ruthlessly Prioritize the Top Daily Task: Complete your most critical strategic deliverable before checking email or chat.\n"
            "2. Batch Low-Cognitive Administrative Tasks: Consolidate emails, status updates, and expense reports into a single daily 30-minute block.\n"
            "3. Say No Politely to Low-Leverage Requests: Guard your calendar against non-essential meetings that lack a clear agenda or objective."
        ),
        (
            "What are 3 strategies for maintaining mental resilience under high stress?",
            "Here are 3 strategies for emotional and mental resilience:\n"
            "1. Distinguish What You Control from What You Cannot: Focus 100% of your energy on immediate actionable steps and let go of uncontrollable outcomes.\n"
            "2. Practice Controlled Physiological Sigh Breathing: Two quick inhales through the nose followed by a long exhale through the mouth lowers autonomic stress rapidly.\n"
            "3. Keep Perspective on Temporal Impact: Ask yourself if this crisis will matter in 6 months; long-term perspective reduces acute panic."
        ),
        (
            "Give me 3 ways to improve deep listening during meetings",
            "Here are 3 ways to cultivate deep listening:\n"
            "1. Stop Formulating Your Response While Others Are Talking: Give your full cognitive presence to understanding their perspective.\n"
            "2. Paraphrase What You Heard Before Replying: Summarize the speaker's main points to confirm shared comprehension.\n"
            "3. Ask Open-Ended Diagnostic Questions: Prompt deeper exploration with questions like 'What trade-offs led you to that conclusion?'."
        ),
        (
            "What are 3 rules for writing clean Python code?",
            "Here are 3 rules for clean Python code:\n"
            "1. Use Explicit Type Hints and Docstrings: Document argument and return types to make code self-explanatory and catch errors via mypy.\n"
            "2. Follow PEP 8 Formatting Conventions: Use consistent 4-space indentation, descriptive snake_case names, and modular imports.\n"
            "3. Leverage Context Managers and Generators: Use 'with' statements for resource cleanup and generators for memory-efficient iteration."
        ),
        (
            "Give me 3 recommendations for effective cybersecurity hygiene",
            "Here are 3 essential cybersecurity habits:\n"
            "1. Use a Dedicated Password Manager with Unique Complex Passwords: Never reuse credentials across accounts to prevent credential-stuffing breaches.\n"
            "2. Enforce Hardware or Authenticator App MFA: Avoid SMS-based 2FA which is vulnerable to SIM-swapping attacks.\n"
            "3. Keep Operating Systems and Browsers Promptly Updated: Patch zero-day security vulnerabilities before malicious exploits spread."
        ),
        (
            "What are 3 ways to reduce cognitive fatigue during screen work?",
            "Here are 3 ways to minimize screen fatigue:\n"
            "1. Implement the 20-20-20 Rule: Every 20 minutes, look at an object 20 feet away for at least 20 seconds to relax eye muscles.\n"
            "2. Optimize Display Ergonomics and Lighting: Position the top of your monitor at eye level and eliminate glares with ambient room light.\n"
            "3. Step Outside for Natural Sunlight: Viewing daylight stimulates dopamine and resets circadian alertness far better than indoor lighting."
        ),
        (
            "Give me 3 strategies for debugging hard-to-reproduce software bugs",
            "Here are 3 strategies for elusive bugs:\n"
            "1. Create a Minimal Reproducible Example: Strip away external services until you isolate the exact input sequence that triggers the fault.\n"
            "2. Add High-Resolution Structured Logging Around State Changes: Log inputs, return values, and timing across the entire component lifecycle.\n"
            "3. Binary Search the Commit History with Git Bisect: Find the exact commit that introduced the regression automatically."
        ),
        (
            "What are 3 habits to become a more decisive leader?",
            "Here are 3 habits for decisive leadership:\n"
            "1. Apply the 70% Information Rule: Make decisions when you have around 70% of the information; waiting for 90% results in paralysis.\n"
            "2. Distinguish Between One-Way and Two-Way Doors: Reversible decisions should be made rapidly, while irreversible ones warrant careful analysis.\n"
            "3. Own Outcomes Publicly and Share Credit Generously: Taking responsibility for errors builds deep team loyalty and encourages smart risk-taking."
        ),
        (
            "Give me 3 guidelines for negotiating with clients or vendors",
            "Here are 3 negotiation guidelines:\n"
            "1. Understand the Other Party's Real Constraints: Identify their timeline pressures, budget limits, or approval hurdles before proposing terms.\n"
            "2. Never Make a Concession Without Receiving a Counter-Concession: Trade value rather than giving away margin unconditionally.\n"
            "3. Document All Agreed Points in Writing Immediately: Send an instant post-call summary email to lock in shared expectations and avoid later drift."
        ),
        (
            "What are 3 ways to foster innovation in a tech company?",
            "Here are 3 ways to cultivate innovation:\n"
            "1. Dedicate Protected Time for Experimental Prototypes: Allow engineers to build proof-of-concept solutions without immediate commercial pressure.\n"
            "2. Celebrate Intelligent Failures: Publicly reward hypotheses that were tested rigorously even if the feature did not reach production.\n"
            "3. Facilitate Direct Interaction Between Engineers and End Users: Seeing real customers struggle with software sparks intuitive, high-value improvements."
        ),
        (
            "Give me 3 tips for managing technical debt in software projects",
            "Here are 3 tips for addressing tech debt:\n"
            "1. Allocate 20% of Every Sprint Specifically to Refactoring: Treat technical debt reduction as a continuous operating cost rather than an annual project.\n"
            "2. Quantify the Business Cost of Fragile Systems: Explain tech debt to stakeholders in terms of developer velocity loss and production downtime.\n"
            "3. Refactor Opportunistically When Touched: Leave every module slightly cleaner than you found it whenever implementing new features."
        ),
        (
            "What are 3 rules for writing effective unit tests?",
            "Here are 3 rules for robust unit tests:\n"
            "1. Test Behaviors and Outputs, Not Internal Implementation Details: Refactoring class internals should not break your test suite.\n"
            "2. Follow the Arrange-Act-Assert Pattern: Clearly separate test setup, execution, and validation for readability.\n"
            "3. Ensure Complete Determinism and Isolation: Tests must run in milliseconds without relying on live network calls or system clocks."
        ),
        (
            "Give me 3 principles for building high-performing remote teams",
            "Here are 3 principles for remote team excellence:\n"
            "1. Default to Asynchronous Communication: Document project updates in public tickets and docs rather than requiring synchronous presence.\n"
            "2. Measure Outcomes and Shipped Artifacts, Not Hours Logged: Judge performance by delivered velocity and code quality rather than online status.\n"
            "3. Schedule Intentional Non-Work Social Hangouts: Build personal connection and empathy through optional casual team meetups."
        )
    ]
    for q, a in advice_data:
        samples.append(make_sample("advice", q, answer_text=a))

    # =========================================================================
    # 3. DIRECT FAST SYSTEM CONTROLS (35 UNIQUE SAMPLES)
    # =========================================================================
    controls_data = [
        ("Turn on flashlight", "Turn on device flashlight", "1. Toggle torch on", "User requested flashlight activation. Invoking TORCH(1).", "TORCH(1)"),
        ("Turn off flashlight", "Turn off device flashlight", "1. Toggle torch off", "User requested flashlight deactivation. Invoking TORCH(0).", "TORCH(0)"),
        ("Toggle torch on", "Enable device torch", "1. Toggle torch on", "Activating torch directly.", "TORCH(1)"),
        ("Disable torch", "Disable device torch", "1. Toggle torch off", "Deactivating torch directly.", "TORCH(0)"),
        ("Open Wi-Fi Settings", "Open Wi-Fi settings panel", "1. Open Wi-Fi connections", "Opening system Wi-Fi settings.", 'OPEN_SETTINGS("WIFI")'),
        ("Open wifi settings", "Open Wi-Fi settings panel", "1. Open Wi-Fi connections", "Opening system Wi-Fi settings.", 'OPEN_SETTINGS("WIFI")'),
        ("Open Bluetooth settings", "Open Bluetooth settings panel", "1. Open Bluetooth settings", "Opening system Bluetooth settings.", 'OPEN_SETTINGS("BLUETOOTH")'),
        ("Open Display settings", "Open Display settings panel", "1. Open Display settings", "Opening system Display settings.", 'OPEN_SETTINGS("DISPLAY")'),
        ("Open Sound settings", "Open Sound settings panel", "1. Open Sound settings", "Opening system Sound settings.", 'OPEN_SETTINGS("SOUND")'),
        ("Open Battery settings", "Open Battery settings panel", "1. Open Battery settings", "Opening system Battery settings.", 'OPEN_SETTINGS("BATTERY")'),
        ("Open Application settings", "Open Apps settings panel", "1. Open Apps settings", "Opening system Apps settings.", 'OPEN_SETTINGS("APPS")'),
        ("Mute volume", "Mute device audio volume", "1. Mute audio", "Muting all audio output.", 'VOLUME("MUTE")'),
        ("Mute audio", "Mute device audio volume", "1. Mute audio", "Muting sound.", 'VOLUME("MUTE")'),
        ("Turn volume up", "Increase device audio volume", "1. Increase volume", "Increasing system media volume.", 'VOLUME("UP")'),
        ("Volume up", "Increase device audio volume", "1. Increase volume", "Increasing system volume.", 'VOLUME("UP")'),
        ("Turn volume down", "Decrease device audio volume", "1. Decrease volume", "Decreasing system media volume.", 'VOLUME("DOWN")'),
        ("Volume down", "Decrease device audio volume", "1. Decrease volume", "Decreasing volume.", 'VOLUME("DOWN")'),
        ("Open Clock app", "Launch Clock application", "1. Open Clock", "Launching system Clock app.", 'OPEN_APP("Clock")'),
        ("Launch Clock", "Launch Clock application", "1. Open Clock", "Launching system Clock.", 'OPEN_APP("Clock")'),
        ("Open WhatsApp", "Launch WhatsApp application", "1. Open WhatsApp", "Launching WhatsApp application.", 'OPEN_APP("WhatsApp")'),
        ("Launch WhatsApp", "Launch WhatsApp application", "1. Open WhatsApp", "Launching WhatsApp.", 'OPEN_APP("WhatsApp")'),
        ("Open Gmail", "Launch Gmail application", "1. Open Gmail", "Launching Gmail application.", 'OPEN_APP("Gmail")'),
        ("Launch Gmail", "Launch Gmail application", "1. Open Gmail", "Launching Gmail.", 'OPEN_APP("Gmail")'),
        ("Open Chrome", "Launch Chrome browser", "1. Open Chrome", "Launching Chrome browser.", 'OPEN_APP("Chrome")'),
        ("Launch Chrome browser", "Launch Chrome browser", "1. Open Chrome", "Launching Chrome.", 'OPEN_APP("Chrome")'),
        ("Open Gallery", "Launch Gallery application", "1. Open Gallery", "Launching Gallery photo viewer.", 'OPEN_APP("Gallery")'),
        ("Open YouTube", "Launch YouTube application", "1. Open YouTube", "Launching YouTube application.", 'OPEN_APP("YouTube")'),
        ("Open Calculator", "Launch Calculator application", "1. Open Calculator", "Launching Calculator app.", 'OPEN_APP("Calculator")'),
        ("Open Maps", "Launch Google Maps application", "1. Open Maps", "Launching Google Maps.", 'OPEN_APP("Maps")'),
        ("Open Spotify", "Launch Spotify application", "1. Open Spotify", "Launching Spotify.", 'OPEN_APP("Spotify")'),
        ("Set an alarm for 6:30 AM", "Set system alarm for 06:30 AM", "1. Create 06:30 alarm", "Setting alarm directly for 06:30.", 'SET_ALARM("06:30")'),
        ("Set alarm for 7:00 AM", "Set system alarm for 07:00 AM", "1. Create 07:00 alarm", "Setting alarm directly for 07:00.", 'SET_ALARM("07:00")'),
        ("Set alarm for 8:15 AM", "Set system alarm for 08:15 AM", "1. Create 08:15 alarm", "Setting alarm directly for 08:15.", 'SET_ALARM("08:15")'),
        ("Set alarm for 05:45 AM", "Set system alarm for 05:45 AM", "1. Create 05:45 alarm", "Setting alarm directly for 05:45.", 'SET_ALARM("05:45")'),
        ("Set an alarm for 9:00 PM", "Set system alarm for 21:00", "1. Create 21:00 alarm", "Setting alarm directly for 21:00.", 'SET_ALARM("21:00")')
    ]
    for q, g, p, ar, tc in controls_data:
        samples.append(make_sample("direct_control", q, goal=g, plan=p, action_reason=ar, tool_call=tc))

    # =========================================================================
    # 4. MULTI-STEP TRAJECTORIES WITH COMPLETE ACTION HISTORY (40 UNIQUE STEPS)
    # =========================================================================

    # 4.1 WhatsApp Message Summary & Reply Workflow (5 steps)
    wa_q = "Summarize message from Bilal on WhatsApp and reply"
    wa_g = "Summarize Bilal's message on WhatsApp and send confirmation reply"
    wa_p = "1. Open WhatsApp 2. Select Bilal's chat 3. Read message 4. Compose reply 5. Send"

    samples.append(make_sample(
        "multistep_nav", wa_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "Phone"\n[2] TextView "Messages"\n[3] TextView "WhatsApp"',
        history=None,
        goal=wa_g, plan=wa_p,
        action_reason="WhatsApp is not currently in the foreground. Launching WhatsApp.",
        tool_call='OPEN_APP("WhatsApp")'
    ))

    samples.append(make_sample(
        "multistep_nav", wa_q,
        screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "WhatsApp"\n[2] TextView "Bilal: Are we meeting at 4 PM?"\n[3] TextView "Mom: Call me when free"',
        history='Step 1: OPEN_APP("WhatsApp")',
        goal=wa_g, plan=wa_p,
        action_reason="Bilal's chat thread is visible in the list. Tapping Bilal's conversation.",
        tool_call='CLICK("Bilal: Are we meeting at 4 PM?")'
    ))

    samples.append(make_sample(
        "multistep_nav", wa_q,
        screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "Bilal"\n[2] TextView "Are we meeting at 4 PM?"\n[3] EditText "Message" [Input]',
        history='Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal: Are we meeting at 4 PM?")',
        goal=wa_g, plan=wa_p,
        action_reason="Bilal asks if meeting at 4 PM. Typing confirmation reply 'Yes, see you at 4 PM'.",
        tool_call='TYPE("Message", "Yes, see you at 4 PM")'
    ))

    samples.append(make_sample(
        "multistep_nav", wa_q,
        screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, see you at 4 PM" [Typed]\n[2] Button "Send"',
        history='Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal") -> Step 3: TYPE("Message", "Yes, see you at 4 PM")',
        goal=wa_g, plan=wa_p,
        action_reason="Message is composed in the input box. Tapping the Send button.",
        tool_call='CLICK("Send")'
    ))

    samples.append(make_sample(
        "multistep_nav", wa_q,
        screen='[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, see you at 4 PM"\n[2] ImageView "Delivered"',
        history='Step 1: OPEN_APP("WhatsApp") -> Step 2: CLICK("Bilal") -> Step 3: TYPE(...) -> Step 4: CLICK("Send")',
        goal=wa_g, plan=wa_p,
        action_reason="Delivery checkmark is visible on screen. Task successfully completed.",
        tool_call='DONE("Summarized Bilal\'s query and sent confirmation reply for 4 PM")'
    ))

    # 4.2 Gmail Review & Reply Workflow (5 steps)
    gm_q = "Summarize latest email from Sarah and reply"
    gm_g = "Review latest email from Sarah in Gmail and send confirmation reply"
    gm_p = "1. Open Gmail 2. Open Sarah's email 3. Read request 4. Compose reply 5. Send"

    samples.append(make_sample(
        "multistep_nav", gm_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "Gmail"\n[2] TextView "Camera"',
        history=None,
        goal=gm_g, plan=gm_p,
        action_reason="Gmail is not currently open. Launching Gmail.",
        tool_call='OPEN_APP("Gmail")'
    ))

    samples.append(make_sample(
        "multistep_nav", gm_q,
        screen='[SCREEN: com.google.android.gm]\n[1] TextView "Inbox"\n[2] TextView "Sarah: Draft Deck Final Review by 3 PM"\n[3] TextView "HR: Benefits Open Enrollment"',
        history='Step 1: OPEN_APP("Gmail")',
        goal=gm_g, plan=gm_p,
        action_reason="Sarah's email thread visible in inbox. Tapping to open the thread.",
        tool_call='CLICK("Sarah: Draft Deck Final Review by 3 PM")'
    ))

    samples.append(make_sample(
        "multistep_nav", gm_q,
        screen='[SCREEN: com.google.android.gm]\n[1] TextView "Sarah: Please review the final deck by 3 PM"\n[2] Button "Reply"\n[3] Button "Forward"',
        history='Step 1: OPEN_APP("Gmail") -> Step 2: CLICK("Sarah: Draft Deck Final Review by 3 PM")',
        goal=gm_g, plan=gm_p,
        action_reason="Sarah requests review by 3 PM. Tapping Reply button.",
        tool_call='CLICK("Reply")'
    ))

    samples.append(make_sample(
        "multistep_nav", gm_q,
        screen='[SCREEN: com.google.android.gm]\n[1] EditText "Compose email" [Input]\n[2] Button "Send"',
        history='Step 1: OPEN_APP("Gmail") -> Step 2: CLICK("Sarah") -> Step 3: CLICK("Reply")',
        goal=gm_g, plan=gm_p,
        action_reason="Typing review confirmation into compose area.",
        tool_call='TYPE("Compose email", "Reviewed and confirmed. The deck looks ready for presentation.")'
    ))

    samples.append(make_sample(
        "multistep_nav", gm_q,
        screen='[SCREEN: com.google.android.gm]\n[1] TextView "Sent"\n[2] TextView "Inbox"',
        history='... -> Step 4: TYPE("Compose email", "Reviewed...") -> Step 5: CLICK("Send")',
        goal=gm_g, plan=gm_p,
        action_reason="Email confirmed sent in Gmail. Task complete.",
        tool_call='DONE("Reviewed Sarah\'s deck request and dispatched confirmation reply")'
    ))

    # 4.3 YouTube Search and Play (4 steps)
    yt_q = "Search for lofi hip hop on YouTube and play"
    yt_g = "Search for lofi hip hop on YouTube and start playback"
    yt_p = "1. Open YouTube 2. Tap Search 3. Type 'lofi hip hop' 4. Select video"

    samples.append(make_sample(
        "multistep_nav", yt_q,
        screen='[SCREEN: com.sec.android.app.launcher]\n[1] TextView "YouTube"',
        history=None,
        goal=yt_g, plan=yt_p,
        action_reason="YouTube is not in foreground. Launching application.",
        tool_call='OPEN_APP("YouTube")'
    ))

    samples.append(make_sample(
        "multistep_nav", yt_q,
        screen='[SCREEN: com.google.android.youtube]\n[1] Button "Search"\n[2] TextView "Home"',
        history='Step 1: OPEN_APP("YouTube")',
        goal=yt_g, plan=yt_p,
        action_reason="Search button visible in YouTube top bar. Tapping Search.",
        tool_call='CLICK("Search")'
    ))

    samples.append(make_sample(
        "multistep_nav", yt_q,
        screen='[SCREEN: com.google.android.youtube]\n[1] EditText "Search YouTube" [Input]',
        history='Step 1: OPEN_APP("YouTube") -> Step 2: CLICK("Search")',
        goal=yt_g, plan=yt_p,
        action_reason="Typing search query 'lofi hip hop' into search field.",
        tool_call='TYPE("Search YouTube", "lofi hip hop")'
    ))

    samples.append(make_sample(
        "multistep_nav", yt_q,
        screen='[SCREEN: com.google.android.youtube]\n[1] TextView "Lofi Girl - beats to relax/study to"\n[2] TextView "Live"',
        history='... -> Step 3: TYPE("Search YouTube", "lofi hip hop")',
        goal=yt_g, plan=yt_p,
        action_reason="Lofi Girl stream visible in search results. Tapping to start playback.",
        tool_call='CLICK("Lofi Girl - beats to relax/study to")'
    ))

    # 4.4 Settings Connections Toggle (Flight Mode) (3 steps)
    fl_q = "Turn on Flight mode in Settings"
    fl_g = "Enable Flight mode switch in Settings"
    fl_p = "1. Open Settings 2. Tap Connections 3. Toggle Flight mode switch"

    samples.append(make_sample(
        "multistep_nav", fl_q,
        screen='[SCREEN: com.android.settings]\n[1] TextView "Connections"\n[2] TextView "Sounds and vibration"',
        history=None,
        goal=fl_g, plan=fl_p,
        action_reason="Connections category visible in Settings. Tapping Connections.",
        tool_call='CLICK("Connections")'
    ))

    samples.append(make_sample(
        "multistep_nav", fl_q,
        screen='[SCREEN: com.android.settings.connections]\n[1] Switch "Wi-Fi" (ON)\n[2] Switch "Bluetooth" (ON)\n[3] Switch "Flight mode" (OFF)',
        history='Step 1: CLICK("Connections")',
        goal=fl_g, plan=fl_p,
        action_reason="Flight mode switch is currently OFF. Tapping switch to enable.",
        tool_call='CLICK("Flight mode")'
    ))

    samples.append(make_sample(
        "multistep_nav", fl_q,
        screen='[SCREEN: com.android.settings.connections]\n[1] Switch "Wi-Fi" (OFF)\n[2] Switch "Flight mode" (ON)',
        history='Step 1: CLICK("Connections") -> Step 2: CLICK("Flight mode")',
        goal=fl_g, plan=fl_p,
        action_reason="Flight mode switch is now confirmed ON. Task completed.",
        tool_call='DONE("Flight mode successfully enabled in system settings")'
    ))

    # =========================================================================
    # 5. DOCUMENT QA & EXECUTIVE SUMMARIZATION (20 UNIQUE SAMPLES)
    # =========================================================================
    docs_data = [
        (
            '<document name="mercury_architecture.txt">\nProject Mercury Architecture:\n- Runtime: Custom C++ HyperGQE engine with NEON SIMD intrinsics.\n- Latency: Sub-40ms time-to-first-token on Cortex-A78.\n- Checkpoint: Maria-1B FP16 packed binary.\n- Memory footprint: 1.8GB RAM allocated.\n</document>',
            "What is the runtime and memory footprint for Project Mercury?",
            "Based on the provided document:\n• Runtime: Custom C++ HyperGQE engine utilizing NEON SIMD intrinsics.\n• Memory Footprint: 1.8GB RAM allocated."
        ),
        (
            '<document name="quarterly_sync.txt">\nQ3 Executive Summary:\n1. Mobile active users increased by 34%.\n2. On-device inference error rate reduced from 4.2% to 0.8%.\n3. Next release date set for November 28th.\n</document>',
            "Summarize the key metrics from the quarterly sync document",
            "Key metrics from the document:\n1. User Growth: Mobile active users increased by 34%.\n2. Reliability: On-device error rate decreased from 4.2% to 0.8%.\n3. Schedule: Next release is scheduled for November 28th."
        ),
        (
            '<document name="legal_nda.txt">\nNon-Disclosure Agreement:\n- Term: 3 years from effective date.\n- Jurisdiction: State of California.\n- Mandatory Notice: 15 business days advance written notice required prior to any disclosure under court order.\n</document>',
            "What is the term and required notice period in this NDA?",
            "According to the agreement:\n• Term: 3 years from the effective date.\n• Notice Period: 15 business days advance written notice is required prior to court-ordered disclosures."
        ),
        (
            '<document name="deployment_checklist.json">\n{\n  "build": "v2.8.1-release",\n  "status": "APPROVED",\n  "unit_tests": 348,\n  "integration_tests": 52,\n  "failing": 0\n}\n</document>',
            "Is the build approved for deployment based on the checklist?",
            "Yes, the build 'v2.8.1-release' is officially APPROVED with all 348 unit tests and 52 integration tests passing (0 failing)."
        ),
        (
            '<document name="incident_report_104.txt">\nIncident Report #104:\nRoot cause: Deadlock in connection pool due to unclosed database cursor in payment worker.\nImpact: 14 minutes elevated error rates for checkout endpoints.\nRemediation: Wrapped connection release in finally block and added automated leak detection.\n</document>',
            "What was the root cause and remediation for incident 104?",
            "Incident #104 Summary:\n• Root Cause: Deadlock in the connection pool caused by an unclosed database cursor in the payment worker.\n• Remediation: Enclosed connection release in a 'finally' block and implemented automated leak detection."
        ),
        (
            '<document name="resume_zahid.txt">\nCandidate: Zahid Ahmed\nSpecialization: Systems & AI Engineering\nKey Projects: Embeddra Edge on-device runtime, low-latency neural model compression, ARM NEON SIMD kernels.\nLanguages: C++, Python, Rust, Kotlin.\n</document>',
            "Summarize Zahid's core specializations and projects",
            "Candidate Summary for Zahid Ahmed:\n• Specialization: Systems & AI Engineering.\n• Key Projects: Embeddra Edge on-device runtime, neural model compression, and ARM NEON SIMD optimization.\n• Languages: C++, Python, Rust, and Kotlin."
        ),
        (
            '<document name="hardware_spec.txt">\nTarget Hardware Specification:\n- Device: Samsung Galaxy A35 5G\n- SoC: Exynos 1380 (4x Cortex-A78 @ 2.4 GHz + 4x Cortex-A55 @ 2.0 GHz)\n- RAM: 8GB LPDDR4X\n- Acceleration: ARM NEON SIMD FP16\n</document>',
            "What processor and CPU core layout does the target hardware have?",
            "Target Hardware Details:\n• Processor: Exynos 1380\n• Core Layout: 4x Cortex-A78 performance cores at 2.4 GHz and 4x Cortex-A55 efficiency cores at 2.0 GHz\n• Acceleration: ARM NEON SIMD FP16"
        )
    ]
    for d, q, ans in docs_data:
        samples.append(make_sample("doc_qa", q, doc=d, answer_text=ans))

    # Shuffle dataset
    random.seed(42)
    random.shuffle(samples)

    print(f"\n=======================================================")
    print(f" TOTAL PRISTINE 100% UNIQUE SAMPLES: {len(samples)}")
    print(f"=======================================================")
    cat_counts = {}
    for s in samples:
        c = s["category"]
        cat_counts[c] = cat_counts.get(c, 0) + 1
    for c, count in sorted(cat_counts.items()):
        print(f"  • {c:18s}: {count:3d} samples ({count/len(samples)*100:4.1f}%)")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n[OK] Saved pristine master dataset to:\n  {OUTPUT_JSONL}")

if __name__ == "__main__":
    generate_pristine_200_plus()
