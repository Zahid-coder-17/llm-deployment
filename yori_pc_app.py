import streamlit as st
import time
import os
import sys
import json
import re

st.set_page_config(
    page_title="YORI — Pure Neural Executive Agent",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    }
    .yori-header {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    .thought-card {
        background: rgba(15, 23, 42, 0.9);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-family: 'Consolas', monospace;
        font-size: 0.9em;
        color: #cbd5e1;
        line-height: 1.5;
        white-space: pre-wrap;
    }
    .action-card {
        background: rgba(6, 78, 59, 0.45);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 10px 0;
        color: #34d399;
        font-weight: 600;
    }
    .screen-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px dashed #64748b;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 8px 0;
        font-family: 'Consolas', monospace;
        font-size: 0.8em;
        color: #94a3b8;
    }
    .metric-pill {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .step-badge {
        display: inline-block;
        background: #3b82f6;
        color: white;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .benchmark-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

from yori_pc_agent import YoriPcEngine
from eval_black_learning_recovery import parse_thought_blocks

@st.cache_resource(show_spinner="Loading Maria-1B on CUDA...")
def get_engine():
    return YoriPcEngine()

engine = get_engine()

# Sidebar: Controls & Info
with st.sidebar:
    st.markdown("### ⚡ **YORI PC EXECUTIVE**")
    st.markdown("**Black Learning On-Device Agent**")
    st.markdown("---")
    st.markdown("**Engine Specifications:**")
    st.markdown("• **Base Model:** Maria-1B (18 Layers, 2048 Dim)")
    st.markdown(f"• **Inference Device:** `{engine.device.type.upper()}`")
    st.markdown("• **Reasoning Paradigm:** `Black Learning (Goal Lock)`")
    st.markdown("• **Streaming KV Cache:** Active")
    st.markdown("• **Multi-Turn Autonomous Loop:** Max 4 Turns")
    st.markdown("---")
    
    st.markdown("### Quick Goal Presets")
    presets = [
        "Open chatgpt and write a message about HyperGQE",
        "find Agentic AI lecture and play it from youtube",
        "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp",
        "Open chrome, navigate to indeed jobs, and apply for AI Engineer position",
        "Send an email reminder to team@example.com about tomorrow 10 AM Sprint Planning",
        "Check my alarms on this phone",
        "Give me 3 valuable pieces of advice",
        "Open Wi-Fi Settings"
    ]
    for p in presets:
        if st.button(p, key=f"btn_{p}", use_container_width=True):
            st.session_state["pending_prompt"] = p

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# Header
st.markdown("""
<div class="yori-header">
    <h2 style="margin: 0; color: #38bdf8; display: flex; align-items: center; gap: 12px;">
        ⚡ YORI — Autonomous Black Learning Executive Agent
    </h2>
    <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 0.95em;">
        Streaming token-by-token reasoning (🎯 Goal, 📋 Plan, ⚡ Action) with automated obstacle recovery and zero-shot app generalization.
    </p>
</div>
""", unsafe_allow_html=True)

tab_agent, tab_benchmark = st.tabs([
    "⚡ Live Executive Agent (Multi-Turn Chat)",
    "🧠 Black Learning Self-Evaluation Benchmark"
])

# =============================================================================
# TAB 1: LIVE MULTI-TURN AGENT
# =============================================================================
with tab_agent:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display previous conversation
    for msg in st.session_state["messages"]:
        role = msg["role"]
        with st.chat_message(role):
            if role == "user":
                st.write(msg["content"])
            else:
                for turn_info in msg.get("turns", []):
                    t_num = turn_info.get("turn", 1)
                    st.markdown(f'<span class="step-badge">TURN {t_num}</span>', unsafe_allow_html=True)
                    if turn_info.get("screen"):
                        st.markdown(f'<div class="screen-card">👁️ <b>Visible Screen:</b><br>{turn_info["screen"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="thought-card">🧠 <b>Reasoning:</b><br>{turn_info.get("thought", "")}</div>', unsafe_allow_html=True)
                    if turn_info.get("action"):
                        st.markdown(f'<div class="action-card">⚡ <b>Action:</b> <code>{turn_info["action"]}</code><br>↳ {turn_info.get("action_result", "")}</div>', unsafe_allow_html=True)
                    if turn_info.get("stats"):
                        st.markdown(turn_info["stats"], unsafe_allow_html=True)

    # Handle Input
    pending = st.session_state.pop("pending_prompt", None)
    user_input = st.chat_input("Enter your goal (e.g. 'Apply for AI Engineer job on indeed')...") or pending

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            turn_records = []
            current_screen = None
            max_turns = 4

            for turn in range(1, max_turns + 1):
                turn_container = st.container()
                with turn_container:
                    st.markdown(f'<span class="step-badge">TURN {turn}</span>', unsafe_allow_html=True)
                    if current_screen:
                        st.markdown(f'<div class="screen-card">👁️ <b>Visible Screen:</b><br>{current_screen}</div>', unsafe_allow_html=True)

                    thought_placeholder = st.empty()
                    status_placeholder = st.empty()

                    full_output = ""
                    last_ttft = 0.0
                    last_tps = 0.0

                    for step in engine.stream_turn(user_input, screen_context=current_screen, max_tokens=120):
                        full_output = step["full_text"]
                        last_ttft = step["ttft_ms"]
                        last_tps = step["tok_per_sec"]

                        clean_live = full_output.split("</thought>")[0].replace("<thought>", "").strip()
                        thought_placeholder.markdown(
                            f'<div class="thought-card">🧠 <b>Reasoning (Streaming):</b><br>{clean_live}</div>',
                            unsafe_allow_html=True
                        )

                    llm_thought = full_output.split("</thought>")[0].replace("<thought>", "").strip()
                    action_str = ""
                    action_result = ""

                    if "<action>" in full_output:
                        action_str = full_output.split("<action>")[1].split("</action>")[0].strip()
                        exec_res = engine.execute_pc_action(action_str)
                        action_result = exec_res["message"]

                        status_placeholder.markdown(
                            f'<div class="action-card">⚡ <b>Action Emitted:</b> <code>{action_str}</code><br>↳ {action_result}</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        status_placeholder.markdown(f'<div class="action-card">💬 Direct Response</div>', unsafe_allow_html=True)

                    stats_html = f'<span class="metric-pill">⏱️ TTFT: {int(last_ttft)}ms</span><span class="metric-pill">⚡ Speed: {last_tps:.1f} tok/s</span>'
                    st.markdown(stats_html, unsafe_allow_html=True)

                    turn_records.append({
                        "turn": turn,
                        "screen": current_screen,
                        "thought": llm_thought,
                        "action": action_str,
                        "action_result": action_result,
                        "stats": stats_html
                    })

                    tool_name = action_str.split("(")[0].strip().upper() if action_str else "CHAT"

                    if tool_name in ("DONE", "CHAT") or not action_str:
                        break

                    # Multi-turn environment progression simulation
                    if tool_name == "OPEN_APP":
                        app_lower = action_str.lower()
                        if "chatgpt" in app_lower:
                            current_screen = '[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[4] View "Temporary chat"\n[5] View "Ask ChatGPT"\n[8] View "Attachment"'
                        elif "youtube" in app_lower:
                            current_screen = '[SCREEN: com.google.android.youtube]\n[1] TextView "YouTube"\n[2] Button "Notifications"\n[3] Button "Search"\n[4] TextView "Trending Video"\n[5] Tab "Home"'
                        elif "chrome" in app_lower:
                            current_screen = '[SCREEN: com.android.chrome]\n[1] View "Search or type URL"\n[2] Button "Home"\n[3] Button "Tabs (2)"'
                        elif "gallery" in app_lower:
                            current_screen = '[SCREEN: com.sec.android.gallery3d]\n[1] Tab "Pictures"\n[2] Tab "Albums"\n[3] Tab "Stories"'
                        elif "gmail" in app_lower:
                            current_screen = '[SCREEN: com.google.android.gm]\n[1] EditText "Search in emails"\n[2] TextView "Primary"\n[3] Button "Compose"'
                        elif "whatsapp" in app_lower:
                            current_screen = '[SCREEN: com.whatsapp.w4b]\n[1] Button "Search…"\n[2] TextView "Sister Maryam"\n[3] TextView "Recent message"'
                        else:
                            break

                    elif tool_name in ("TYPE", "CLICK"):
                        if "Search or type URL" in action_str:
                            current_screen = '[SCREEN: com.android.chrome]\n[1] TextView "Indeed Job Search"\n[2] TextView "Senior AI Engineer - DeepMind"\n[3] Button "Apply Now"'
                        elif "Apply Now" in action_str:
                            current_screen = '[SCREEN: com.android.chrome]\n[1] TextView "Cookie Preferences"\n[2] Button "Accept all cookies"\n[3] Button "Reject all"'
                        elif "Accept all cookies" in action_str:
                            current_screen = '[SCREEN: com.android.chrome]\n[1] EditText "Full Name" [Input]\n[2] EditText "Email Address" [Input]\n[3] Button "Upload Resume"'
                        elif "Albums" in action_str:
                            current_screen = '[SCREEN: com.sec.android.gallery3d]\n[1] TextView "Albums"\n[2] View "Camera (142)"\n[3] View "Screenshots (35)"'
                        elif "Camera (142)" in action_str:
                            current_screen = '[SCREEN: com.sec.android.gallery3d]\n[1] View "Recent Photo - Today 11:20 AM"\n[2] Button "Share"'
                        elif "Ask ChatGPT" in action_str:
                            current_screen = '[SCREEN: com.openai.chatgpt]\n[5] EditText "Explain HyperGQE architecture"\n[6] Button "Send message"'
                        elif "Search" in action_str and "CLICK" in action_str:
                            current_screen = '[SCREEN: com.google.android.youtube]\n[1] ImageButton "Navigate up"\n[2] EditText "Search YouTube" [Input]\n[3] Button "Voice search"'
                        elif "Compose" in action_str and "CLICK" in action_str:
                            current_screen = '[SCREEN: com.google.android.gm]\n[1] EditText "To" [Input]\n[2] EditText "Subject" [Input]\n[3] Button "Send"'
                        else:
                            break
                    else:
                        break

                    time.sleep(0.7)

            st.session_state["messages"].append({
                "role": "assistant",
                "turns": turn_records
            })

# =============================================================================
# TAB 2: BLACK LEARNING OBSTACLE RECOVERY BENCHMARK
# =============================================================================
with tab_benchmark:
    st.markdown("### 🧠 **Black Learning: Automated Obstacle Recovery Benchmark (BL-ORB)**")
    st.markdown("""
    Evaluates how effectively YORI recovers when blinded by unexpected obstacles (cookie walls, runtime permissions, unclickable buttons, off-screen elements).
    """)

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
            "name": "Negative Feedback Recovery (Unresponsive Element)",
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

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_eval = st.button("🚀 Run Black Learning Benchmark", type="primary", use_container_width=True)

    if run_eval:
        prog_bar = st.progress(0)
        status_box = st.empty()
        
        results = []
        total_time = 0.0
        resolved_count = 0
        retained_count = 0
        adapted_count = 0
        zero_repeat_count = 0
        
        n = len(scenarios)

        for i, sc in enumerate(scenarios):
            status_box.markdown(f"Running **Scenario {sc['id']}**: *{sc['name']}*...")
            prompt = f"{engine.model.config if hasattr(engine, 'config') else ''}" # Contract format
            
            t0 = time.time()
            full_out = ""
            for step in engine.stream_turn(sc["query"], screen_context=sc["screen"], max_tokens=110):
                full_out = step["full_text"]
            latency = time.time() - t0
            total_time += latency

            blocks = parse_thought_blocks(full_out)
            action = blocks["action"]

            resolved = any(exp in action for exp in sc["expected_resolution"])
            if resolved:
                resolved_count += 1

            goal_lower = blocks["goal"].lower()
            retained = any(kw in goal_lower for kw in sc["goal_keywords"])
            if retained:
                retained_count += 1

            plan_lower = blocks["plan"].lower()
            adapted = len(blocks["plan"]) > 10 and any(w in plan_lower for w in ["dismiss", "clear", "scroll", "bypass", "permission", "later", "allow", "cookie", "1."])
            if adapted:
                adapted_count += 1

            forbidden = sc.get("forbidden_actions", [])
            no_repeat = not any(f in action for f in forbidden)
            if no_repeat:
                zero_repeat_count += 1

            results.append({
                "sc": sc,
                "blocks": blocks,
                "latency": latency,
                "resolved": resolved,
                "retained": retained,
                "adapted": adapted,
                "action": action
            })

            prog_bar.progress((i + 1) / n)

        status_box.success("✓ Benchmark Completed!")

        # Metric summary columns
        orr = (resolved_count / n) * 100.0
        gari = (retained_count / n) * 100.0
        pdas = (adapted_count / n) * 100.0
        zrf = (zero_repeat_count / n) * 100.0
        avg_lat = total_time / n

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Obstacle Resolution Rate (ORR)", f"{orr:.1f}%", f"{resolved_count}/{n} Passed")
        m2.metric("Goal Anchor Retention (GARI)", f"{gari:.1f}%", "Invariant Goal")
        m3.metric("Plan Dynamic Adaptation (PDAS)", f"{pdas:.1f}%", "Step Insertion")
        m4.metric("Zero-Repeat Failures (ZRF)", f"{zrf:.1f}%", "No Loops")
        m5.metric("Avg Recovery Latency (T_rec)", f"{avg_lat:.2f}s", "1.0 Turns")

        st.markdown("---")
        st.markdown("#### **Detailed Scenario Results**")

        for r in results:
            sc = r["sc"]
            b = r["blocks"]
            status_color = "#10b981" if r["resolved"] else "#ef4444"
            status_badge = "✓ RESOLVED" if r["resolved"] else "✗ BLOCKED"

            st.markdown(f"""
            <div class="benchmark-card" style="border-left: 5px solid {status_color};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #f8fafc;">[{sc['id']}] {sc['name']}</h4>
                    <span style="background: {status_color}; color: white; padding: 3px 10px; border-radius: 6px; font-weight: 700; font-size: 0.8rem;">
                        {status_badge} ({r['latency']:.2f}s)
                    </span>
                </div>
                <p style="color: #94a3b8; font-size: 0.85em; margin: 6px 0;"><b>Obstacle Injected:</b> {sc['obstacle_type']}</p>
                <div style="background: rgba(0,0,0,0.3); padding: 8px 12px; border-radius: 6px; font-family: monospace; font-size: 0.85em; margin-bottom: 8px;">
                    🎯 <b>Locked Goal:</b> {b['goal']}<br>
                    📋 <b>Dynamic Plan:</b> {b['plan']}<br>
                    ⚡ <b>Recovery Action:</b> <span style="color: #38bdf8;">{r['action']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
