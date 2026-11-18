import os
import sys

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
cache_dir = os.path.join(BASE_DIR, ".torch_cache")
os.makedirs(cache_dir, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = cache_dir
os.environ["TORCH_HOME"] = cache_dir
os.environ["TMPDIR"] = cache_dir
os.environ["TEMP"] = cache_dir

import time
import math
import random
import types
import struct
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

BASE_CKPT = os.path.join(BASE_DIR, "checkpoints", "maria_1b_20B_weights.pt")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")

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

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r=16, lora_alpha=32):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        self.r = r
        self.scaling = lora_alpha / r
        self.lora_A = nn.Parameter(torch.randn(r, original_linear.in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(original_linear.out_features, r))

    def forward(self, x):
        base = self.original_linear(x)
        lora = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base + lora.to(base.dtype)

def make_sample(category: str, query: str, screen: str = None, goal: str = None, plan: str = None, action_reason: str = None, tool_call: str = None, answer_text: str = None):
    prompt = f"{SYSTEM_CONTRACT}<user>{query}</user>\n"
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
    return {"category": category, "prompt": prompt, "target": target}

def build_balanced_dataset():
    random.seed(42)
    dataset = []

    # 1. CHAT, ADVICE, PARAGRAPHS & CONCEPTUAL EXPLANATIONS (NO ACTION)
    chat_items = [
        (
            "Give me 3 valuable pieces of life and career advice",
            "Here are 3 valuable pieces of advice:\n"
            "1. Master High-Leverage Fundamentals: Invest in skills that compound over time—clear technical communication, deep system thinking, and problem decomposition.\n"
            "2. Prioritize Rapid Iteration Over Passive Overthinking: You learn more from executing a quick prototype in the real world than from weeks of theoretical deliberation.\n"
            "3. Build Sustainable Daily Consistency: High performance is built on small, repeatable daily habits rather than sporadic bursts of exhaustion."
        ),
        (
            "3 advices pls",
            "Here are 3 practical pieces of advice:\n"
            "1. Focus on one critical priority every morning before responding to messages.\n"
            "2. Keep solutions as simple as possible—unnecessary complexity creates bugs and fragility.\n"
            "3. Welcome direct feedback and use it to calibrate your trajectory continuously."
        ),
        (
            "What is Maria-1B?",
            "Maria-1B is a 1-billion-parameter neural architecture designed specifically for on-device executive autonomy. "
            "It features grouped query attention (GQA), efficient KV caching, and sub-second token generation on mobile ARM NEON and desktop GPUs."
        ),
        (
            "Explain how on-device AI preserves privacy",
            "On-device AI executes all neural network computations locally on the device silicon. "
            "Because personal prompts, notifications, and visible screen data never leave the phone's memory, sensitive user information is completely protected from external data logging or cloud leaks."
        ),
        (
            "Who are you?",
            "I am YORI, an on-device executive agent running completely locally on your device. "
            "I can assist you with conversations, drafting replies, and autonomously executing multi-step tasks across your applications."
        ),
        (
            "Hello Yori! How can you help me today?",
            "Hello! I am ready to help you navigate apps, manage device settings, draft messages, summarize emails, and answer your questions directly on this phone."
        )
    ]
    for q, ans in chat_items:
        for _ in range(28):
            dataset.append(make_sample("chat", q, None, answer_text=ans))

    # 2. EMAIL REPLY DRAFTING & EXECUTIVE SUMMARIZATION (NO ACTION)
    draft_summary_items = [
        (
            "Draft a professional reply confirming the report will be ready by Friday",
            "Hi [Manager's Name],\n\n"
            "I wanted to confirm that the report is progressing well and will be ready for you by this Friday at 5:00 PM. Please let me know if there are specific sections you would like emphasized.\n\n"
            "Best regards,\n[Your Name]"
        ),
        (
            "Summarize the key takeaways: 'Our sync confirmed that the QA dry run is scheduled for Friday. The database migration will take place on Monday. All pull requests are frozen after Thursday.'",
            "• Friday: QA staging dry run.\n"
            "• Thursday Midnight: Pull request merge freeze.\n"
            "• Monday: Production database migration."
        ),
        (
            "Draft a polite email declining the invitation due to scheduling conflict",
            "Hi [Name],\n\n"
            "Thank you very much for the invitation. Unfortunately, I have a prior commitment at that time and will not be able to attend. I appreciate you thinking of me and hope the event goes wonderfully.\n\n"
            "Best regards,\n[Your Name]"
        ),
        (
            "Summarize the latest project updates in 2 bullet points",
            "1. Model weights merged and successfully pushed to on-device storage.\n"
            "2. Validation latency confirmed at sub-second response times on mobile NEON."
        )
    ]
    for q, ans in draft_summary_items:
        for _ in range(12):
            dataset.append(make_sample("draft_summary", q, None, answer_text=ans))

    # 2.2 DOCUMENT SUMMARIZATION & QA (NO ACTION)
    doc_items = [
        (
            '<document name="meeting_notes.txt">\nProject Mercury Sync:\n- Backend API latency reduced by 40% using C++ runtime.\n- Android APK size optimized to 38MB.\n- Next milestone: production validation on Friday.\n</document>\n\nPlease summarize the attached document.',
            "Summary of Project Mercury Sync:\n"
            "• Performance: Backend API latency decreased by 40% with C++ runtime.\n"
            "• App Size: Android APK successfully reduced to 38MB.\n"
            "• Next Milestone: Production validation scheduled for Friday."
        ),
        (
            '<document name="contract_summary.pdf">\nAgreement terms:\n1. Non-disclosure duration: 3 years.\n2. Payment terms: Net-30 upon milestone delivery.\n3. Termination notice: 14 days written notice required.\n</document>\n\nWhat are the main terms in this document?',
            "Key terms from the agreement:\n"
            "1. Confidentiality: 3-year non-disclosure obligation.\n"
            "2. Payment: Net-30 days upon milestone completion.\n"
            "3. Termination: Requires 14 days advance written notice."
        ),
        (
            '<document name="release_checklist.json">\n{\n  "version": "2.4.0",\n  "status": "Ready",\n  "tests_passed": 142,\n  "failed": 0\n}\n</document>\n\nAnalyze this document',
            "Document Analysis:\n"
            "The release checklist for version 2.4.0 is in 'Ready' status with 100% test pass rate (142 passed, 0 failed)."
        )
    ]
    for q, ans in doc_items:
        for _ in range(8):
            dataset.append(make_sample("doc_qa", q, None, answer_text=ans))

    # 2.3 MULTI-TURN CONTEXT SAMPLES (NO ACTION)
    def make_multiturn(history, query, ans):
        h_str = ""
        for u, a in history:
            h_str += f"<user>{u}</user>\n<thought>\n{a}\n</thought>\n"
        prompt = f"{SYSTEM_CONTRACT}{h_str}<user>{query}</user>\n<thought>\n"
        target = f"{ans.strip()}\n</thought>"
        return {"category": "multiturn", "prompt": prompt, "target": target}

    multiturn_items = [
        (
            [("Hi Yori", "Hello! How can I assist you today?")],
            "Give me 3 valuable pieces of life and career advice",
            "Here are 3 valuable pieces of advice:\n"
            "1. Master High-Leverage Fundamentals: Invest in skills that compound over time—clear technical communication, deep system thinking, and problem decomposition.\n"
            "2. Prioritize Rapid Iteration Over Passive Overthinking: You learn more from executing a quick prototype in the real world than from weeks of theoretical deliberation.\n"
            "3. Build Sustainable Daily Consistency: High performance is built on small, repeatable daily habits rather than sporadic bursts of exhaustion."
        ),
        (
            [("Who built you?", "I was developed as YORI, an on-device executive agent powered by Maria-1B.")],
            "Can you run offline?",
            "Yes, I operate 100% locally on your device without relying on cloud APIs or external servers."
        )
    ]
    for h, q, ans in multiturn_items:
        for _ in range(8):
            dataset.append(make_multiturn(h, q, ans))

    # 3. CORE DEVICE CONTROLS (FAST ONE-TURN WITH GOAL & PLAN)
    core_actions = [
        ("Open Wi-Fi Settings", "Open Wi-Fi Settings", "1. Open Wi-Fi settings", "User requested Wi-Fi settings. Invoking OPEN_SETTINGS.", 'OPEN_SETTINGS("WIFI")'),
        ("Open wifi settings", "Open Wi-Fi Settings", "1. Open Wi-Fi settings", "User requested Wi-Fi settings. Invoking OPEN_SETTINGS.", 'OPEN_SETTINGS("WIFI")'),
        ("Turn off Wi-Fi", "Open Wi-Fi Settings to toggle Wi-Fi", "1. Open Wi-Fi settings", "Opening Wi-Fi settings to adjust connection.", 'OPEN_SETTINGS("WIFI")'),
        ("Open Bluetooth settings", "Open Bluetooth settings", "1. Open Bluetooth settings", "User requested Bluetooth settings. Invoking OPEN_SETTINGS.", 'OPEN_SETTINGS("BLUETOOTH")'),
        ("Turn on flashlight", "Turn on device flashlight", "1. Turn on torch", "User requested flashlight on. Invoking TORCH(1).", 'TORCH(1)'),
        ("Turn off flashlight", "Turn off device flashlight", "1. Turn off torch", "User requested flashlight off. Invoking TORCH(0).", 'TORCH(0)'),
        ("Mute volume", "Mute audio volume", "1. Mute volume", "User requested muting sound. Invoking VOLUME(MUTE).", 'VOLUME("MUTE")'),
        ("Volume up", "Increase audio volume", "1. Increase volume", "User wants volume higher. Invoking VOLUME(UP).", 'VOLUME("UP")'),
        ("Check my alarms on this phone", "Check alarms on device", "1. Open Clock app to view alarms", "User wants to view device alarms. Launching Clock app.", 'OPEN_APP("Clock")'),
        ("Check Alarms", "Check alarms on device", "1. Open Clock app to view alarms", "Launching Clock app to inspect alarms.", 'OPEN_APP("Clock")'),
        ("Set an alarm for 6:30 AM", "Set device alarm for 06:30 AM", "1. Invoke SET_ALARM", "Setting alarm for 06:30 directly.", 'SET_ALARM("06:30")'),
        ("Set an alarm for 7:00 AM", "Set device alarm for 07:00 AM", "1. Invoke SET_ALARM", "Setting alarm for 07:00 directly.", 'SET_ALARM("07:00")'),
        ("Open WhatsApp", "Launch WhatsApp application", "1. Open WhatsApp", "User requested WhatsApp. Launching application.", 'OPEN_APP("WhatsApp")'),
        ("Open Chrome", "Launch Chrome browser", "1. Open Chrome", "User requested Chrome. Launching browser.", 'OPEN_APP("Chrome")'),
        ("Open Gallery", "Launch Gallery application", "1. Open Gallery", "User requested Gallery. Launching application.", 'OPEN_APP("Gallery")')
    ]
    for q, g, p, ar, tc in core_actions:
        for _ in range(5):
            dataset.append(make_sample("core_action", q, None, g, p, ar, tc))

    # 4. MULTI-STEP GOALS WITH VISIBLE SCREEN OBSERVATION
    # 4.1 WhatsApp Message & Reply
    wa_query = "Summarize message from Bilal on WhatsApp and reply"
    wa_goal = "Summarize Bilal's message on WhatsApp and send reply"
    wa_plan = "1. Open WhatsApp 2. Tap Bilal's chat 3. Read and summarize message 4. Type reply 5. Send"
    dataset.extend([
        make_sample("black_learning_ui", wa_query, None, wa_goal, wa_plan, "WhatsApp is not open. Launching WhatsApp.", 'OPEN_APP("WhatsApp")'),
        make_sample("black_learning_ui", wa_query, '[SCREEN: com.whatsapp.w4b]\n[1] TextView "WhatsApp"\n[2] TextView "Bilal: Are we meeting at 4 PM?"\n[3] TextView "Mom"', wa_goal, wa_plan, "Bilal's chat is visible. Tapping Bilal.", 'CLICK("Bilal: Are we meeting at 4 PM?")'),
        make_sample("black_learning_ui", wa_query, '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Bilal"\n[2] TextView "Are we meeting at 4 PM?"\n[3] EditText "Message" [Input]', wa_goal, wa_plan, "Bilal asked if we are meeting at 4 PM. Typing reply 'Yes, see you at 4 PM'.", 'TYPE("Message", "Yes, see you at 4 PM")'),
        make_sample("black_learning_ui", wa_query, '[SCREEN: com.whatsapp.w4b]\n[1] Button "Send"', wa_goal, wa_plan, "Reply message populated. Tapping Send.", 'CLICK("Send")'),
        make_sample("black_learning_ui", wa_query, '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Yes, see you at 4 PM"\n[2] ImageView "Delivered"', wa_goal, "Completed", "Message sent and confirmed delivered. Task complete.", 'DONE("Summarized Bilal\'s query and replied confirming 4 PM meeting")')
    ] * 6)

    # 4.2 Email Summarize & Reply Workflow
    email_query = "Summarize the latest email from Sarah and send reply"
    email_goal = "Summarize latest email from Sarah in Gmail and send confirmation reply"
    email_plan = "1. Open Gmail 2. Select email from Sarah 3. Summarize requirements 4. Tap Reply 5. Send confirmation"
    dataset.extend([
        make_sample("black_learning_ui", email_query, None, email_goal, email_plan, "Gmail is not open. Launching Gmail.", 'OPEN_APP("Gmail")'),
        make_sample("black_learning_ui", email_query, '[SCREEN: com.google.android.gm]\n[1] TextView "Inbox"\n[2] TextView "Sarah: Draft Review Final Sync"\n[3] TextView "HR Portal"', email_goal, email_plan, "Sarah's email thread visible. Opening thread.", 'CLICK("Sarah: Draft Review Final Sync")'),
        make_sample("black_learning_ui", email_query, '[SCREEN: com.google.android.gm]\n[1] TextView "Sarah: Please review the final deck by 3 PM"\n[2] Button "Reply"', email_goal, email_plan, "Email asks to review deck by 3 PM. Tapping Reply.", 'CLICK("Reply")'),
        make_sample("black_learning_ui", email_query, '[SCREEN: com.google.android.gm]\n[1] EditText "Compose email" [Input]', email_goal, email_plan, "Typing confirmation reply.", 'TYPE("Compose email", "Reviewed and confirmed. All slides look good.")'),
        make_sample("black_learning_ui", email_query, '[SCREEN: com.google.android.gm]\n[1] Button "Send"', email_goal, email_plan, "Reply is composed. Tapping Send button.", 'CLICK("Send")'),
        make_sample("black_learning_ui", email_query, '[SCREEN: com.google.android.gm]\n[1] TextView "Sent"', email_goal, "Completed", "Email sent successfully. Task complete.", 'DONE("Reviewed Sarah\'s deck request and sent confirmation reply")')
    ] * 6)

    # 4.3 Alarm Creation via Clock UI
    alarm_query = "Set an alarm for 6:30 AM in Clock"
    alarm_goal = "Create a 06:30 AM alarm in Clock application"
    alarm_plan = "1. Open Clock app 2. Tap add alarm button 3. Set time to 06:30 AM 4. Save"
    dataset.extend([
        make_sample("black_learning_ui", alarm_query, None, alarm_goal, alarm_plan, "Clock app is not open. Launching Clock.", 'OPEN_APP("Clock")'),
        make_sample("black_learning_ui", alarm_query, '[SCREEN: com.sec.android.app.clockpackage]\n[1] TextView "Alarm"\n[2] Button "Add alarm"', alarm_goal, alarm_plan, "Alarm tab is open. Tapping Add alarm '+' button.", 'CLICK("Add alarm")'),
        make_sample("black_learning_ui", alarm_query, '[SCREEN: com.sec.android.app.clockpackage.edit]\n[1] TimePicker "Hour"\n[2] TimePicker "Minute"\n[3] Button "Save"', alarm_goal, alarm_plan, "Time is set to 06:30. Tapping Save to store alarm.", 'CLICK("Save")'),
        make_sample("black_learning_ui", alarm_query, '[SCREEN: com.sec.android.app.clockpackage]\n[1] Switch "6:30 AM" (ON)', alarm_goal, "Completed", "Alarm for 6:30 AM is active. Task complete.", 'DONE("Alarm for 6:30 AM created and enabled")')
    ] * 6)

    # 5. OBSTACLE RECOVERY (CLOSED LOOP UNBLOCKING)
    dataset.extend([
        make_sample(
            "obstacle_recovery",
            "Open chrome, navigate to indeed jobs, and apply for AI Engineer position",
            '[SCREEN: com.android.chrome]\n[1] TextView "Cookie Preferences"\n[2] Button "Accept all cookies"\n[3] Button "Reject all"',
            "Open Chrome and apply for AI Engineer position on Indeed",
            "1. Clear blocking cookie modal 2. Fill application form 3. Submit",
            "A cookie overlay is blocking the form. Tapping 'Accept all cookies' to clear it.",
            'CLICK("Accept all cookies")'
        ),
        make_sample(
            "obstacle_recovery",
            "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp",
            '[SCREEN: com.google.android.permissioncontroller]\n[1] TextView "Allow Gallery to access photos and videos on this device?"\n[2] Button "Allow"\n[3] Button "Don\'t allow"',
            "Find recent photo in Gallery Camera album and share via WhatsApp",
            "1. Grant storage permission 2. Open Camera album 3. Share photo",
            "A system permission dialog is requesting storage access. Tapping 'Allow'.",
            'CLICK("Allow")'
        ),
        make_sample(
            "obstacle_recovery",
            "Open chatgpt and write a message about HyperGQE",
            '[SCREEN: com.openai.chatgpt]\n[1] View "Ask ChatGPT"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]',
            "Open ChatGPT and compose message about HyperGQE",
            "1. Bypass unresponsive Menu 2. Type directly into Ask ChatGPT 3. Send",
            "Previous click on Menu had no effect. Typing directly into visible 'Ask ChatGPT'.",
            'TYPE("Ask ChatGPT", "Explain HyperGQE architecture")'
        )
    ] * 12)

    random.shuffle(dataset)
    print(f"Total Perfectly Balanced Dataset Size: {len(dataset)} samples")
    return dataset

def export_to_eer(pt_or_sd, eer_path: str):
    print(f"\n[EXPORT] Converting state dict -> .eer binary: {eer_path}...", flush=True)
    if isinstance(pt_or_sd, dict):
        sd = pt_or_sd
    else:
        sd = torch.load(pt_or_sd, map_location="cpu", weights_only=False)
        if "model_state_dict" in sd:
            sd = sd["model_state_dict"]

    MAGIC = b"EER1"
    VERSION = 1
    VOCAB_SIZE = 128256
    HIDDEN_SIZE = 2048
    INTERMEDIATE = 5632
    NUM_LAYERS = 18
    NUM_HEADS = 16
    NUM_KV_HEADS = 4
    HEAD_DIM = 128
    MAX_SEQ_LEN = 8192
    DTYPE_FP16 = 1

    entries = []
    raw_data_blobs = []
    current_offset = 0

    tensor_names = sorted(list(sd.keys()))
    for name in tensor_names:
        t = sd[name].detach().cpu().half().contiguous()
        shape = list(t.shape)
        data_bytes = t.numpy().tobytes()
        entries.append((name, shape, DTYPE_FP16, len(data_bytes), data_bytes))

    index_buf = bytearray()
    for name, shape, dtype_code, size_bytes, _ in entries:
        name_bytes = name.encode("utf-8")
        index_buf.extend(struct.pack("<HB", len(name_bytes), len(shape)))
        index_buf.extend(name_bytes)
        index_buf.extend(struct.pack(f"<{len(shape)}I", *shape))
        index_buf.extend(struct.pack("<BQQ", dtype_code, 0, size_bytes))

    header_len = 48 + 4
    index_len = len(index_buf)
    data_start = header_len + index_len

    final_index_buf = bytearray()
    blob_offset = data_start
    for name, shape, dtype_code, size_bytes, data_bytes in entries:
        name_bytes = name.encode("utf-8")
        final_index_buf.extend(struct.pack("<HB", len(name_bytes), len(shape)))
        final_index_buf.extend(name_bytes)
        final_index_buf.extend(struct.pack(f"<{len(shape)}I", *shape))
        final_index_buf.extend(struct.pack("<BQQ", dtype_code, blob_offset, size_bytes))
        blob_offset += size_bytes

    header = struct.pack(
        "<4sIIIIIIIIIII",
        MAGIC,
        VERSION,
        VOCAB_SIZE,
        HIDDEN_SIZE,
        INTERMEDIATE,
        NUM_LAYERS,
        NUM_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        MAX_SEQ_LEN,
        DTYPE_FP16,
        len(entries)
    )

    with open(eer_path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", len(final_index_buf)))
        f.write(final_index_buf)
        for _, _, _, _, data_bytes in entries:
            f.write(data_bytes)

    print(f"[OK] Exported {len(entries)} tensors to {eer_path} ({os.path.getsize(eer_path)/(1024*1024):.1f} MB)", flush=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING PERFECT EXECUTIVE YORI ON {device} ===", flush=True)

    tokenizer = Tokenizer.from_file(TOK_PATH)
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float32)

    print(f"Loading pristine base checkpoint: {BASE_CKPT}...", flush=True)
    pt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = pt["model_state_dict"] if "model_state_dict" in pt else pt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Base weights loaded (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)

    for p in model.parameters():
        p.requires_grad = False

    print("Injecting LoRA adapters (r=16, alpha=32) into Attention & SwiGLU FFN...", flush=True)
    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, r=16, lora_alpha=32).to(device)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, r=16, lora_alpha=32).to(device)
        block.attn.o_proj = LoRALinear(block.attn.o_proj, r=16, lora_alpha=32).to(device)
        block.ffn.gate = LoRALinear(block.ffn.gate, r=16, lora_alpha=32).to(device)
        block.ffn.up = LoRALinear(block.ffn.up, r=16, lora_alpha=32).to(device)
        block.ffn.down = LoRALinear(block.ffn.down, r=16, lora_alpha=32).to(device)

    model.to(device=device)
    dataset = build_balanced_dataset()

    print("Pre-tokenizing balanced dataset...", flush=True)
    tokenized_dataset = []
    for item in dataset:
        full = item["prompt"] + item["target"]
        full_ids = tokenizer.encode(full).ids
        prompt_ids = tokenizer.encode(item["prompt"]).ids
        p_len = len(prompt_ids) - 1
        tokenized_dataset.append((full_ids, p_len))
    print(f"Pre-tokenized {len(tokenized_dataset)} samples successfully.", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    # Fast 1-epoch sweep over balanced dataset
    optimizer = torch.optim.AdamW(trainable_params, lr=1.1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    BATCH_SIZE = 4
    EPOCHS = 1
    model.train()

    t0 = time.time()
    step = 0
    total_steps = (len(tokenized_dataset) // BATCH_SIZE) * EPOCHS
    ema_loss = 5.0
    print(f"\n--- Starting 1-Epoch Balanced Sweep ({total_steps} Steps) ---", flush=True)

    done_training = False
    for epoch in range(1, EPOCHS + 1):
        shuffled = list(tokenized_dataset)
        random.shuffle(shuffled)
        for b_idx in range(0, len(shuffled) - BATCH_SIZE + 1, BATCH_SIZE):
            step += 1
            batch = shuffled[b_idx : b_idx + BATCH_SIZE]
            optimizer.zero_grad()
            loss_total = 0.0

            for full_ids, p_len in batch:
                input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
                labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)
                labels[0, :p_len] = -100

                with torch.amp.autocast('cuda', dtype=torch.float16):
                    logits = model(input_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

                scaler.scale(loss / len(batch)).backward()
                loss_total += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            avg_loss = loss_total / len(batch)
            ema_loss = 0.85 * ema_loss + 0.15 * avg_loss

            if step % 10 == 0 or step == 1 or step == total_steps:
                print(f"Epoch {epoch}/{EPOCHS} | Step {step:3d}/{total_steps} | Loss: {avg_loss:.4f} (EMA: {ema_loss:.4f}) | Elapsed: {time.time()-t0:.1f}s", flush=True)

    lora_sd = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_sd, os.path.join(BASE_DIR, "lora_perfect_executive.pt"))
    print("Saved LoRA adapter checkpoint to lora_perfect_executive.pt", flush=True)

    print("\nMerging LoRA weights into base weights...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for attr in ["q_proj", "v_proj", "o_proj"]:
                mod = getattr(block.attn, attr)
                delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                mod.original_linear.weight.data.add_(delta.to(mod.original_linear.weight.dtype))
                setattr(block.attn, attr, mod.original_linear)
            for attr in ["gate", "up", "down"]:
                mod = getattr(block.ffn, attr)
                delta = (mod.lora_B @ mod.lora_A) * mod.scaling
                mod.original_linear.weight.data.add_(delta.to(mod.original_linear.weight.dtype))
                setattr(block.ffn, attr, mod.original_linear)

    export_to_eer(model.state_dict(), OUT_EER)
    print("\n=== TRAINING & EXPORT COMPLETE ===", flush=True)

if __name__ == "__main__":
    main()
