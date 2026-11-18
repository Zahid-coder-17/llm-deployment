import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import math
import time
import types
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge", "export", "mobile_export"))

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

BASE_CKPT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
VAL_FILE = os.path.join(BASE_DIR, "hf_cache", "sweetspot_val.jsonl")

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

def make_sample(category: str, query: str, screen: str, goal: str = None, plan: str = None, action_reason: str = None, tool_call: str = None, answer_text: str = None):
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

def build_sweetspot_dataset():
    random.seed(42)
    dataset = []

    # =========================================================================
    # CATEGORY 1: CHATTING, EXPLANATIONS & ELOQUENT PARAGRAPHS
    # =========================================================================
    chat_items = [
        (
            "Give me 3 valuable pieces of life and career advice",
            "Here are 3 profound principles that create compounding value in life and career:\n\n"
            "1. Focus on High-Leverage Skills: Master skills that don't depreciate—clear technical communication, deep architectural thinking, and rapid problem decomposition. These compound across any industry shifts.\n\n"
            "2. Bias Toward Action and Iteration: Intellectualizing problems in isolation creates paralysis. Real-world feedback from early execution is vastly richer than abstract deliberation.\n\n"
            "3. Consistency Over Intensity: Sustainable daily momentum always outperforms sporadic bursts of exhaustion. Establish repeatable routines and protect your focus ruthlessly."
        ),
        (
            "What is Maria-1B and why is on-device inference important?",
            "Maria-1B is a high-throughput 1-billion-parameter causal language model engineered specifically for edge execution on consumer mobile hardware and embedded NPUs.\n\n"
            "On-device inference represents a critical paradigm shift in AI deployment for three core reasons:\n"
            "- Zero Network Latency: Responses and action decisions are computed entirely locally on device silicon, eliminating cloud round-trips and enabling real-time closed-loop executive autonomy.\n"
            "- Total Privacy & Security: Personal user context, sensitive screen views, and local notifications never leave the device boundary, completely safeguarding user confidentiality.\n"
            "- Offline Autonomy: The system remains fully operational without internet connectivity, ensuring consistent reliability anywhere in the world."
        ),
        (
            "Explain the difference between Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL)",
            "Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) address fundamentally different phases of model capability:\n\n"
            "Supervised Fine-Tuning trains the model via maximum likelihood estimation to mimic high-quality target demonstrations. It teaches format adherence, tone, and surface syntactical patterns. However, SFT suffers from compounding exposure bias when the environment deviates from training demonstrations.\n\n"
            "Reinforcement Learning (such as PPO or GRPO) optimizes behavior through reward signals derived from execution outcomes. Rather than memorizing exact token sequences, the model learns exploratory problem-solving, self-correction, and long-horizon policy optimization based on environmental consequences."
        ),
        (
            "How does group query attention (GQA) reduce KV cache memory?",
            "Grouped Query Attention (GQA) is an architectural interpolation between Multi-Head Attention (MHA) and Multi-Query Attention (MQA).\n\n"
            "In standard MHA, each query head possesses its own dedicated Key and Value heads, causing the KV cache to scale linearly with the number of attention heads. In GQA, multiple query heads share a single Key-Value head group. This dramatically reduces the memory footprint and bandwidth required to store and reload cached past tokens during auto-regressive generation, while preserving over 99% of full multi-head expressive capacity."
        )
    ]
    for q, ans in chat_items:
        for _ in range(16):
            dataset.append(make_sample("chat", q, None, answer_text=ans))

    # =========================================================================
    # CATEGORY 2: GENERATING REPLIES & PROFESSIONAL COMPOSING
    # =========================================================================
    reply_items = [
        (
            "Draft a professional reply to my manager confirming I will deliver the project report by Friday 5 PM",
            "Here is a polished, professional response for your manager:\n\n"
            "Hi [Manager's Name],\n\n"
            "Thanks for checking in. I wanted to confirm that the project report is on schedule and proceeding smoothly. I am currently finalizing the analysis and will deliver the complete document to you by Friday at 5:00 PM.\n\n"
            "Please let me know if there are any specific emphasis areas you would like highlighted in the executive summary.\n\n"
            "Best regards,\n[Your Name]"
        ),
        (
            "Draft a polite decline email for a speaking invitation due to scheduling conflicts",
            "Here is a courteous email declining the invitation:\n\n"
            "Dear [Organizer's Name],\n\n"
            "Thank you very much for inviting me to speak at [Event Name]. I truly appreciate you thinking of me for this opportunity.\n\n"
            "Unfortunately, due to prior commitments scheduled for that date, I will be unable to accept the invitation this time. I hope the event is a resounding success, and I would welcome staying in touch for future collaborations.\n\n"
            "Warm regards,\n[Your Name]"
        ),
        (
            "Compose a concise Slack message letting the team know deployment is complete and stable",
            "🚀 Team Update: Today's production deployment (v2.4.0) has finished successfully. All health checks, latency benchmarks, and regression suites are green. Monitoring is active. Thank you everyone for the great work!"
        )
    ]
    for q, ans in reply_items:
        for _ in range(16):
            dataset.append(make_sample("reply_drafting", q, None, answer_text=ans))

    # =========================================================================
    # CATEGORY 3: SUMMARIZATION & INFORMATION DISTILLATION
    # =========================================================================
    summary_items = [
        (
            "Summarize this email thread into 3 key takeaways: 'Team, during our sync today we agreed to postpone the database migration from Wednesday to next Monday to give the QA team 48 hours for load testing. John will oversee the staging dry run on Friday. All developers must freeze PR merges by Thursday midnight.'",
            "Here is the executive summary:\n\n"
            "• Migration Postponed: Database migration moved to next Monday to allow 48 hours for QA load testing.\n"
            "• Friday Dry Run: John will lead the staging dry run on Friday.\n"
            "• Code Freeze: All pull request merges freeze by Thursday midnight."
        ),
        (
            "Summarize the main benefits of on-device neural agents in 3 bullet points",
            "• Ultra-Low Latency: Instantaneous execution without network round-trips or cloud queuing.\n"
            "• Complete Data Privacy: User context, screens, and personal inputs remain strictly confined to device silicon.\n"
            "• Uninterrupted Resilience: Operates continuously regardless of network connectivity or cloud server outages."
        )
    ]
    for q, ans in summary_items:
        for _ in range(16):
            dataset.append(make_sample("summarization", q, None, answer_text=ans))

    # =========================================================================
    # CATEGORY 4: MULTI-STEP BLACK LEARNING UI AUTOMATION
    # =========================================================================
    # 4.1 Chrome Job Application Pipeline
    cj_query = "Open chrome, navigate to indeed jobs, and apply for AI Engineer position"
    c_goal = "Open Chrome and apply for AI Engineer position on Indeed"
    c_plan = "1. Open Chrome 2. Navigate to indeed.com/jobs 3. Select AI Engineer role 4. Fill application form 5. Submit"

    dataset.extend([
        make_sample("black_learning_ui", cj_query, None, c_goal, c_plan, "Chrome is not currently open. I must launch Chrome first.", 'OPEN_APP("Chrome")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] View "Search or type URL"\n[2] Button "Home"', c_goal, c_plan, "Chrome is active. I will tap the URL address bar to enter the website.", 'CLICK("Search or type URL")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] EditText "Search or type URL" [Input]\n[2] Button "Voice search"', c_goal, c_plan, "URL bar is focused. I will enter indeed.com/jobs to navigate to job listings.", 'TYPE("Search or type URL", "indeed.com/jobs")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] TextView "Indeed"\n[2] TextView "Senior AI Engineer - DeepMind"\n[3] Button "Apply Now"', c_goal, c_plan, "Job listings loaded. I see 'Senior AI Engineer - DeepMind' with 'Apply Now'. I will tap Apply Now.", 'CLICK("Apply Now")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] TextView "Application Form"\n[2] EditText "Full Name" [Input]\n[3] EditText "Email Address" [Input]', c_goal, c_plan, "Application form is displayed. I will enter the applicant full name.", 'TYPE("Full Name", "Alex Chen")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] EditText "Alex Chen"\n[2] EditText "Email Address" [Input]\n[3] Button "Submit Application"', c_goal, c_plan, "Full name entered. Now I will enter the contact email address.", 'TYPE("Email Address", "alex.chen@embeddra.ai")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] EditText "alex.chen@embeddra.ai"\n[2] Button "Submit Application"', c_goal, c_plan, "All fields are populated. I will tap Submit Application to finish.", 'CLICK("Submit Application")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] TextView "Application Submitted Successfully!"', c_goal, "Completed", "Application has been submitted and confirmed by the portal. Task complete.", 'DONE("AI Engineer job application submitted successfully on Indeed")')
    ] * 6)

    # 4.2 Gallery & Photos Multi-Step
    gal_query = "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp"
    g_goal = "Find recent photo in Gallery Camera album and share via WhatsApp"
    g_plan = "1. Open Gallery 2. Go to Albums 3. Open Camera album 4. Select recent photo 5. Tap Share 6. Choose WhatsApp"

    dataset.extend([
        make_sample("black_learning_ui", gal_query, None, g_goal, g_plan, "Gallery is not open. Launching Samsung Gallery.", 'OPEN_APP("Gallery")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] Tab "Pictures"\n[2] Tab "Albums"\n[3] Tab "Stories"', g_goal, g_plan, "On Pictures tab. I will tap Albums to locate the Camera album.", 'CLICK("Albums")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] TextView "Albums"\n[2] View "Camera (142)"\n[3] View "Screenshots (35)"', g_goal, g_plan, "In Albums list. I see Camera album. I will tap Camera to view photos.", 'CLICK("Camera (142)")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] View "Recent Photo - Today 11:20 AM"\n[2] Button "Select"', g_goal, g_plan, "Camera album opened. I will tap the latest photo to view it.", 'CLICK("Recent Photo - Today 11:20 AM")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] Button "Share"\n[2] Button "Edit"\n[3] Button "Delete"', g_goal, g_plan, "Photo viewer active. I will tap Share to open the share sheet.", 'CLICK("Share")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: android]\n[1] TextView "Share via"\n[2] View "WhatsApp"\n[3] View "Gmail"', g_goal, g_plan, "Share sheet is open. I will tap WhatsApp to share the photo.", 'CLICK("WhatsApp")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Send to..."\n[2] TextView "Sister Maryam"', g_goal, "Completed", "Photo forwarded to WhatsApp target. Task completed.", 'DONE("Recent photo shared to WhatsApp successfully")')
    ] * 6)

    # 4.3 Gmail Compose & Reminder Multi-Step
    gmail_query = "Send an email reminder to team@example.com about tomorrow 10 AM Sprint Planning"
    m_goal = "Send email reminder about Sprint Planning to team@example.com"
    m_plan = "1. Open Gmail 2. Tap Compose 3. Enter recipient 4. Enter subject 5. Enter body text 6. Send"

    dataset.extend([
        make_sample("black_learning_ui", gmail_query, None, m_goal, m_plan, "Gmail is not open. Launching Gmail.", 'OPEN_APP("Gmail")'),
        make_sample("black_learning_ui", gmail_query, '[SCREEN: com.google.android.gm]\n[1] EditText "Search in emails"\n[2] Button "Compose"', m_goal, m_plan, "Inbox is displayed. I will tap Compose to draft a new email.", 'CLICK("Compose")'),
        make_sample("black_learning_ui", gmail_query, '[SCREEN: com.google.android.gm]\n[1] EditText "To" [Input]\n[2] EditText "Subject" [Input]', m_goal, m_plan, "Compose screen open. Entering recipient into 'To'.", 'TYPE("To", "team@example.com")'),
        make_sample("black_learning_ui", gmail_query, '[SCREEN: com.google.android.gm]\n[1] TextView "team@example.com"\n[2] EditText "Subject" [Input]', m_goal, m_plan, "Recipient set. Now entering the reminder subject.", 'TYPE("Subject", "Reminder: Sprint Planning Tomorrow at 10 AM")'),
        make_sample("black_learning_ui", gmail_query, '[SCREEN: com.google.android.gm]\n[1] EditText "Subject: Reminder: Sprint Planning..."\n[2] EditText "Compose email" [Input]', m_goal, m_plan, "Entering reminder message in body.", 'TYPE("Compose email", "Hi team, please prepare your backlog updates for tomorrow.")'),
        make_sample("black_learning_ui", gmail_query, '[SCREEN: com.google.android.gm]\n[1] Button "Send"\n[2] Button "More options"', m_goal, m_plan, "Email draft complete. Tapping Send.", 'CLICK("Send")'),
        make_sample("black_learning_ui", gmail_query, '[SCREEN: com.google.android.gm]\n[1] TextView "Sending message..."', m_goal, "Completed", "Reminder email sent. Task completed.", 'DONE("Sprint planning reminder email sent to team@example.com")')
    ] * 6)

    # 4.4 Clock & Alarms
    clk_query = "Set alarm for 07:15 AM called Gym Workout in Clock"
    cl_goal = "Set 07:15 AM alarm labeled Gym Workout in Clock app"
    cl_plan = "1. Open Clock 2. Tap Add alarm 3. Set time 07:15 4. Set label Gym Workout 5. Save"

    dataset.extend([
        make_sample("black_learning_ui", clk_query, None, cl_goal, cl_plan, "Clock is not open. Launching Clock.", 'OPEN_APP("Clock")'),
        make_sample("black_learning_ui", clk_query, '[SCREEN: com.sec.android.app.clockpackage]\n[1] Tab "Alarm"\n[2] Button "Add alarm"', cl_goal, cl_plan, "Clock open. Tapping Add alarm.", 'CLICK("Add alarm")'),
        make_sample("black_learning_ui", clk_query, '[SCREEN: com.sec.android.app.clockpackage]\n[1] EditText "Time" [Input]\n[2] Button "Save"', cl_goal, cl_plan, "Setting alarm time to 07:15.", 'TYPE("Time", "07:15")'),
        make_sample("black_learning_ui", clk_query, '[SCREEN: com.sec.android.app.clockpackage]\n[1] TextView "07:15 AM"\n[2] EditText "Alarm name" [Input]', cl_goal, cl_plan, "Setting alarm name to Gym Workout.", 'TYPE("Alarm name", "Gym Workout")'),
        make_sample("black_learning_ui", clk_query, '[SCREEN: com.sec.android.app.clockpackage]\n[1] EditText "Gym Workout"\n[2] Button "Save"', cl_goal, cl_plan, "Saving configured alarm.", 'CLICK("Save")'),
        make_sample("black_learning_ui", clk_query, '[SCREEN: com.sec.android.app.clockpackage]\n[1] Switch "Gym Workout 07:15 AM"', cl_goal, "Completed", "Alarm saved and active.", 'DONE("Alarm set for 07:15 AM called Gym Workout")')
    ] * 6)

    # =========================================================================
    # CATEGORY 5: SPONTANEOUS OBSTACLE RECOVERY & NEGATIVE FEEDBACK
    # =========================================================================
    obstacle_samples = [
        # Cookie overlay
        make_sample(
            "obstacle_recovery",
            "Open chrome, navigate to indeed jobs, and apply for AI Engineer position",
            '[SCREEN: com.android.chrome]\n[1] TextView "Cookie Preferences"\n[2] TextView "We use cookies to enhance your experience"\n[3] Button "Accept all cookies"\n[4] Button "Reject all"',
            "Open Chrome and apply for AI Engineer position on Indeed",
            "1. Clear blocking cookie modal 2. Fill application form 3. Submit",
            "A cookie overlay is blocking the form. I will tap 'Accept all cookies' to clear it.",
            'CLICK("Accept all cookies")'
        ),
        # Storage permission
        make_sample(
            "obstacle_recovery",
            "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp",
            '[SCREEN: com.google.android.permissioncontroller]\n[1] TextView "Allow Gallery to access photos and videos on this device?"\n[2] Button "Allow"\n[3] Button "Don\'t allow"',
            "Find recent photo in Gallery Camera album and share via WhatsApp",
            "1. Grant storage permission 2. Open Camera album 3. Share photo",
            "A system permission dialog is requesting storage access. I will tap 'Allow'.",
            'CLICK("Allow")'
        ),
        # Negative feedback recovery
        make_sample(
            "obstacle_recovery",
            "Open chatgpt and write a message about HyperGQE",
            '[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[5] View "Ask ChatGPT"\n[6] View "Attachment"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]',
            "Open ChatGPT and compose message about HyperGQE",
            "1. Bypass unresponsive Menu 2. Type directly into Ask ChatGPT 3. Send",
            "Previous click on Menu had no effect. I will type directly into 'Ask ChatGPT'.",
            'TYPE("Ask ChatGPT", "Explain HyperGQE architecture")'
        ),
        # Off-screen scrolling
        make_sample(
            "obstacle_recovery",
            "Complete checkout and place order",
            '[SCREEN: com.amazon.mShop.android.shopping]\n[1] TextView "Cart (1 item)"\n[2] TextView "Total: $79.99"',
            "Complete checkout and place order",
            "1. Scroll down to reveal Place Order button 2. Click Place Order",
            "The order button is below the visible viewport. Scrolling down to reveal it.",
            'SCROLL("DOWN")'
        ),
        # Update modal dismiss
        make_sample(
            "obstacle_recovery",
            "Send message on WhatsApp to Sister Maryam",
            '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Update WhatsApp"\n[2] Button "Later"\n[3] Button "Update"',
            "Send WhatsApp message to Sister Maryam",
            "1. Dismiss update dialog 2. Open chat 3. Send message",
            "An update prompt is intercepting the screen. Tapping 'Later' to bypass.",
            'CLICK("Later")'
        )
    ]
    for obs in obstacle_samples:
        for _ in range(12):
            dataset.append(obs)

    random.shuffle(dataset)
    print(f"Total Balanced Multi-Task Dataset Size: {len(dataset)} samples", flush=True)

    # Split 15% for validation and PPL benchmarking
    split_idx = int(len(dataset) * 0.85)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item) + "\n")
    print(f"Train samples: {len(train_data)} | Validation samples: {len(val_data)} (saved to {VAL_FILE})", flush=True)

    return train_data, val_data

def evaluate_val_ppl(model, tokenizer, val_data, device):
    model.eval()
    cat_stats = {}
    with torch.no_grad():
        for item in val_data:
            cat = item["category"]
            full = item["prompt"] + item["target"]
            full_ids = tokenizer.encode(full).ids
            prompt_ids = tokenizer.encode(item["prompt"]).ids

            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            p_len = len(prompt_ids) - 1
            labels[0, :p_len] = -100

            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100, reduction="sum").item()

            ntoks = (labels != -100).sum().item()
            if cat not in cat_stats:
                cat_stats[cat] = [0.0, 0]
            cat_stats[cat][0] += nll
            cat_stats[cat][1] += ntoks

    results = {}
    for cat, (total_nll, count) in cat_stats.items():
        if count > 0:
            avg_nll = total_nll / count
            ppl = math.exp(min(avg_nll, 20.0))
            results[cat] = {"nll": avg_nll, "ppl": ppl, "tokens": count}
    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING SWEET-SPOT MULTI-TASK & BLACK LEARNING AGENT ON {device} ===", flush=True)
    t0 = time.time()

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.float32)
    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=16, lora_alpha=32).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=16, lora_alpha=32).to(device))

    tokenizer = Tokenizer.from_file(TOK_PATH)
    train_data, val_data = build_sweetspot_dataset()

    print("\n--- Evaluating Initial Pre-Training Validation Perplexity ---", flush=True)
    init_ppl = evaluate_val_ppl(model, tokenizer, val_data, device)
    for cat, m in init_ppl.items():
        print(f"  * {cat.upper():20s} | NLL: {m['nll']:.4f} | Perplexity (PPL): {m['ppl']:.2f}", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    STEPS = 100
    BATCH_SIZE = 4
    model.train()

    print(f"\n--- Starting Ampere FP16 Mixed Precision LoRA Training ({STEPS} Steps) ---", flush=True)
    for step in range(1, STEPS + 1):
        batch = random.sample(train_data, BATCH_SIZE)
        optimizer.zero_grad()
        loss_total = 0.0

        for item in batch:
            full = item["prompt"] + item["target"]
            full_ids = tokenizer.encode(full).ids
            prompt_ids = tokenizer.encode(item["prompt"]).ids

            input_ids = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([full_ids[1:]], dtype=torch.long, device=device)

            p_len = len(prompt_ids) - 1
            labels[0, :p_len] = -100

            with torch.cuda.amp.autocast(dtype=torch.float16):
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

        if step % 15 == 0 or step == 1 or step == STEPS:
            avg_loss = loss_total / len(batch)
            print(f"SweetSpot Step {step:3d}/{STEPS} | Train Loss: {avg_loss:.4f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    print("\n--- Evaluating Final Post-Training Validation Perplexity ---", flush=True)
    final_ppl = evaluate_val_ppl(model, tokenizer, val_data, device)
    for cat, m in final_ppl.items():
        print(f"  * {cat.upper():20s} | NLL: {m['nll']:.4f} | Perplexity (PPL): {m['ppl']:.2f}", flush=True)

    print("\nMerging LoRA weights into FP32 base...", flush=True)
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

    print(f"Saving merged checkpoint to {OUT_PT}...", flush=True)
    torch.save(model.state_dict(), OUT_PT)

    print(f"Serializing to binary EER at {OUT_EER}...", flush=True)
    serialize_checkpoint_to_eer(OUT_PT, OUT_EER, "FP16")
    print(f"[SUCCESS] Sweet-spot training & EER serialization complete in {time.time()-t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
