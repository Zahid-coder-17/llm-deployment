import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
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

BASE_PRISTINE_CKPT = os.path.join(BASE_DIR, "checkpoints", "maria_1b_20B_weights.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")

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

def build_pristine_dataset():
    random.seed(42)
    dataset = []

    # =========================================================================
    # 1. PURE ENGLISH CONVERSATION, ADVICE & DEEP EXPLANATIONS (NO ACTION)
    # =========================================================================
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
            "Here are 3 pieces of practical advice:\n"
            "1. Focus on one critical priority every morning before responding to messages.\n"
            "2. Keep solutions as simple as possible—unnecessary complexity is the root of bugs and fragility.\n"
            "3. Welcome direct feedback and use it to calibrate your trajectory continuously."
        ),
        (
            "What is Maria-1B?",
            "Maria-1B is a 1-billion-parameter language model architecture specifically optimized for on-device executive autonomy. "
            "It features grouped query attention (GQA), efficient KV caching, and sub-second latency on mobile ARM NEON and desktop GPUs."
        ),
        (
            "Explain how on-device AI preserves privacy",
            "On-device AI executes all neural network computations locally on the device silicon (CPU, GPU, or NPU). "
            "Because personal prompts, notifications, and visible screen data never leave the phone's memory, sensitive user information is completely protected from cloud interception or external data logging."
        ),
        (
            "What is the difference between supervised fine-tuning and reinforcement learning?",
            "Supervised fine-tuning trains a model to imitate high-quality target demonstrations by predicting the next token. "
            "Reinforcement learning optimizes model decision-making through trial-and-error rewards based on environmental outcomes, enabling the model to explore and self-correct."
        )
    ]
    for q, ans in chat_items:
        for _ in range(16):
            dataset.append(make_sample("chat", q, None, answer_text=ans))

    # =========================================================================
    # 2. PROFESSIONAL REPLY DRAFTING & SUMMARIZATION (NO ACTION)
    # =========================================================================
    draft_and_summary_items = [
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
        )
    ]
    for q, ans in draft_and_summary_items:
        for _ in range(16):
            dataset.append(make_sample("draft_summary", q, None, answer_text=ans))

    # =========================================================================
    # 3. CORE DEVICE CONTROLS & FAST ONE-TURN ACTIONS
    # =========================================================================
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
        ("Open WhatsApp", "Launch WhatsApp application", "1. Open WhatsApp", "User requested WhatsApp. Launching application.", 'OPEN_APP("WhatsApp")'),
        ("Open Chrome", "Launch Chrome browser", "1. Open Chrome", "User requested Chrome. Launching browser.", 'OPEN_APP("Chrome")'),
        ("Open Gallery", "Launch Gallery application", "1. Open Gallery", "User requested Gallery. Launching application.", 'OPEN_APP("Gallery")'),
        ("Open Calculator", "Launch Calculator", "1. Open Calculator", "User requested Calculator. Launching application.", 'OPEN_APP("Calculator")'),
    ]
    for q, g, p, ar, tc in core_actions:
        for _ in range(16):
            dataset.append(make_sample("core_action", q, None, g, p, ar, tc))

    # =========================================================================
    # 4. BLACK LEARNING MULTI-STEP TRAJECTORIES
    # =========================================================================
    # 4.1 Chrome Job Application
    cj_query = "Open chrome, navigate to indeed jobs, and apply for AI Engineer position"
    c_goal = "Open Chrome and apply for AI Engineer position on Indeed"
    c_plan = "1. Open Chrome 2. Navigate to indeed.com/jobs 3. Select AI Engineer role 4. Fill application form 5. Submit"

    dataset.extend([
        make_sample("black_learning_ui", cj_query, None, c_goal, c_plan, "Chrome is not open. Launching Chrome.", 'OPEN_APP("Chrome")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] View "Search or type URL"\n[2] Button "Home"', c_goal, c_plan, "Address bar visible. Tapping to enter URL.", 'CLICK("Search or type URL")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] EditText "Search or type URL" [Input]', c_goal, c_plan, "Typing indeed.com/jobs into URL bar.", 'TYPE("Search or type URL", "indeed.com/jobs")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] TextView "Senior AI Engineer - DeepMind"\n[2] Button "Apply Now"', c_goal, c_plan, "I see 'Apply Now' for Senior AI Engineer. Tapping Apply Now.", 'CLICK("Apply Now")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] EditText "Full Name" [Input]\n[2] EditText "Email Address" [Input]', c_goal, c_plan, "Application form is open. Entering full name.", 'TYPE("Full Name", "Alex Chen")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] EditText "Alex Chen"\n[2] EditText "Email Address" [Input]', c_goal, c_plan, "Entering contact email into Email Address.", 'TYPE("Email Address", "alex.chen@embeddra.ai")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] Button "Submit Application"', c_goal, c_plan, "Fields are populated. Submitting application.", 'CLICK("Submit Application")'),
        make_sample("black_learning_ui", cj_query, '[SCREEN: com.android.chrome]\n[1] TextView "Application Submitted Successfully!"', c_goal, "Completed", "Job application confirmed by portal. Task complete.", 'DONE("AI Engineer job application submitted successfully on Indeed")')
    ] * 6)

    # 4.2 Gallery Photo Sharing
    gal_query = "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp"
    g_goal = "Find recent photo in Gallery Camera album and share via WhatsApp"
    g_plan = "1. Open Gallery 2. Go to Albums 3. Open Camera album 4. Select recent photo 5. Tap Share 6. Choose WhatsApp"

    dataset.extend([
        make_sample("black_learning_ui", gal_query, None, g_goal, g_plan, "Gallery is not open. Launching Gallery.", 'OPEN_APP("Gallery")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] Tab "Pictures"\n[2] Tab "Albums"', g_goal, g_plan, "Tapping Albums tab to locate Camera album.", 'CLICK("Albums")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] View "Camera (142)"\n[2] View "Screenshots"', g_goal, g_plan, "Camera album located. Tapping Camera.", 'CLICK("Camera (142)")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] View "Recent Photo - Today 11:20 AM"', g_goal, g_plan, "Opening latest photo.", 'CLICK("Recent Photo - Today 11:20 AM")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.sec.android.gallery3d]\n[1] Button "Share"\n[2] Button "Edit"', g_goal, g_plan, "Photo open. Tapping Share.", 'CLICK("Share")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: android]\n[1] View "WhatsApp"\n[2] View "Gmail"', g_goal, g_plan, "Choosing WhatsApp from share sheet.", 'CLICK("WhatsApp")'),
        make_sample("black_learning_ui", gal_query, '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Sister Maryam"', g_goal, "Completed", "Photo forwarded to WhatsApp. Task complete.", 'DONE("Recent photo shared to WhatsApp successfully")')
    ] * 6)

    # 4.3 ChatGPT Multi-Turn
    cg_query = "Open chatgpt and write a message about HyperGQE"
    cg_goal = "Open ChatGPT and compose message about HyperGQE"
    cg_plan = "1. Open ChatGPT 2. Enter prompt into Ask ChatGPT 3. Send message"

    dataset.extend([
        make_sample("black_learning_ui", cg_query, None, cg_goal, cg_plan, "ChatGPT is not open. Launching ChatGPT.", 'OPEN_APP("ChatGPT")'),
        make_sample("black_learning_ui", cg_query, '[SCREEN: com.openai.chatgpt]\n[1] View "Ask ChatGPT"\n[2] View "Menu"', cg_goal, cg_plan, "Visible Ask ChatGPT input found. Typing prompt.", 'TYPE("Ask ChatGPT", "Explain HyperGQE architecture")'),
        make_sample("black_learning_ui", cg_query, '[SCREEN: com.openai.chatgpt]\n[1] Button "Send message"', cg_goal, cg_plan, "Prompt is entered. Tapping Send message.", 'CLICK("Send message")')
    ] * 6)

    # =========================================================================
    # 5. SPONTANEOUS OBSTACLE RECOVERY & NEGATIVE FEEDBACK
    # =========================================================================
    dataset.extend([
        # Cookie overlay
        make_sample(
            "obstacle_recovery",
            "Open chrome, navigate to indeed jobs, and apply for AI Engineer position",
            '[SCREEN: com.android.chrome]\n[1] TextView "Cookie Preferences"\n[2] Button "Accept all cookies"\n[3] Button "Reject all"',
            "Open Chrome and apply for AI Engineer position on Indeed",
            "1. Clear blocking cookie modal 2. Fill application form 3. Submit",
            "A cookie overlay is blocking the form. Tapping 'Accept all cookies' to clear it.",
            'CLICK("Accept all cookies")'
        ),
        # Storage permission
        make_sample(
            "obstacle_recovery",
            "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp",
            '[SCREEN: com.google.android.permissioncontroller]\n[1] TextView "Allow Gallery to access photos and videos on this device?"\n[2] Button "Allow"\n[3] Button "Don\'t allow"',
            "Find recent photo in Gallery Camera album and share via WhatsApp",
            "1. Grant storage permission 2. Open Camera album 3. Share photo",
            "A system permission dialog is requesting storage access. Tapping 'Allow'.",
            'CLICK("Allow")'
        ),
        # Unresponsive button negative feedback
        make_sample(
            "obstacle_recovery",
            "Open chatgpt and write a message about HyperGQE",
            '[SCREEN: com.openai.chatgpt]\n[1] View "Ask ChatGPT"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]',
            "Open ChatGPT and compose message about HyperGQE",
            "1. Bypass unresponsive Menu 2. Type directly into Ask ChatGPT 3. Send",
            "Previous click on Menu had no effect. Typing directly into visible 'Ask ChatGPT'.",
            'TYPE("Ask ChatGPT", "Explain HyperGQE architecture")'
        ),
        # Off-screen scrolling
        make_sample(
            "obstacle_recovery",
            "Complete checkout and place order",
            '[SCREEN: com.amazon.mShop.android.shopping]\n[1] TextView "Cart (1 item)"\n[2] TextView "Total: $79.99"',
            "Complete checkout and place order",
            "1. Scroll down to reveal Place Order button 2. Click Place Order",
            "The order button is below visible screen. Scrolling down to reveal it.",
            'SCROLL("DOWN")'
        ),
        # Update dialog
        make_sample(
            "obstacle_recovery",
            "Send message on WhatsApp to Sister Maryam",
            '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Update WhatsApp"\n[2] Button "Later"\n[3] Button "Update"',
            "Send WhatsApp message to Sister Maryam",
            "1. Dismiss update dialog 2. Open chat 3. Send message",
            "An update prompt is intercepting the screen. Tapping 'Later' to bypass.",
            'CLICK("Later")'
        )
    ] * 8)

    random.shuffle(dataset)
    print(f"Total Pristine Master Dataset Size: {len(dataset)} samples", flush=True)
    return dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING CLEAN PRISTINE MASTER YORI ON {device} ===", flush=True)
    t0 = time.time()

    print(f"Loading pristine pre-trained base from {BASE_PRISTINE_CKPT}...", flush=True)
    pt = torch.load(BASE_PRISTINE_CKPT, map_location="cpu", weights_only=False)
    cfg = MariaConfig()
    model = MariaForCausalLM(cfg).to(device=device, dtype=torch.float32)
    model.load_state_dict(pt["model_state_dict"], strict=True)
    print("[OK] Pristine base model successfully loaded into FP32!", flush=True)

    for param in model.parameters():
        param.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=16, lora_alpha=32).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=16, lora_alpha=32).to(device))

    tokenizer = Tokenizer.from_file(TOK_PATH)
    dataset = build_pristine_dataset()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=2.2e-5, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda')

    STEPS = 110
    BATCH_SIZE = 4
    model.train()

    print(f"\n--- Starting Clean Balanced Training ({STEPS} Steps) ---", flush=True)
    for step in range(1, STEPS + 1):
        batch = random.sample(dataset, BATCH_SIZE)
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

        if step % 15 == 0 or step == 1 or step == STEPS:
            avg_loss = loss_total / len(batch)
            print(f"Pristine Step {step:3d}/{STEPS} | Loss: {avg_loss:.4f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    print("\nMerging LoRA weights cleanly into pristine FP32 base...", flush=True)
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
    print(f"[SUCCESS] Clean training and EER export complete in {time.time()-t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
