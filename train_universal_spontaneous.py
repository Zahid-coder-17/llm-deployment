import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import math
import time
import types
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

def make_sample(query: str, screen: str, goal: str, plan: str, action_reason: str, tool_call: str = None, answer_text: str = None):
    prompt = f"{SYSTEM_CONTRACT}<user>{query}</user>\n"
    if screen:
        prompt += f"<screen>\n{screen.strip()}\n</screen>\n"
    prompt += "<thought>\n"

    if answer_text:
        # Conversational / advice / general answer (NO ACTION)
        target = f"{answer_text.strip()}\n</thought>"
    else:
        # Structured executive action turn
        target = (
            f"🎯 Goal: {goal.strip()}\n"
            f"📋 Plan: {plan.strip()}\n"
            f"⚡ Immediate Action: {action_reason.strip()}\n"
            f"</thought>\n"
            f"<action>{tool_call.strip()}</action>"
        )
    return {"prompt": prompt, "target": target}

def build_universal_dataset():
    random.seed(42)
    samples = []

    # =========================================================================
    # 1. CHROME / WEB AUTOMATION & JOB APPLICATION
    # =========================================================================
    chrome_query = "Open chrome, navigate to indeed jobs, and apply for AI Engineer position"
    cg = "Open Chrome and apply for AI Engineer position on Indeed"
    cp = "1. Open Chrome 2. Navigate to indeed.com/jobs 3. Select AI Engineer role 4. Fill application form (Name, Email, Resume) 5. Submit"

    s_c1 = make_sample(chrome_query, None, cg, cp, "Chrome is not currently open. I must launch Chrome first.", 'OPEN_APP("Chrome")')
    s_c2 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] View "Search or type URL"\n[2] Button "Home"\n[3] Button "Tabs (2)"',
        cg, cp,
        "Chrome is active. I see the address bar 'Search or type URL'. I will tap it to enter the jobs website.",
        'CLICK("Search or type URL")'
    )
    s_c3 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] EditText "Search or type URL" [Input]\n[2] Button "Voice search"',
        cg, cp,
        "The URL bar is focused. I will enter indeed.com/jobs to navigate to the job portal.",
        'TYPE("Search or type URL", "indeed.com/jobs")'
    )
    s_c4 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] TextView "Indeed Job Search"\n[2] TextView "Senior AI Engineer - DeepMind"\n[3] Button "Apply Now"\n[4] TextView "Junior Web Developer"',
        cg, cp,
        "Job listings loaded. I see 'Senior AI Engineer - DeepMind' with an 'Apply Now' button. I will tap Apply Now.",
        'CLICK("Apply Now")'
    )
    # Spontaneous obstacle recovery: Cookie overlay blocking the form
    s_c5_obstacle = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] TextView "Cookie Preferences"\n[2] TextView "We use cookies to enhance your experience"\n[3] Button "Accept all cookies"\n[4] Button "Reject all"',
        cg,
        "1. Clear blocking cookie modal 2. Fill application form 3. Submit",
        "A cookie consent overlay is blocking the application form. I will tap 'Accept all cookies' to clear the obstacle.",
        'CLICK("Accept all cookies")'
    )
    s_c6 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] TextView "Apply to DeepMind"\n[2] EditText "Full Name" [Input]\n[3] EditText "Email Address" [Input]\n[4] Button "Upload Resume"',
        cg, cp,
        "The application form is visible. I will enter the applicant name into the 'Full Name' field.",
        'TYPE("Full Name", "Alex Chen")'
    )
    s_c7 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] EditText "Alex Chen"\n[2] EditText "Email Address" [Input]\n[3] Button "Upload Resume"\n[4] Button "Submit Application"',
        cg, cp,
        "Name is entered. Now I will enter the contact email into 'Email Address'.",
        'TYPE("Email Address", "alex.chen@embeddra.ai")'
    )
    s_c8 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] EditText "Alex Chen"\n[2] EditText "alex.chen@embeddra.ai"\n[3] Button "Upload Resume"\n[4] Button "Submit Application"',
        cg, cp,
        "Personal details are filled. Now I will tap 'Upload Resume' to attach the CV.",
        'CLICK("Upload Resume")'
    )
    s_c9 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] TextView "alex_chen_cv.pdf attached"\n[2] CheckBox "I agree to terms"\n[3] Button "Submit Application"',
        cg, cp,
        "Resume is attached. I will tap 'Submit Application' to finalize and send the job application.",
        'CLICK("Submit Application")'
    )
    s_c10 = make_sample(
        chrome_query,
        '[SCREEN: com.android.chrome]\n[1] TextView "Application Submitted!"\n[2] TextView "We have received your application for Senior AI Engineer."',
        cg, "Completed",
        "The job application has been successfully submitted and confirmed by the portal. Task complete.",
        'DONE("AI Engineer job application submitted successfully on Indeed")'
    )

    for _ in range(12):
        samples.extend([s_c1, s_c2, s_c3, s_c4, s_c5_obstacle, s_c6, s_c7, s_c8, s_c9, s_c10])

    # =========================================================================
    # 2. GALLERY / PHOTOS WORKFLOWS & SPONTANEOUS RECOVERY
    # =========================================================================
    gal_query = "Go to Gallery, locate Camera album, select recent photo, and share via WhatsApp"
    gg = "Find recent photo in Gallery Camera album and share via WhatsApp"
    gp = "1. Open Gallery 2. Go to Albums 3. Open Camera album 4. Select recent photo 5. Tap Share 6. Choose WhatsApp"

    s_g1 = make_sample(gal_query, None, gg, gp, "Gallery is not open. I will launch Samsung Gallery.", 'OPEN_APP("Gallery")')
    s_g2 = make_sample(
        gal_query,
        '[SCREEN: com.sec.android.gallery3d]\n[1] Tab "Pictures"\n[2] Tab "Albums"\n[3] Tab "Stories"\n[4] Button "Search"',
        gg, gp,
        "Currently on Pictures tab. I will tap the Albums tab to find the Camera album.",
        'CLICK("Albums")'
    )
    # Spontaneous obstacle recovery: Storage permission dialog
    s_g_perm = make_sample(
        gal_query,
        '[SCREEN: com.google.android.permissioncontroller]\n[1] TextView "Allow Gallery to access photos and videos on this device?"\n[2] Button "Allow"\n[3] Button "Don\'t allow"',
        gg,
        "1. Grant required storage permission 2. Open Camera album 3. Share photo",
        "A system permission dialog is requesting photo access. I will tap 'Allow' so Gallery can display photos.",
        'CLICK("Allow")'
    )
    s_g3 = make_sample(
        gal_query,
        '[SCREEN: com.sec.android.gallery3d]\n[1] TextView "Albums"\n[2] View "Camera (142)"\n[3] View "Screenshots (35)"\n[4] View "Downloads (18)"',
        gg, gp,
        "In Albums view. I see the Camera album listed. I will tap 'Camera (142)' to view its photos.",
        'CLICK("Camera (142)")'
    )
    s_g4 = make_sample(
        gal_query,
        '[SCREEN: com.sec.android.gallery3d]\n[1] TextView "Camera"\n[2] View "Recent Photo - Today 11:20 AM"\n[3] View "Photo - Yesterday"\n[4] Button "Select"',
        gg, gp,
        "Camera album opened. I see 'Recent Photo - Today 11:20 AM'. I will tap it to open photo details.",
        'CLICK("Recent Photo - Today 11:20 AM")'
    )
    s_g5 = make_sample(
        gal_query,
        '[SCREEN: com.sec.android.gallery3d]\n[1] ImageButton "Back"\n[2] Button "Share"\n[3] Button "Edit"\n[4] Button "Delete"\n[5] Button "More"',
        gg, gp,
        "Photo viewer is open. I see the Share action on the toolbar. I will tap 'Share'.",
        'CLICK("Share")'
    )
    s_g6 = make_sample(
        gal_query,
        '[SCREEN: android]\n[1] TextView "Share via"\n[2] View "Quick Share"\n[3] View "WhatsApp"\n[4] View "Gmail"\n[5] View "Messages"',
        gg, gp,
        "Share sheet is visible. I will tap 'WhatsApp' to share the selected photo.",
        'CLICK("WhatsApp")'
    )
    s_g7 = make_sample(
        gal_query,
        '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Send to..."\n[2] TextView "Sister Maryam"\n[3] TextView "Family Group"',
        gg, "Completed",
        "WhatsApp share target opened. Photo is ready to send. Task completed.",
        'DONE("Recent photo shared to WhatsApp successfully")'
    )

    for _ in range(12):
        samples.extend([s_g1, s_g2, s_g_perm, s_g3, s_g4, s_g5, s_g6, s_g7])

    # =========================================================================
    # 3. GMAIL / EMAIL REMINDERS & SEARCH WORKFLOWS
    # =========================================================================
    mail_rem_query = "Send an email reminder to team@example.com about tomorrow 10 AM Sprint Planning"
    mg = "Send email reminder about Sprint Planning to team@example.com"
    mp = "1. Open Gmail 2. Tap Compose 3. Enter recipient 4. Enter subject 5. Enter body text 6. Send email"

    s_m1 = make_sample(mail_rem_query, None, mg, mp, "Gmail is not open. I will launch Gmail.", 'OPEN_APP("Gmail")')
    s_m2 = make_sample(
        mail_rem_query,
        '[SCREEN: com.google.android.gm]\n[1] EditText "Search in emails"\n[2] TextView "Primary"\n[3] Button "Compose"\n[4] TextView "Team Updates"',
        mg, mp,
        "Gmail inbox is displayed. I see the Compose button. I will tap 'Compose' to write a new email.",
        'CLICK("Compose")'
    )
    s_m3 = make_sample(
        mail_rem_query,
        '[SCREEN: com.google.android.gm]\n[1] TextView "Compose"\n[2] EditText "To" [Input]\n[3] EditText "Subject" [Input]\n[4] EditText "Compose email" [Input]',
        mg, mp,
        "Compose screen is open. I will enter team@example.com into the 'To' field.",
        'TYPE("To", "team@example.com")'
    )
    s_m4 = make_sample(
        mail_rem_query,
        '[SCREEN: com.google.android.gm]\n[1] TextView "team@example.com"\n[2] EditText "Subject" [Input]\n[3] EditText "Compose email" [Input]\n[4] Button "Send"',
        mg, mp,
        "Recipient is set. Now I will enter the reminder subject into 'Subject'.",
        'TYPE("Subject", "Reminder: Sprint Planning Tomorrow at 10 AM")'
    )
    s_m5 = make_sample(
        mail_rem_query,
        '[SCREEN: com.google.android.gm]\n[1] EditText "Subject: Reminder: Sprint Planning Tomorrow at 10 AM"\n[2] EditText "Compose email" [Input]\n[3] Button "Send"',
        mg, mp,
        "Subject is set. Now I will write the reminder details in 'Compose email'.",
        'TYPE("Compose email", "Hi team, please be prepared with your backlog tickets for our 10 AM sprint planning tomorrow.")'
    )
    s_m6 = make_sample(
        mail_rem_query,
        '[SCREEN: com.google.android.gm]\n[1] EditText "Compose email: Hi team..."\n[2] Button "Send"\n[3] Button "More options"',
        mg, mp,
        "Email draft is ready. I will tap 'Send' to deliver the reminder.",
        'CLICK("Send")'
    )
    s_m7 = make_sample(
        mail_rem_query,
        '[SCREEN: com.google.android.gm]\n[1] TextView "Sending message..."\n[2] TextView "Primary Inbox"',
        mg, "Completed",
        "The reminder email has been sent successfully. Task completed.",
        'DONE("Sprint planning reminder email sent to team@example.com")'
    )

    # Search email workflow
    search_mail_query = "Search emails for invoice and check the payment due date"
    smg = "Search emails for invoice and check payment due date"
    smp = "1. Open Gmail 2. Search for invoice 3. Open top invoice email 4. Read due date"

    s_sm1 = make_sample(search_mail_query, None, smg, smp, "Gmail is not open. Launching Gmail.", 'OPEN_APP("Gmail")')
    s_sm2 = make_sample(
        search_mail_query,
        '[SCREEN: com.google.android.gm]\n[1] EditText "Search in emails" [Input]\n[2] Button "Compose"',
        smg, smp,
        "I will tap 'Search in emails' and enter 'invoice'.",
        'TYPE("Search in emails", "invoice")'
    )
    s_sm3 = make_sample(
        search_mail_query,
        '[SCREEN: com.google.android.gm]\n[1] TextView "Search results"\n[2] TextView "AWS Hosting Invoice #8492 - Due Nov 25 ($120.00)"\n[3] TextView "Old Invoice - Paid"',
        smg, smp,
        "Search results loaded. I see 'AWS Hosting Invoice #8492 - Due Nov 25 ($120.00)'. I will tap it to inspect.",
        'CLICK("AWS Hosting Invoice #8492 - Due Nov 25 ($120.00)")'
    )
    s_sm4 = make_sample(
        search_mail_query,
        '[SCREEN: com.google.android.gm]\n[1] TextView "AWS Hosting Invoice #8492"\n[2] TextView "Amount: $120.00 | Due Date: November 25, 2026"\n[3] Button "Reply"',
        smg, "Completed",
        "I have read the email content: The AWS Hosting Invoice #8492 amount is $120.00 and is due on November 25, 2026.",
        'DONE("AWS Hosting Invoice #8492 is due on November 25, 2026 for $120.00")'
    )

    for _ in range(12):
        samples.extend([s_m1, s_m2, s_m3, s_m4, s_m5, s_m6, s_m7, s_sm1, s_sm2, s_sm3, s_sm4])

    # =========================================================================
    # 4. ALARMS & REMINDERS (DIRECT & MULTI-STEP)
    # =========================================================================
    # Direct fast alarm
    s_al1 = make_sample(
        "Set an alarm for 6:30 AM",
        None,
        "Set device alarm for 06:30 AM",
        "1. Trigger SET_ALARM tool",
        "User requested a simple 6:30 AM alarm. I will invoke SET_ALARM directly.",
        'SET_ALARM("06:30")'
    )
    s_al2 = make_sample(
        "Wake me up at 7:00 tomorrow morning",
        None,
        "Set device alarm for 07:00 AM",
        "1. Trigger SET_ALARM tool",
        "User wants an alarm for 7:00 AM. I will invoke SET_ALARM directly.",
        'SET_ALARM("07:00")'
    )

    # Complex named alarm via Clock app
    named_alarm_q = "Set alarm for 07:15 AM called Gym Workout in Clock"
    nag = "Set 07:15 AM alarm labeled Gym Workout in Clock app"
    nap = "1. Open Clock 2. Tap Alarm tab 3. Tap Add alarm 4. Set time 07:15 5. Set label Gym Workout 6. Save"

    s_na1 = make_sample(named_alarm_q, None, nag, nap, "Clock is not open. Launching Clock app.", 'OPEN_APP("Clock")')
    s_na2 = make_sample(
        named_alarm_q,
        '[SCREEN: com.sec.android.app.clockpackage]\n[1] Tab "Alarm"\n[2] Tab "World clock"\n[3] Tab "Stopwatch"\n[4] Button "Add alarm"',
        nag, nap,
        "Clock app opened. I see 'Add alarm'. I will tap 'Add alarm' to create a new labeled alarm.",
        'CLICK("Add alarm")'
    )
    s_na3 = make_sample(
        named_alarm_q,
        '[SCREEN: com.sec.android.app.clockpackage]\n[1] EditText "Time" [Input]\n[2] EditText "Alarm name" [Input]\n[3] Button "Save"',
        nag, nap,
        "Setting time to 07:15.",
        'TYPE("Time", "07:15")'
    )
    s_na4 = make_sample(
        named_alarm_q,
        '[SCREEN: com.sec.android.app.clockpackage]\n[1] TextView "07:15 AM"\n[2] EditText "Alarm name" [Input]\n[3] Button "Save"',
        nag, nap,
        "Time is configured. Now typing the label 'Gym Workout' into 'Alarm name'.",
        'TYPE("Alarm name", "Gym Workout")'
    )
    s_na5 = make_sample(
        named_alarm_q,
        '[SCREEN: com.sec.android.app.clockpackage]\n[1] TextView "07:15 AM"\n[2] EditText "Gym Workout"\n[3] Button "Save"',
        nag, nap,
        "Label is set. I will tap 'Save' to activate the alarm.",
        'CLICK("Save")'
    )
    s_na6 = make_sample(
        named_alarm_q,
        '[SCREEN: com.sec.android.app.clockpackage]\n[1] TextView "Alarm set for 13 hours 45 minutes from now"\n[2] Switch "Gym Workout 07:15 AM"',
        nag, "Completed",
        "Alarm successfully created and enabled for 07:15 AM.",
        'DONE("Alarm set for 07:15 AM called Gym Workout")'
    )

    # Calendar Reminder
    rem_q = "Set reminder for Dentist appointment tomorrow at 3 PM"
    rg = "Create reminder for Dentist appointment at 3 PM tomorrow"
    rp = "1. Open Tasks 2. Tap Add a task 3. Type title Dentist appointment 3 PM 4. Save"

    s_r1 = make_sample(rem_q, None, rg, rp, "Tasks app is not open. Launching Tasks.", 'OPEN_APP("Tasks")')
    s_r2 = make_sample(
        rem_q,
        '[SCREEN: com.google.android.apps.tasks]\n[1] TextView "My Tasks"\n[2] Button "Create new task" [Input]\n[3] Button "More"',
        rg, rp,
        "Tasks is open. I will tap 'Create new task' to create the reminder.",
        'CLICK("Create new task")'
    )
    s_r3 = make_sample(
        rem_q,
        '[SCREEN: com.google.android.apps.tasks]\n[1] EditText "New task" [Input]\n[2] Button "Save"\n[3] Button "Set date/time"',
        rg, rp,
        "I will enter the reminder text 'Dentist appointment tomorrow at 3 PM'.",
        'TYPE("New task", "Dentist appointment tomorrow at 3 PM")'
    )
    s_r4 = make_sample(
        rem_q,
        '[SCREEN: com.google.android.apps.tasks]\n[1] EditText "Dentist appointment tomorrow at 3 PM"\n[2] Button "Save"',
        rg, rp,
        "Title is entered. I will tap 'Save' to schedule the reminder.",
        'CLICK("Save")'
    )
    s_r5 = make_sample(
        rem_q,
        '[SCREEN: com.google.android.apps.tasks]\n[1] TextView "Dentist appointment tomorrow at 3 PM"\n[2] CheckBox "Mark completed"',
        rg, "Completed",
        "The reminder has been saved in Tasks.",
        'DONE("Reminder created for Dentist appointment tomorrow at 3 PM")'
    )

    for _ in range(12):
        samples.extend([s_al1, s_al2, s_na1, s_na2, s_na3, s_na4, s_na5, s_na6, s_r1, s_r2, s_r3, s_r4, s_r5])

    # =========================================================================
    # 5. SPONTANEOUS CLOSED-LOOP RECOVERY & NEGATIVE FEEDBACK
    # =========================================================================
    # Screen unchanged negative feedback recovery
    s_fb1 = make_sample(
        "Open chatgpt and write a message about HyperGQE",
        '[SCREEN: com.openai.chatgpt]\n[2] View "Menu"\n[5] View "Ask ChatGPT"\n[6] View "Attachment"\n[ENVIRONMENT FEEDBACK: Previous action \'CLICK("Menu")\' had no effect. Visible screen is unchanged. Choose a different element from visible list or alternate action.]',
        "Open ChatGPT and compose message about HyperGQE",
        "1. Bypass unresponsive Menu 2. Type directly into Ask ChatGPT 3. Send",
        "Previous click on Menu had no effect. I will adapt spontaneously and type directly into the visible 'Ask ChatGPT' field.",
        'TYPE("Ask ChatGPT", "Explain HyperGQE architecture")'
    )

    # Off-screen button requires scrolling
    s_fb2 = make_sample(
        "Complete checkout and place order",
        '[SCREEN: com.amazon.mShop.android.shopping]\n[1] TextView "Cart (1 item)"\n[2] TextView "Wireless Headphones - $79.99"\n[3] TextView "Shipping Address: 123 Main St"\n[4] TextView "Payment: Visa ending 4021"',
        "Complete checkout and place order",
        "1. Scroll down to reveal Place Order button 2. Click Place Order",
        "The 'Place Your Order' button is below the current visible viewport. I will scroll down to reveal it.",
        'SCROLL("DOWN")'
    )
    s_fb3 = make_sample(
        "Complete checkout and place order",
        '[SCREEN: com.amazon.mShop.android.shopping]\n[1] TextView "Order Total: $79.99"\n[2] Button "Place your order in USD"\n[3] TextView "Terms and Conditions"',
        "Complete checkout and place order",
        "1. Click Place your order",
        "The order button is now visible. I will tap 'Place your order in USD'.",
        'CLICK("Place your order in USD")'
    )

    # Permission obstacle recovery
    s_fb4 = make_sample(
        "Find nearest gas stations on Google Maps",
        '[SCREEN: com.google.android.apps.maps]\n[1] TextView "Allow Maps to access this device\'s location?"\n[2] Button "While using the app"\n[3] Button "Only this time"\n[4] Button "Don\'t allow"',
        "Find nearest gas stations on Google Maps",
        "1. Grant location permission 2. Search gas stations 3. Display nearest results",
        "Location permission dialog is blocking the map. I will grant location access by tapping 'While using the app'.",
        'CLICK("While using the app")'
    )

    # Update dialog dismiss obstacle recovery
    s_fb5 = make_sample(
        "Send message on WhatsApp to Sister Maryam",
        '[SCREEN: com.whatsapp.w4b]\n[1] TextView "Update WhatsApp"\n[2] TextView "A new version of WhatsApp is available."\n[3] Button "Later"\n[4] Button "Update"',
        "Send WhatsApp message to Sister Maryam",
        "1. Dismiss update dialog 2. Open chat with Sister Maryam 3. Send message",
        "An update dialog is intercepting the screen. I will dismiss it by tapping 'Later' to proceed with the message.",
        'CLICK("Later")'
    )

    for _ in range(12):
        samples.extend([s_fb1, s_fb2, s_fb3, s_fb4, s_fb5])

    # =========================================================================
    # 6. SYSTEM CONTROLS & SETTINGS
    # =========================================================================
    sys_samples = [
        ("Open Wi-Fi Settings", None, "Open Wi-Fi Settings", "1. Open Wi-Fi settings", "User requested Wi-Fi settings. Invoking OPEN_SETTINGS.", 'OPEN_SETTINGS("WIFI")'),
        ("Open Bluetooth settings", None, "Open Bluetooth settings", "1. Open Bluetooth settings", "User requested Bluetooth settings. Invoking OPEN_SETTINGS.", 'OPEN_SETTINGS("BLUETOOTH")'),
        ("Turn on flashlight", None, "Turn on device flashlight", "1. Turn on torch", "User wants flashlight turned on. Invoking TORCH(1).", 'TORCH(1)'),
        ("Turn off flashlight", None, "Turn off device flashlight", "1. Turn off torch", "User wants flashlight turned off. Invoking TORCH(0).", 'TORCH(0)'),
        ("Mute volume", None, "Mute device audio", "1. Mute volume", "User requested muting sound. Invoking VOLUME(MUTE).", 'VOLUME("MUTE")'),
        ("Volume up", None, "Increase device audio volume", "1. Increase volume", "User wants volume increased. Invoking VOLUME(UP).", 'VOLUME("UP")'),
    ]
    for q, s, g, p, ar, tc in sys_samples:
        for _ in range(12):
            samples.append(make_sample(q, s, g, p, ar, tc))

    # =========================================================================
    # 7. CONVERSATIONAL / CHIT-CHAT / ADVICE (ZERO ACTION)
    # =========================================================================
    chat_samples = [
        (
            "3 advices pls",
            "Here are 3 high-impact pieces of advice:\n"
            "1. Focus on compounding habits: small, consistent actions performed daily yield exponential long-term outcomes.\n"
            "2. Prioritize clarity over complexity: in engineering, communication, and decision-making, simplest solutions are most robust.\n"
            "3. Act and iterate rapidly: feedback from real-world execution always out-teaches passive overthinking."
        ),
        (
            "What is Maria-1B?",
            "Maria-1B is an efficient 1-billion-parameter language model architecture optimized for on-device executive autonomy. "
            "It features high-throughput KV caching, grouped query attention, and low latency on mobile ARM NEON and edge GPUs."
        ),
        (
            "How do I prepare for a machine learning engineer interview?",
            "To prepare effectively for an ML engineer interview:\n"
            "1. Master core fundamentals: linear algebra, gradient descent, transformer architectures, and loss formulations.\n"
            "2. Practice ML systems design: latency/throughput trade-offs, quantizations (INT8/FP16), KV caching, and distributed inference.\n"
            "3. Coding proficiency: implement custom attention layers, tensor operations, and data pipelines in PyTorch.\n"
            "4. Showcase end-to-end projects demonstrating deployed models solving real-world challenges."
        ),
        (
            "Can you explain on-device agentic closed loops?",
            "An on-device agentic closed loop is an autonomous execution cycle where the model observes the live environment state (e.g. accessibility node trees or DOM elements), "
            "reasons about its objective and plan, emits an executable action (click, type, scroll), and verifies the environment feedback to dynamically course-correct until the goal is achieved."
        )
    ]
    for q, ans in chat_samples:
        for _ in range(15):
            samples.append(make_sample(q, None, "", "", "", None, answer_text=ans))

    random.shuffle(samples)
    print(f"Total Universal Spontaneous Training Samples: {len(samples)}", flush=True)
    return samples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TRAINING UNIVERSAL SPONTANEOUS AGENTIC MODEL ON {device} ===", flush=True)
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
    samples = build_universal_dataset()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=0.01)

    STEPS = 160
    BATCH_SIZE = 4
    for step in range(1, STEPS + 1):
        batch = random.sample(samples, BATCH_SIZE)
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

            logits = model(input_ids)
            if isinstance(logits, tuple):
                logits = logits[0]

            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            (loss / len(batch)).backward()
            loss_total += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 20 == 0 or step == 1 or step == STEPS:
            avg_loss = loss_total / len(batch)
            print(f"Universal Step {step:3d}/{STEPS} | Loss: {avg_loss:.4f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    print("Merging Universal LoRA weights into FP32 base...", flush=True)
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
    print(f"[SUCCESS] Training, LoRA merge, and EER serialization finished in {time.time()-t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()
