import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import json
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

BASE_CKPT = os.path.join(BASE_DIR, "checkpoints", "maria_1b_20B_weights.pt")
LORA_RESUME = os.path.join(BASE_DIR, "lora_final_polish_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_master_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
LABEL_UI_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_label_ui_train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")
OUT_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_master_train.jsonl")


def build_dataset():
    random.seed(3001)
    items = []

    # =========================================================================
    # 1. Real Device Multi-Step YouTube Search & Play
    # =========================================================================
    yt_topics = [
        ("Agentic AI lecture", "Agentic AI Lecture - Full Course"),
        ("lo-fi beats", "Lofi Hip Hop Radio - Beats to Relax/Study to"),
        ("python tutorial", "Python for Beginners - Full Course"),
        ("transformer architecture", "Transformers Explained Clearly"),
        ("deep learning course", "Deep Learning Crash Course"),
        ("Andrej Karpathy lecture", "Intro to Large Language Models by Andrej Karpathy"),
    ]

    for q_txt, res_title in yt_topics:
        user_prompt = f"find {q_txt} and play it from youtube"
        # Turn 1: No screen
        items.append({
            "category": "tool_call",
            "prompt": f"<user>{user_prompt}</user>\n<thought>",
            "target": 'User wants to use YouTube. I will open YouTube first.</thought>\n<action>OPEN_APP("YouTube")</action>'
        })

        # Turn 2: YouTube Home Screen (real device layout with and without distractors)
        yt_home_1 = (
            '[SCREEN: com.google.android.youtube]\n'
            '[2] ImageView "YouTube"\n'
            '[3] Button "Notifications"\n'
            '[4] Button "Search"\n'
            '[5] Button "Home"\n'
            '[6] ViewGroup "Shorts"\n'
            '[8] Button "Action menu"'
        )
        items.append({
            "category": "ui_grounding",
            "prompt": f"<user>{user_prompt}</user>\n<screen>\n{yt_home_1}\n</screen>\n<thought>",
            "target": 'I will click "Search".</thought>\n<action>CLICK("Search")</action>'
        })

        # Turn 3: Search Box active with suggestions (REAL DEVICE FORMAT!)
        yt_search_box_real = (
            '[SCREEN: com.google.android.youtube]\n'
            '[1] Button "Voice search"\n'
            '[2] Button "Navigate up"\n'
            '[3] EditText "Search YouTube"\n'
            '[6] TextView "trending songs"\n'
            '[9] TextView "popular podcast"\n'
            '[12] TextView "news live"'
        )
        items.append({
            "category": "ui_grounding",
            "prompt": f"<user>{user_prompt}</user>\n<screen>\n{yt_search_box_real}\n</screen>\n<thought>",
            "target": f'I will type "{q_txt}" into "Search YouTube".</thought>\n<action>TYPE("Search YouTube", "{q_txt}")</action>'
        })

        # Turn 3 alt: with [Input] tag
        yt_search_box_input = (
            '[SCREEN: com.google.android.youtube]\n'
            '[1] ImageButton "Back"\n'
            '[2] EditText "Search YouTube" [Input]\n'
            '[3] Button "Voice"'
        )
        items.append({
            "category": "ui_grounding",
            "prompt": f"<user>{user_prompt}</user>\n<screen>\n{yt_search_box_input}\n</screen>\n<thought>",
            "target": f'I will type "{q_txt}" into "Search YouTube".</thought>\n<action>TYPE("Search YouTube", "{q_txt}")</action>'
        })

        # Turn 4: Search results visible
        yt_results_screen = (
            '[SCREEN: com.google.android.youtube]\n'
            f'[1] EditText "{q_txt}"\n'
            '[2] Button "Clear search"\n'
            '[3] Button "Filters"\n'
            f'[4] ViewGroup "{res_title}"\n'
            '[5] ViewGroup "Sponsored Ad"'
        )
        items.append({
            "category": "ui_grounding",
            "prompt": f"<user>{user_prompt}</user>\n<screen>\n{yt_results_screen}\n</screen>\n<thought>",
            "target": f'I will click "{res_title}".</thought>\n<action>CLICK("{res_title}")</action>'
        })

    # Multiply YouTube items so they are well learned
    yt_subset = list(items)
    for _ in range(6):
        items.extend(yt_subset)

    # =========================================================================
    # 2. Screen Summarization & QA (Strictly NO <action>)
    # =========================================================================
    screen_qa = [
        # YORI's own home screen
        (
            "summarize what is on my screen",
            '[SCREEN: com.embeddra.edge]\n[1] TextView "Y"\n[2] TextView "YORI"\n[3] TextView "OFFLINE"\n[4] Button "LLM Ready ✓ • Inspector: Active ✓"\n[5] Button "New Chat"\n[6] TextView "Hello! I\'m Yori. What task would you like me to execute on your device today?"',
            'You are on the YORI home screen. The local offline LLM and Inspector are active and ready to chat or execute tasks.</thought>'
        ),
        (
            "summarize what is on my screen",
            '[SCREEN: com.embeddra.edge]\n[1] TextView "Y"\n[2] TextView "YORI"\n[3] TextView "OFFLINE"\n[4] Button "LLM Ready ✓ • Inspector: Active ✓"\n[5] Button "New Chat"\n[6] TextView "Explain recursion simply"\n[10] TextView "take a screenshot"\n[13] TextView "📸 Took screenshot"',
            'You are in YORI chat. The screen displays previous messages discussing recursion and a successfully captured screenshot.</thought>'
        ),
        (
            "what is on my screen",
            '[SCREEN: com.embeddra.edge]\n[2] TextView "YORI"\n[3] TextView "OFFLINE"\n[5] Button "New Chat"\n[6] TextView "Hello! I\'m Yori."',
            'The screen shows the YORI assistant interface running offline on your device.</thought>'
        ),
        # WhatsApp chat
        (
            "summarize what is on my screen",
            '[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] TextView "Hey, are we still meeting at 4 PM?"\n[3] EditText "Message" [Input]',
            'You are in a WhatsApp chat with Bilal, who asked: "Hey, are we still meeting at 4 PM?"</thought>'
        ),
        # YouTube video
        (
            "tell me what's happening on screen",
            '[SCREEN: com.google.android.youtube]\n[1] TextView "Agentic AI Lecture - Full Course"\n[2] Button "Subscribe"',
            'You are viewing "Agentic AI Lecture - Full Course" on YouTube with a Subscribe button visible.</thought>'
        ),
        # SMS messages
        (
            "read the messages on screen",
            '[SCREEN: com.google.android.apps.messaging]\n[1] TextView "Mom"\n[2] TextView "Dinner is ready at 8 PM."\n[3] TextView "Call me when you leave office."',
            'Messages from Mom: "Dinner is ready at 8 PM." and "Call me when you leave office."</thought>'
        ),
        # Chrome article
        (
            "summarize this page",
            '[SCREEN: com.android.chrome]\n[1] TextView "Local LLM Deployment Guide"\n[2] TextView "FP16 reduces memory to 2.5GB."\n[3] TextView "ARM NEON speeds up mobile inference."',
            'This page is a Local LLM Deployment Guide explaining that FP16 uses 2.5GB memory and ARM NEON speeds up mobile inference.</thought>'
        ),
    ]
    for q, scr, ans in screen_qa:
        for _ in range(16):
            items.append({"category": "chat", "prompt": f"<user>{q}</user>\n<screen>\n{scr}\n</screen>\n<thought>", "target": ans})

    # =========================================================================
    # 3. Clean Pure Chat (Strictly NO <action>)
    # =========================================================================
    chat_terminations = [
        ("Who are you?", "I'm Yori, an offline executive AI assistant built by Embeddra AI to chat locally and control your Android phone.</thought>"),
        ("Explain recursion simply", "Recursion is when a function calls itself with a smaller input until it hits a stopping condition called the base case.</thought>"),
        ("Why is the sky blue?", "Sunlight hits air molecules in Earth's atmosphere, which scatter short blue light waves much more than red waves, making the sky look blue.</thought>"),
        ("Write a polite message saying I am running late", "Hi! I am running a few minutes late due to traffic. I will be there as soon as possible, thank you for understanding!</thought>"),
        ("Explain what a flashlight is", "A flashlight is a portable hand-held electric light powered by batteries, usually using an LED bulb and reflector.</thought>"),
    ]
    for q, ans in chat_terminations:
        for _ in range(20):
            items.append({"category": "chat", "prompt": f"<user>{q}</user>\n<thought>", "target": ans})

    # =========================================================================
    # 4. OS Tool Calls
    # =========================================================================
    tool_calls = [
        ("take a screenshot", "User wants to capture a screenshot of the screen.</thought>\n<action>SCREENSHOT()</action>"),
        ("capture screenshot", "User wants to capture a screenshot of the screen.</thought>\n<action>SCREENSHOT()</action>"),
        ("scroll down", "User wants to scroll down.</thought>\n<action>SCROLL_DOWN()</action>"),
        ("scroll up", "User wants to scroll up.</thought>\n<action>SCROLL_UP()</action>"),
        ("turn on flashlight", "User wants to turn on the flashlight.</thought>\n<action>TORCH(1)</action>"),
        ("turn off flashlight", "User wants to turn off the flashlight.</thought>\n<action>TORCH(0)</action>"),
    ]
    for q, tgt in tool_calls:
        for _ in range(20):
            items.append({"category": "tool_call", "prompt": f"<user>{q}</user>\n<thought>", "target": tgt})

    # =========================================================================
    # 5. Multi-Step for Other 7 Apps (Settings, WhatsApp, Clock, Notes, Spotify, Maps, Play Store)
    # =========================================================================
    other_multistep = [
        # Settings
        ("open settings and turn on bluetooth", None, 'User wants to use Settings. I will open Settings first.</thought>\n<action>OPEN_APP("Settings")</action>'),
        ("open settings and turn on bluetooth", '[SCREEN: com.android.settings]\n[1] TextView "Settings"\n[2] TextView "Connections"\n[3] TextView "Sounds and vibration"', 'I will click "Connections".</thought>\n<action>CLICK("Connections")</action>'),
        ("open settings and turn on bluetooth", '[SCREEN: com.android.settings]\n[1] Switch "Wi-Fi" (ON)\n[2] Switch "Bluetooth" (OFF)\n[3] Switch "NFC" (OFF)', 'I will click "Bluetooth".</thought>\n<action>CLICK("Bluetooth")</action>'),
        # WhatsApp
        ("open whatsapp and message Bilal saying Running late", None, 'User wants to use WhatsApp. I will open WhatsApp first.</thought>\n<action>OPEN_APP("WhatsApp")</action>'),
        ("open whatsapp and message Bilal saying Running late", '[SCREEN: com.whatsapp]\n[1] TextView "Pinned: Mom"\n[2] TextView "Bilal"\n[3] TextView "Sarah"', 'I will click "Bilal".</thought>\n<action>CLICK("Bilal")</action>'),
        ("open whatsapp and message Bilal saying Running late", '[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] EditText "Message"\n[3] Button "Attach"', 'Message is not typed yet. I will type "Running late" into "Message".</thought>\n<action>TYPE("Message", "Running late")</action>'),
        ("open whatsapp and message Bilal saying Running late", '[SCREEN: com.whatsapp]\n[1] EditText "Running late"\n[2] Button "Send"', 'Message is typed. I will click "Send".</thought>\n<action>CLICK("Send")</action>'),
        ("reply I am on my way", '[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] EditText "Message"\n[3] Button "Send"', 'Reply is not typed yet. I will type "I am on my way" into "Message".</thought>\n<action>TYPE("Message", "I am on my way")</action>'),
        # Clock
        ("open clock and create alarm named Yori Lunch", None, 'User wants to use Clock. I will open Clock first.</thought>\n<action>OPEN_APP("Clock")</action>'),
        ("open clock and create alarm named Yori Lunch", '[SCREEN: com.sec.android.app.clockpackage]\n[1] TextView "Alarm"\n[2] Button "Add alarm"', 'I will click "Add alarm".</thought>\n<action>CLICK("Add alarm")</action>'),
        ("open clock and create alarm named Yori Lunch", '[SCREEN: com.sec.android.app.clockpackage]\n[1] Button "Cancel"\n[2] EditText "Alarm name"\n[3] Button "Save"', 'I will type "Yori Lunch" into "Alarm name".</thought>\n<action>TYPE("Alarm name", "Yori Lunch")</action>'),
        ("open clock and create alarm named Yori Lunch", '[SCREEN: com.sec.android.app.clockpackage]\n[1] EditText "Yori Lunch"\n[2] Button "Save"', 'Alarm name is typed. I will click "Save".</thought>\n<action>CLICK("Save")</action>'),
        # Notes
        ("open notes and write Buy milk", None, 'User wants to use Notes. I will open Notes first.</thought>\n<action>OPEN_APP("Notes")</action>'),
        ("open notes and write Buy milk", '[SCREEN: com.samsung.android.app.notes]\n[1] TextView "All notes"\n[2] Button "Create note"', 'I will click "Create note".</thought>\n<action>CLICK("Create note")</action>'),
        ("open notes and write Buy milk", '[SCREEN: com.samsung.android.app.notes]\n[1] EditText "Title"\n[2] Button "Save"', 'Note is not typed yet. I will type "Buy milk" into "Title".</thought>\n<action>TYPE("Title", "Buy milk")</action>'),
        ("open notes and write Buy milk", '[SCREEN: com.samsung.android.app.notes]\n[1] EditText "Buy milk"\n[2] Button "Save"', 'Note is typed. I will click "Save".</thought>\n<action>CLICK("Save")</action>'),
        # Play Store
        ("install notion from play store", None, 'User wants to use Play Store. I will open Play Store first.</thought>\n<action>OPEN_APP("Play Store")</action>'),
        ("install notion from play store", '[SCREEN: com.android.vending]\n[1] TextView "For you"\n[2] Button "Search apps & games"', 'I will click "Search apps & games".</thought>\n<action>CLICK("Search apps & games")</action>'),
        ("install notion from play store", '[SCREEN: com.android.vending]\n[1] ImageButton "Back"\n[2] EditText "Search apps & games"', 'I will type "notion" into "Search apps & games".</thought>\n<action>TYPE("Search apps & games", "notion")</action>'),
        ("install notion from play store", '[SCREEN: com.android.vending]\n[1] TextView "Notion: Notes, Tasks"\n[2] TextView "Install"', 'I will click "Install".</thought>\n<action>CLICK("Install")</action>'),
        # Spotify
        ("play lofi beats on spotify", None, 'User wants to use Spotify. I will open Spotify first.</thought>\n<action>OPEN_APP("Spotify")</action>'),
        ("play lofi beats on spotify", '[SCREEN: com.spotify.music]\n[1] TextView "Home"\n[2] Button "Search"', 'I will click "Search".</thought>\n<action>CLICK("Search")</action>'),
        ("play lofi beats on spotify", '[SCREEN: com.spotify.music]\n[1] EditText "What do you want to listen to?"', 'I will type "lofi beats" into "What do you want to listen to?".</thought>\n<action>TYPE("What do you want to listen to?", "lofi beats")</action>'),
        ("play lofi beats on spotify", '[SCREEN: com.spotify.music]\n[1] ViewGroup "Lofi Beats Playlist"', 'I will click "Lofi Beats Playlist".</thought>\n<action>CLICK("Lofi Beats Playlist")</action>'),
        # Maps
        ("navigate to fast university on maps", None, 'User wants to use Google Maps. I will open Maps first.</thought>\n<action>OPEN_APP("Maps")</action>'),
        ("navigate to fast university on maps", '[SCREEN: com.google.android.apps.maps]\n[1] EditText "Search here"\n[2] Button "Explore"', 'I will type "fast university" into "Search here".</thought>\n<action>TYPE("Search here", "fast university")</action>'),
        ("navigate to fast university on maps", '[SCREEN: com.google.android.apps.maps]\n[1] ViewGroup "FAST University"\n[2] Button "Directions"', 'I will click "Directions".</thought>\n<action>CLICK("Directions")</action>'),
    ]
    for goal, scr, tgt in other_multistep:
        for _ in range(6):
            p = f"<user>{goal}</user>\n<screen>\n{scr}\n</screen>\n<thought>" if scr else f"<user>{goal}</user>\n<thought>"
            cat = "ui_grounding" if scr else "tool_call"
            items.append({"category": cat, "prompt": p, "target": tgt})

    # =========================================================================
    # 6. Replay UI labels & general data to preserve zero-shot grounding
    # =========================================================================
    ui_pool, chat_pool, tool_pool = [], [], []
    with open(LABEL_UI_TRAIN, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            if obj["category"] == "ui_grounding": ui_pool.append(obj)
            elif obj["category"] == "chat": chat_pool.append(obj)
            elif obj["category"] == "tool_call": tool_pool.append(obj)

    random.shuffle(ui_pool)
    random.shuffle(chat_pool)
    random.shuffle(tool_pool)
    items.extend(ui_pool[:80])
    items.extend(chat_pool[:60])
    items.extend(tool_pool[:40])

    random.shuffle(items)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Built master dataset: {len(items)} examples -> {OUT_TRAIN}", flush=True)


class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 32, lora_alpha: int = 64):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.scaling = lora_alpha / r
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, original_linear.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(original_linear.out_features, r, dtype=dtype))
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = self.original_linear(x)
        lora = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return orig + lora


def load_tokenized(path, tokenizer, max_len=440):
    eos_id = tokenizer.token_to_id("<|end_of_text|>") or 128001
    bos_id = tokenizer.token_to_id("<|begin_of_text|>") or 128000
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            p_ids = [bos_id] + tokenizer.encode(obj["prompt"]).ids
            t_ids = tokenizer.encode(obj["target"]).ids + [eos_id]
            full_ids = (p_ids + t_ids)[:max_len]
            labels = ([-100] * len(p_ids) + t_ids)[:max_len]
            if any(x != -100 for x in labels):
                data.append((obj.get("category", "chat"), full_ids, labels))
    return data


@torch.no_grad()
def eval_val_ppl(model, val_items, device):
    model.eval()
    stats = {"chat": [0.0, 0], "tool_call": [0.0, 0]}
    for cat, full_ids, labels in val_items:
        c = "tool_call" if cat != "chat" else "chat"
        x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100, reduction="sum")
        n_tok = int((y != -100).sum().item())
        stats[c][0] += float(loss.item())
        stats[c][1] += n_tok
    return {k: (v[0] / max(1, v[1]), math.exp(min(20.0, v[0] / max(1, v[1])))) for k, v in stats.items()}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== MASTER MULTI-STEP POLISH ON {device} ===", flush=True)
    t0 = time.time()

    build_dataset()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    train_items = load_tokenized(OUT_TRAIN, tokenizer)
    val_items = load_tokenized(VAL_JSONL, tokenizer)

    model = MariaForCausalLM.from_pretrained_checkpoint(BASE_CKPT, device=device, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            setattr(block.attn, attr, LoRALinear(getattr(block.attn, attr), r=32, lora_alpha=64).to(device))
        for attr in ["gate", "up", "down"]:
            setattr(block.ffn, attr, LoRALinear(getattr(block.ffn, attr), r=32, lora_alpha=64).to(device))

    lora_ckpt = torch.load(LORA_RESUME, map_location=device, weights_only=False)
    model.load_state_dict(lora_ckpt, strict=False)
    print(f"Resumed cleanly from {LORA_RESUME}.", flush=True)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    MAX_STEPS = 70
    WARMUP_STEPS = 6
    PEAK_LR = 2.5e-5
    MIN_LR = 5e-6
    GRAD_ACCUM = 8

    optimizer = torch.optim.AdamW(trainable_params, lr=PEAK_LR, weight_decay=0.01)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return float(step + 1) / float(WARMUP_STEPS)
        prog = float(step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
        return (MIN_LR / PEAK_LR) + (1.0 - (MIN_LR / PEAK_LR)) * 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    pre = eval_val_ppl(model, val_items, device)
    print(f"  [VAL @ Step 0] chat PPL={pre['chat'][1]:.2f} | tool_call PPL={pre['tool_call'][1]:.2f}", flush=True)

    idx_pool = list(range(len(train_items)))
    random.shuffle(idx_pool)
    ptr = 0
    best_t_ppl = float("inf")
    best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    for step in range(1, MAX_STEPS + 1):
        model.train()
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(GRAD_ACCUM):
            if ptr >= len(idx_pool):
                random.shuffle(idx_pool)
                ptr = 0
            _, full_ids, labels = train_items[idx_pool[ptr]]
            ptr += 1
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100) / GRAD_ACCUM
            loss.backward()
            step_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()

        if step % 25 == 0 or step == 1 or step == MAX_STEPS:
            val = eval_val_ppl(model, val_items, device)
            c_ppl, t_ppl = val["chat"][1], val["tool_call"][1]
            print(f"Step {step:2d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | VAL: chat={c_ppl:.2f}, tool={t_ppl:.2f} | Elapsed={time.time() - t0:.1f}s", flush=True)
            if t_ppl <= best_t_ppl or step == MAX_STEPS:
                best_t_ppl = min(best_t_ppl, t_ppl)
                best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                torch.save(best_lora_state, OUT_LORA)

    model.load_state_dict(best_lora_state, strict=False)

    print("\nFusing LoRA into standalone checkpoint...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for mod, attrs in [(block.attn, ["q_proj", "k_proj", "v_proj", "o_proj"]), (block.ffn, ["gate", "up", "down"])]:
                for attr in attrs:
                    layer = getattr(mod, attr)
                    if isinstance(layer, LoRALinear):
                        delta = (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
                        fused = (layer.original_linear.weight.float() + delta).to(layer.original_linear.weight.dtype)
                        layer.original_linear.weight.copy_(fused)
                        setattr(mod, attr, layer.original_linear)

    ckpt = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k in sd:
            sd[k] = v.cpu().to(sd[k].dtype)
    torch.save(ckpt, OUT_PT)

    serialize_checkpoint_to_eer(OUT_PT, OUT_EER, target_dtype="fp16")
    print(f"ALL DONE in {time.time() - t0:.1f}s!", flush=True)


if __name__ == "__main__":
    main()

