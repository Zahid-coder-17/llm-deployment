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
LORA_RESUME = os.path.join(BASE_DIR, "lora_screen_qa_all8_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_final_polish_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
LABEL_UI_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_label_ui_train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")
OUT_TRAIN = os.path.join(BASE_DIR, "hf_cache", "yori_final_polish.jsonl")

def build_dataset():
    random.seed(2090)
    items = []

    # 1. Screen Reading / Summarizing / What's Happening (strictly NO <action>)
    screen_qa = [
        (
            "summarize what is on my screen",
            '[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] TextView "Hey, are we still meeting at 4 PM?"\n[3] EditText "Message" [Input]',
            'You are in a WhatsApp chat with Bilal, who asked: "Hey, are we still meeting at 4 PM?"</thought>'
        ),
        (
            "tell me what's happening on screen",
            '[SCREEN: com.google.android.youtube]\n[1] TextView "Agentic AI Lecture - Full Course"\n[2] Button "Subscribe"',
            'You are viewing "Agentic AI Lecture - Full Course" on YouTube with a Subscribe button visible.</thought>'
        ),
        (
            "read the messages on screen",
            '[SCREEN: com.google.android.apps.messaging]\n[1] TextView "Mom"\n[2] TextView "Dinner is ready at 8 PM."\n[3] TextView "Call me when you leave office."',
            'Messages from Mom: "Dinner is ready at 8 PM." and "Call me when you leave office."</thought>'
        ),
        (
            "summarize this page",
            '[SCREEN: com.android.chrome]\n[1] TextView "Local LLM Deployment Guide"\n[2] TextView "FP16 reduces memory to 2.5GB."\n[3] TextView "ARM NEON speeds up mobile inference."',
            'This page is a Local LLM Deployment Guide explaining that FP16 uses 2.5GB memory and ARM NEON speeds up mobile inference.</thought>'
        ),
    ]
    for q, scr, ans in screen_qa:
        for _ in range(25):
            items.append({"category": "chat", "prompt": f"<user>{q}</user>\n<screen>\n{scr}\n</screen>\n<thought>", "target": ans})

    # 2. Reply & Notes Step 3 fixes
    for _ in range(30):
        items.append({
            "category": "ui_grounding",
            "prompt": '<user>reply I am on my way</user>\n<screen>\n[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] EditText "Message" [Input]\n[3] Button "Send"\n</screen>\n<thought>',
            "target": 'Reply is not typed yet. I will type "I am on my way" into "Message".</thought>\n<action>TYPE("Message", "I am on my way")</action>'
        })
        items.append({
            "category": "ui_grounding",
            "prompt": '<user>open notes and write Buy milk</user>\n<screen>\n[SCREEN: com.samsung.android.app.notes]\n[1] EditText "Title" [Input]\n[2] Button "Save"\n</screen>\n<thought>',
            "target": 'Note is not typed yet. I will type "Buy milk" into "Title".</thought>\n<action>TYPE("Title", "Buy milk")</action>'
        })

    # 3. Distinct Screenshot vs Scroll thoughts & actions
    for _ in range(20):
        items.append({"category": "tool_call", "prompt": "<user>take a screenshot</user>\n<thought>", "target": "User wants to capture a screenshot of the screen.</thought>\n<action>SCREENSHOT()</action>"})
        items.append({"category": "tool_call", "prompt": "<user>capture screenshot</user>\n<thought>", "target": "User wants to capture a screenshot of the screen.</thought>\n<action>SCREENSHOT()</action>"})
        items.append({"category": "tool_call", "prompt": "<user>scroll down</user>\n<thought>", "target": "User wants to scroll down.</thought>\n<action>SCROLL_DOWN()</action>"})
        items.append({"category": "tool_call", "prompt": "<user>scroll up</user>\n<thought>", "target": "User wants to scroll up.</thought>\n<action>SCROLL_UP()</action>"})

    # 4. Replay all 8 RL environments (1x each * 5 = 155)
    all8_steps = [
        ("find Agentic AI lecture and play it from youtube", None, 'User wants to use YouTube. I will open YouTube first.</thought>\n<action>OPEN_APP("YouTube")</action>'),
        ("find Agentic AI lecture and play it from youtube", '[SCREEN: com.google.android.youtube]\n[1] TextView "YouTube"\n[2] Button "Notifications"\n[3] Button "Search"\n[4] TextView "Shorts"', 'I will click "Search".</thought>\n<action>CLICK("Search")</action>'),
        ("find Agentic AI lecture and play it from youtube", '[SCREEN: com.google.android.youtube]\n[1] ImageButton "Navigate up"\n[2] EditText "Search YouTube" [Input]\n[3] Button "Voice search"', 'I will type "Agentic AI lecture" into "Search YouTube".</thought>\n<action>TYPE("Search YouTube", "Agentic AI lecture")</action>'),
        ("find Agentic AI lecture and play it from youtube", '[SCREEN: com.google.android.youtube]\n[1] EditText "Agentic AI lecture" [Input]\n[2] Button "Filters"\n[3] TextView "Agentic AI Lecture - Full Course"', '"Agentic AI lecture" is entered. I will click "Agentic AI Lecture - Full Course".</thought>\n<action>CLICK("Agentic AI Lecture - Full Course")</action>'),
        ("play lofi beats on spotify", None, 'User wants to use Spotify. I will open Spotify first.</thought>\n<action>OPEN_APP("Spotify")</action>'),
        ("play lofi beats on spotify", '[SCREEN: com.spotify.music]\n[1] TextView "Good evening"\n[2] Button "Home"\n[3] Button "Search"', 'I will click "Search".</thought>\n<action>CLICK("Search")</action>'),
        ("play lofi beats on spotify", '[SCREEN: com.spotify.music]\n[1] ImageButton "Back"\n[2] EditText "What do you want to listen to?" [Input]\n[3] Button "Voice"', 'I will type "lofi beats" into "What do you want to listen to?".</thought>\n<action>TYPE("What do you want to listen to?", "lofi beats")</action>'),
        ("play lofi beats on spotify", '[SCREEN: com.spotify.music]\n[1] EditText "lofi beats" [Input]\n[2] TextView "Lofi Beats Playlist"\n[3] TextView "Jazz Classics"', '"lofi beats" is entered. I will click "Lofi Beats Playlist".</thought>\n<action>CLICK("Lofi Beats Playlist")</action>'),
        ("find coffee shops near me on maps", None, 'User wants to use Maps. I will open Maps first.</thought>\n<action>OPEN_APP("Maps")</action>'),
        ("find coffee shops near me on maps", '[SCREEN: com.google.android.apps.maps]\n[1] Button "Search here"\n[2] Button "Restaurants"\n[3] Button "Gas"', 'I will click "Search here".</thought>\n<action>CLICK("Search here")</action>'),
        ("find coffee shops near me on maps", '[SCREEN: com.google.android.apps.maps]\n[1] ImageButton "Back"\n[2] EditText "Search here" [Input]', 'I will type "coffee shops" into "Search here".</thought>\n<action>TYPE("Search here", "coffee shops")</action>'),
        ("find coffee shops near me on maps", '[SCREEN: com.google.android.apps.maps]\n[1] EditText "coffee shops" [Input]\n[2] TextView "Blue Bottle Coffee"\n[3] TextView "Starbucks"', '"coffee shops" is entered. I will click "Blue Bottle Coffee".</thought>\n<action>CLICK("Blue Bottle Coffee")</action>'),
        ("open settings and turn on bluetooth", None, 'User wants to change a setting. I will open Settings first.</thought>\n<action>OPEN_APP("Settings")</action>'),
        ("open settings and turn on bluetooth", '[SCREEN: com.android.settings]\n[1] TextView "Search"\n[2] TextView "Connections"\n[3] TextView "Display"', 'I will click "Connections".</thought>\n<action>CLICK("Connections")</action>'),
        ("open settings and turn on bluetooth", '[SCREEN: com.android.settings]\n[1] Switch "Wi-Fi" (ON)\n[2] Switch "Bluetooth" (OFF)\n[3] Switch "NFC" (OFF)', 'I will click "Bluetooth".</thought>\n<action>CLICK("Bluetooth")</action>'),
        ("open whatsapp and message Bilal saying Running late", None, 'User wants to use WhatsApp. I will open WhatsApp first.</thought>\n<action>OPEN_APP("WhatsApp")</action>'),
        ("open whatsapp and message Bilal saying Running late", '[SCREEN: com.whatsapp]\n[1] TextView "Pinned: Mom"\n[2] TextView "Bilal"\n[3] TextView "Sarah"', 'I will click "Bilal".</thought>\n<action>CLICK("Bilal")</action>'),
        ("open whatsapp and message Bilal saying Running late", '[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] EditText "Message" [Input]\n[3] Button "Attach"', 'Message is not typed yet. I will type "Running late" into "Message".</thought>\n<action>TYPE("Message", "Running late")</action>'),
        ("open whatsapp and message Bilal saying Running late", '[SCREEN: com.whatsapp]\n[1] EditText "Running late" [Input]\n[2] Button "Send"', 'Message is typed. I will click "Send".</thought>\n<action>CLICK("Send")</action>'),
        ("open clock and create alarm named Yori Lunch", None, 'User wants to use Clock. I will open Clock first.</thought>\n<action>OPEN_APP("Clock")</action>'),
        ("open clock and create alarm named Yori Lunch", '[SCREEN: com.sec.android.app.clockpackage]\n[1] TextView "Alarm"\n[2] Button "Add alarm"', 'I will click "Add alarm".</thought>\n<action>CLICK("Add alarm")</action>'),
        ("open clock and create alarm named Yori Lunch", '[SCREEN: com.sec.android.app.clockpackage]\n[1] Button "Cancel"\n[2] EditText "Alarm name" [Input]\n[3] Button "Save"', 'I will type "Yori Lunch" into "Alarm name".</thought>\n<action>TYPE("Alarm name", "Yori Lunch")</action>'),
        ("open clock and create alarm named Yori Lunch", '[SCREEN: com.sec.android.app.clockpackage]\n[1] EditText "Yori Lunch" [Input]\n[2] Button "Save"', 'Alarm name is typed. I will click "Save".</thought>\n<action>CLICK("Save")</action>'),
        ("install notion from play store", None, 'User wants to use Play Store. I will open Play Store first.</thought>\n<action>OPEN_APP("Play Store")</action>'),
        ("install notion from play store", '[SCREEN: com.android.vending]\n[1] TextView "For you"\n[2] Button "Search apps & games"', 'I will click "Search apps & games".</thought>\n<action>CLICK("Search apps & games")</action>'),
        ("install notion from play store", '[SCREEN: com.android.vending]\n[1] ImageButton "Back"\n[2] EditText "Search apps & games" [Input]', 'I will type "notion" into "Search apps & games".</thought>\n<action>TYPE("Search apps & games", "notion")</action>'),
        ("install notion from play store", '[SCREEN: com.android.vending]\n[1] TextView "Notion: Notes, Tasks"\n[2] TextView "Install"', 'I will click "Install".</thought>\n<action>CLICK("Install")</action>'),
        ("open notes and write Buy milk", None, 'User wants to use Notes. I will open Notes first.</thought>\n<action>OPEN_APP("Notes")</action>'),
        ("open notes and write Buy milk", '[SCREEN: com.samsung.android.app.notes]\n[1] TextView "All notes"\n[2] Button "Create note"', 'I will click "Create note".</thought>\n<action>CLICK("Create note")</action>'),
        ("open notes and write Buy milk", '[SCREEN: com.samsung.android.app.notes]\n[1] EditText "Buy milk" [Input]\n[2] Button "Save"', 'Note is typed. I will click "Save".</thought>\n<action>CLICK("Save")</action>'),
    ]
    for goal, scr, tgt in all8_steps:
        for _ in range(4):
            p = f"<user>{goal}</user>\n<screen>\n{scr}\n</screen>\n<thought>" if scr else f"<user>{goal}</user>\n<thought>"
            cat = "ui_grounding" if scr else "tool_call"
            items.append({"category": cat, "prompt": p, "target": tgt})

    # 5. Replay smoke-test UI, chat, tool_call
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
    items.extend(ui_pool[:60])
    items.extend(chat_pool[:60])
    items.extend(tool_pool[:30])

    random.shuffle(items)
    with open(OUT_TRAIN, "w", encoding="utf-8") as f:
        for ex in items:
            f.write(json.dumps(ex) + "\n")
    print(f"Built {OUT_TRAIN}: {len(items)} examples", flush=True)

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 32, lora_alpha: int = 64):
        super().__init__()
        self.original_linear = original_linear
        for p in self.original_linear.parameters():
            p.requires_grad = False
        self.scaling = lora_alpha / r
        dtype = original_linear.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(r, original_linear.in_features, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(original_linear.out_features, r, dtype=dtype))
        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original_linear(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

def load_tokenized(path, tokenizer, max_len=440):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            obj = json.loads(line)
            p_ids = tokenizer.encode(obj["prompt"]).ids
            t_ids = tokenizer.encode(obj["target"]).ids
            full_ids = p_ids + t_ids
            if 4 <= len(full_ids) <= max_len:
                labels = [-100] * len(p_ids) + t_ids
                items.append((obj["category"], full_ids, labels))
    return items

def eval_val_ppl(model, val_items, device):
    model.eval()
    stats = {}
    with torch.no_grad():
        for cat, full_ids, labels in val_items:
            x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([labels[1:]], dtype=torch.long, device=device)
            logits, _ = model(x)
            nll = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100, reduction="sum").item()
            ntoks = (y.view(-1) != -100).sum().item()
            if cat not in stats: stats[cat] = [0.0, 0]
            stats[cat][0] += nll
            stats[cat][1] += ntoks
    return {c: (s[0]/max(1,s[1]), math.exp(min(s[0]/max(1,s[1]), 20.0)), s[1]) for c, s in stats.items()}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI FINAL POLISH (SCREEN QA + REPLY + SCREENSHOT + NOTES) ON {device} ===", flush=True)
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
    MAX_STEPS = 60
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

        if step % 30 == 0 or step == 1 or step == MAX_STEPS:
            val = eval_val_ppl(model, val_items, device)
            c_ppl, t_ppl = val["chat"][1], val["tool_call"][1]
            print(f"Step {step:2d}/{MAX_STEPS} | TrainLoss={step_loss:.4f} | VAL: chat={c_ppl:.2f}, tool={t_ppl:.2f} | Elapsed={time.time() - t0:.1f}s", flush=True)
            if t_ppl <= 1.50:
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

