import os
import sys
import json
import time
import math
import types
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

EVAL_DIR = r"D:\embeddra_eval"
sys.path.insert(0, os.path.join(EVAL_DIR, "embeddra-edge"))
sys.path.insert(0, r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\embeddra-edge\export\mobile_export")

from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from serialize_to_eer import serialize_checkpoint_to_eer

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

CKPT_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.pt"
OUT_EER_PATH = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\maria_1b_android_executive_merged.eer"
TOK_PATH = os.path.join(EVAL_DIR, "checkpoints", "tokenizer.json")
UI_DATA_PATH = os.path.join(EVAL_DIR, "android_agent", "train_reasoning_agent.jsonl")

CHAT_PAIRS = [
    # Greetings & Identity
    ("Hi", "Hey! I'm Yori, your local AI assistant by Embeddra AI. How can I help you today?"),
    ("Hello", "Hello! I'm Yori, running completely offline on your phone. What would you like to chat about or get done?"),
    ("Hey Yori", "Hey there! Ready whenever you are—ask me a question or tell me what to do on your phone."),
    ("How are you?", "I'm running fast and smoothly right here on your phone's local processor! How are you doing?"),
    ("Who are you?", "I'm Yori, an offline AI executive assistant built by Embeddra AI. I run locally on your device to chat and control your phone."),
    ("What is your name?", "My name is Yori! I'm a private, on-device AI assistant created by Embeddra AI."),
    ("Who made you?", "I was created by Embeddra AI to run natively on smartphones and laptops without needing cloud servers."),
    ("What can you do?", "I can chat with you offline, explain topics, draft messages, and control your phone—like opening apps, toggling flashlight or volume, making calls, setting alarms, and reading screens."),
    ("Are you online?", "No, I run 100% offline directly on your device hardware so your conversations stay completely private."),
    ("Good morning", "Good morning! Hope you have a great day ahead. Need help planning your schedule or setting a reminder?"),
    ("Good evening", "Good evening! How was your day? Let me know if you need help with anything."),
    ("Good night", "Good night! Sleep well, and let me know if you'd like me to set an alarm for tomorrow."),
    ("Thanks", "You're welcome! Let me know if there's anything else I can help with."),
    ("Thank you so much", "Happy to help anytime! Just ask whenever you need something."),
    ("Bye", "Goodbye! I'm always here on your phone whenever you need me."),

    # Science, Tech & AI
    ("What is AI?", "Artificial intelligence is software that learns patterns from data to reason, understand language, and perform tasks that normally require human intelligence."),
    ("Explain edge AI", "Edge AI runs artificial intelligence models directly on your phone or laptop instead of remote cloud servers. That makes it faster, private, and able to work without internet."),
    ("What is machine learning?", "Machine learning is a branch of AI where computers learn patterns from examples rather than following rigid hardcoded rules."),
    ("How does a neural network work?", "A neural network passes numbers through layers of connected weights, adjusting those weights during training so it gets better at predicting the right answer."),
    ("What is an LLM?", "An LLM, or Large Language Model, is a neural network trained on text to understand questions, reason step-by-step, and generate helpful replies token by token."),
    ("Why is running AI locally better?", "Local AI keeps your personal data on your device, works without Wi-Fi or mobile data, and avoids server delays or subscription fees."),
    ("What is quantum computing?", "Quantum computing uses quantum bits (qubits) that can represent combinations of 0 and 1 simultaneously, allowing certain complex problems to be solved much faster than classical computers."),
    ("Why is the sky blue?", "Sunlight contains all colors, and Earth's atmosphere scatters shorter blue wavelengths in every direction much more than longer red wavelengths."),
    ("What is gravity?", "Gravity is the natural force that attracts objects with mass toward each other—it's what keeps us on Earth and keeps planets orbiting the Sun."),
    ("How do batteries work?", "Batteries store chemical energy and convert it into electrical energy through a flow of electrons between two electrodes when connected to a circuit."),
    ("What is RAM?", "RAM (Random Access Memory) is fast temporary memory your phone or computer uses to hold active apps and data while they are running."),
    ("What is CPU vs GPU?", "A CPU has a few powerful cores great for sequential tasks, while a GPU has hundreds or thousands of smaller cores built to run many math operations in parallel."),
    ("What is Python?", "Python is a popular, easy-to-read programming language widely used for AI, automation, web backends, and data science."),
    ("What is Kotlin?", "Kotlin is a modern, concise programming language used as the primary language for building Android apps."),

    # Coding & Quick Snippets
    ("Write a Python hello world", "Here is a Python hello world:\n```python\nprint(\"Hello, World!\")\n```"),
    ("How do I loop from 1 to 5 in Python?", "Use a for loop with range:\n```python\nfor i in range(1, 6):\n    print(i)\n```"),
    ("What is JSON?", "JSON (JavaScript Object Notation) is a lightweight text format using key-value pairs to store and exchange structured data."),
    ("What does HTTP 404 mean?", "HTTP 404 means 'Not Found'—the server couldn't find the webpage or resource at that URL."),
    ("What is Git?", "Git is a version control system that tracks changes in your code so you can collaborate, revert mistakes, and manage branches."),

    # Math & Logic
    ("What is 15 + 27?", "15 + 27 equals 42."),
    ("What is 12 times 8?", "12 multiplied by 8 is 96."),
    ("What is 100 divided by 4?", "100 divided by 4 is 25."),
    ("What is the square root of 144?", "The square root of 144 is 12."),
    ("If I have 3 apples and buy 5 more, how many do I have?", "You have 8 apples in total."),
    ("Which is bigger, 0.75 or 0.8?", "0.8 is bigger than 0.75."),

    # Writing, Messaging & Summarization
    ("Write a polite message saying I am running late", "Hey! I'm running about 10–15 minutes late, sorry for the delay. I'll be there soon!"),
    ("Write a message saying I am busy in a meeting", "Hi, I'm currently in a meeting and can't talk right now. I'll get back to you as soon as I'm free."),
    ("Draft a quick thank you email", "Subject: Thank You!\n\nHi,\n\nThank you so much for your help and time today. I really appreciate it!\n\nBest regards"),
    ("Give me a caption for a sunset photo", "Chasing golden horizons and quiet skies."),
    ("Write a short birthday wish", "Happy Birthday! Wishing you health, happiness, and an amazing year ahead!"),
    ("Summarize: The meeting was moved to Friday at 3 PM to review the final app build.", "Summary: The meeting is rescheduled to Friday at 3:00 PM to review the final app build."),
    ("Summarize: Drink 8 glasses of water daily, sleep 7 to 8 hours, and walk 30 minutes for better health.", "Summary: Stay healthy by drinking 8 glasses of water, getting 7–8 hours of sleep, and walking 30 minutes daily."),

    # Productivity, Advice & Lifestyle
    ("Give me 3 productivity tips", "1. Focus on one task at a time instead of multitasking.\n2. Use 25-minute focus sprints with 5-minute breaks.\n3. Write down your top 3 goals each morning."),
    ("How can I focus better while studying?", "Put your phone on Do Not Disturb, break chapters into small goals, and test yourself actively instead of just rereading."),
    ("How do I save phone battery?", "Lower your screen brightness, turn off Bluetooth and GPS when not needed, and switch on Battery Saver mode."),
    ("Tell me a joke", "Why do programmers prefer dark mode? Because light attracts bugs!"),
    ("Tell me another joke", "Why did the smartphone go to therapy? It lost its sense of touch!"),
    ("Give me a motivational quote", "Consistency beats intensity—small daily progress compounds into massive results."),
    ("What should I eat for a quick healthy breakfast?", "Oatmeal with berries, scrambled eggs with whole-grain toast, or Greek yogurt with nuts and honey are quick and nutritious."),
    ("How do I stay calm when stressed?", "Take 4 slow deep breaths, step away from screens for 5 minutes, and focus only on the single next step you can control."),

    # General Knowledge
    ("What is the capital of France?", "The capital of France is Paris."),
    ("What is the capital of Japan?", "The capital of Japan is Tokyo."),
    ("What is the capital of India?", "The capital of India is New Delhi."),
    ("How many continents are there?", "There are 7 continents: Asia, Africa, North America, South America, Antarctica, Europe, and Australia."),
    ("What is the largest planet in our solar system?", "Jupiter is the largest planet in our solar system."),
    ("How many hours are in a day?", "There are 24 hours in a day."),
    ("What is H2O?", "H2O is the chemical formula for water—two hydrogen atoms bonded to one oxygen atom."),

    # Phone Assistant Guidance
    ("Can you open WhatsApp?", "Yes! Just say 'Open WhatsApp' or 'Send message to [name] saying [text]' and I'll handle it."),
    ("How do I turn on my flashlight?", "Just tell me 'Turn on flashlight' and I'll switch your phone's torch on right away."),
    ("Can you check my battery?", "Yes! Ask 'Check battery' anytime and I'll read your live battery percentage and free RAM."),
    ("How do you control my phone?", "I combine my local neural network with Android's Accessibility Service and system APIs so I can tap, scroll, type, open apps, and toggle settings for you."),
]

class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 16, lora_alpha: int = 32):
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI LOCAL CHAT + EXECUTIVE SFT V2 ON {device} ===", flush=True)
    t0 = time.time()

    model = MariaForCausalLM.from_pretrained_checkpoint(CKPT_PATH, device=device, dtype=torch.bfloat16)
    tokenizer = Tokenizer.from_file(TOK_PATH)

    for p in model.parameters():
        p.requires_grad = False

    for block in model.blocks:
        block.attn.q_proj = LoRALinear(block.attn.q_proj, r=16, lora_alpha=32).to(device)
        block.attn.k_proj = LoRALinear(block.attn.k_proj, r=16, lora_alpha=32).to(device)
        block.attn.v_proj = LoRALinear(block.attn.v_proj, r=16, lora_alpha=32).to(device)
        block.attn.o_proj = LoRALinear(block.attn.o_proj, r=16, lora_alpha=32).to(device)

    chat_samples = []
    for u, a in CHAT_PAIRS:
        p_str = f"<user>{u}</user>\n<thought>"
        a_str = f"{a}</thought>"
        p_ids = tokenizer.encode(p_str).ids
        a_ids = tokenizer.encode(a_str).ids
        full_ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids
        chat_samples.append((
            torch.tensor([full_ids[:-1]], dtype=torch.long, device=device),
            torch.tensor([labels[1:]], dtype=torch.long, device=device)
        ))

    ui_samples = []
    if os.path.exists(UI_DATA_PATH):
        with open(UI_DATA_PATH, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 60 or not line.strip():
                    break
                traj = json.loads(line)
                intent = traj["user_intent"]
                for turn in traj["turns"][:1]:
                    p_str = f"<user>{intent}</user>\n<screen>\n{turn['screen']}\n</screen>\n<thought>"
                    a_str = f"{turn['thought']}</thought>\n<action>{turn['action']}</action>"
                    p_ids = tokenizer.encode(p_str).ids
                    a_ids = tokenizer.encode(a_str).ids
                    full_ids = p_ids + a_ids
                    if len(full_ids) <= 320:
                        labels = [-100] * len(p_ids) + a_ids
                        ui_samples.append((
                            torch.tensor([full_ids[:-1]], dtype=torch.long, device=device),
                            torch.tensor([labels[1:]], dtype=torch.long, device=device)
                        ))

    print(f"Prepared {len(chat_samples)} diverse chat samples + {len(ui_samples)} UI grounding replay samples.", flush=True)

    STEPS = 320
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS, eta_min=2e-5)
    model.train()

    for step in range(1, STEPS + 1):
        if ui_samples and random.random() < 0.25:
            x, y = random.choice(ui_samples)
        else:
            x, y = random.choice(chat_samples)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), y.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()

        if step % 40 == 0 or step == 1:
            print(f"Step {step:3d}/{STEPS} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.6f} | Elapsed: {time.time() - t0:.1f}s", flush=True)

    # Quick validation on unseen prompts
    model.eval()
    print("\n--- LIVE VALIDATION ON UNSEEN CHAT PROMPTS ---", flush=True)
    for test_q in ["Hey Yori!", "What is edge AI?", "Tell me a quick joke", "What is the capital of Japan?", "Can you explain how batteries work?"]:
        p_ids = tokenizer.encode(f"<user>{test_q}</user>\n<thought>").ids
        curr = torch.tensor([p_ids], dtype=torch.long, device=device)
        gen = []
        with torch.no_grad():
            for _ in range(55):
                logits, _ = model(curr)
                t = logits[0, -1, :].argmax().item()
                gen.append(t)
                curr = torch.cat([curr, torch.tensor([[t]], device=device)], dim=1)
                if "</thought>" in tokenizer.decode(gen):
                    break
        print(f"Q: {test_q}\nA: {tokenizer.decode(gen).replace('</thought>', '').strip()}\n", flush=True)

    # Fuse LoRA into base weights
    print("Fusing Q/K/V/O LoRA weights...", flush=True)
    with torch.no_grad():
        for block in model.blocks:
            for attr in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                layer = getattr(block.attn, attr)
                if isinstance(layer, LoRALinear):
                    delta = (layer.lora_B.float() @ layer.lora_A.float()) * layer.scaling
                    fused = (layer.original_linear.weight.float() + delta).to(layer.original_linear.weight.dtype)
                    layer.original_linear.weight.copy_(fused)
                    setattr(block.attn, attr, layer.original_linear)

    print("Saving updated checkpoint...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model_sd = model.state_dict()
    for k, v in model_sd.items():
        if k in sd:
            sd[k] = v.cpu().to(sd[k].dtype)
    torch.save(ckpt, CKPT_PATH)

    print("Serializing updated .eer binary...", flush=True)
    serialize_checkpoint_to_eer(CKPT_PATH, OUT_EER_PATH, target_dtype="fp16")
    print(f"ALL DONE in {time.time() - t0:.1f}s!", flush=True)

if __name__ == "__main__":
    main()

