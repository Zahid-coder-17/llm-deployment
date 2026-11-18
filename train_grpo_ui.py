import os
import sys
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_CACHE_DIR"] = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment\hf_cache\torch_cache"
import re
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
LORA_RESUME = os.path.join(BASE_DIR, "lora_grpo_v2_best.pt")
OUT_LORA = os.path.join(BASE_DIR, "lora_grpo_v3_best.pt")
OUT_PT = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.pt")
OUT_EER = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
RICO_UI_JSONL = r"D:\hf_datasets\yori_rico_label_ui.jsonl"
VAL_JSONL = os.path.join(BASE_DIR, "hf_cache", "yori_unified_val.jsonl")

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

def build_rl_tasks():
    random.seed(2041)
    tasks = []

    # Focused Step 3 (TYPE into EditText search bar) & Step 4 (CLICK matching result) across all apps
    step34_specs = [
        ("com.google.android.youtube", "find Agentic AI lecture and play it from youtube", "Search YouTube", "Agentic AI lecture", "Agentic AI Lecture - Full Course"),
        ("com.google.android.youtube", "search transformer architecture on youtube", "Search YouTube", "transformer architecture", "Transformers Explained Clearly"),
        ("com.spotify.music", "play lofi beats on spotify", "What do you want to listen to?", "lofi beats", "Lofi Beats Playlist"),
        ("com.spotify.music", "find jazz classics on spotify", "What do you want to listen to?", "jazz classics", "Jazz Classics Radio"),
        ("com.google.android.apps.maps", "find coffee shops near me on maps", "Search here", "coffee shops", "Blue Bottle Coffee"),
        ("com.google.android.apps.maps", "find gas station on maps", "Search here", "gas station", "Shell Gas Station"),
        ("com.amazon.mShop.android.shopping", "search wireless earbuds on amazon", "Search Amazon", "wireless earbuds", "Sony Wireless Earbuds"),
        ("com.android.vending", "find notion app on play store", "Search apps & games", "notion app", "Notion: Notes, Tasks, AI"),
        ("com.whatsapp", "find Bilal chat on whatsapp", "Search...", "Bilal", "Bilal"),
    ]
    for _ in range(120):
        pkg, u_goal, box_lbl, query_txt, res_lbl = random.choice(step34_specs)
        # Step 3: EditText typing
        s3 = f'[SCREEN: {pkg}]\n[1] ImageButton "Back"\n[2] EditText "{box_lbl}" [Input]\n[3] Button "Voice search"'
        tasks.append({
            "prompt": f"<user>{u_goal}</user>\n<screen>\n{s3}\n</screen>\n<thought>",
            "expected_action": f'TYPE("{box_lbl}", "{query_txt}")'
        })
        # Step 4: Click result at random position [2..5]
        distractor_res = ["Sponsored Ad", "Related Item", "More Results", "Filters"]
        r_pos = random.randint(2, 5)
        r_lines = [f"[SCREEN: {pkg}]", f'[1] EditText "{query_txt}" [Input]']
        d_ptr = 0
        for idx in range(2, 6):
            if idx == r_pos:
                r_lines.append(f'[{idx}] TextView "{res_lbl}"')
            else:
                r_lines.append(f'[{idx}] TextView "{distractor_res[d_ptr % len(distractor_res)]}"')
                d_ptr += 1
        tasks.append({
            "prompt": f"<user>{u_goal}</user>\n<screen>\n" + "\n".join(r_lines) + "\n</screen>\n<thought>",
            "expected_action": f'CLICK("{res_lbl}")'
        })

    random.shuffle(tasks)
    print(f"Built {len(tasks)} focused Step 3/4 RL tasks.", flush=True)
    return tasks

def extract_screen_labels(prompt: str):
    return set(m.lower() for m in re.findall(r'"([^"]+)"', prompt.split("<screen>")[-1].split("</screen>")[0]))

def compute_reward(gen_text: str, prompt: str, expected_action: str) -> float:
    if "</thought>" not in gen_text or "<action>" not in gen_text or "</action>" not in gen_text:
        return -1.0
    pred_act = gen_text.split("<action>")[1].split("</action>")[0].strip()
    if pred_act == expected_action:
        return 1.0

    screen_labels = extract_screen_labels(prompt)
    m_click = re.match(r'^CLICK\("(.*)"\)$', pred_act)
    m_type = re.match(r'^TYPE\("(.*?)",\s*"(.*)"\)$', pred_act)
    if m_click:
        lbl = m_click.group(1).strip().lower()
        if lbl in screen_labels:
            return 0.2   # Valid on-screen element, though not the ideal target
        return -1.0      # Hallucinated label NOT in <screen>!
    if m_type:
        lbl = m_type.group(1).strip().lower()
        if lbl in screen_labels:
            return 0.2
        return -1.0
    return -0.5

@torch.no_grad()
def sample_rollout(model, prompt_ids, device, eos_token_id, max_new_tokens=42, temp=0.7):
    curr = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    gen_ids = []
    for _ in range(max_new_tokens):
        logits, _ = model(curr)
        next_logits = logits[0, -1, :].float() / temp
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        gen_ids.append(next_id)
        curr = torch.cat([curr, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_token_id:
            break
    return gen_ids

def get_seq_logprobs(model, prompt_ids, gen_ids, device):
    full = prompt_ids + gen_ids
    x = torch.tensor([full[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([full[1:]], dtype=torch.long, device=device)
    logits, _ = model(x)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    # Only slice generated tokens
    start = len(prompt_ids) - 1
    gen_logits_lp = log_probs[0, start:start + len(gen_ids), :]
    target_ids = torch.tensor(gen_ids, dtype=torch.long, device=device).unsqueeze(-1)
    token_lp = gen_logits_lp.gather(dim=-1, index=target_ids).squeeze(-1)
    return token_lp

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== YORI GRPO RLVR (ALL-APP UI GROUNDING) ON {device} ===", flush=True)
    t0 = time.time()

    tasks = build_rl_tasks()
    tokenizer = Tokenizer.from_file(TOK_PATH)
    # Find token id for ">" or use stop check on decoded string; we decode per step in rollout
    eos_id = tokenizer.token_to_id("<|end_of_text|>") or 128001

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
    print(f"Resumed policy from {LORA_RESUME}.", flush=True)

    # Save reference LoRA weights for KL anchor
    ref_lora = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    MAX_STEPS = 60
    GROUP_SIZE = 4
    LR = 1.5e-5
    KL_BETA = 0.04
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    running_r = []
    for step in range(1, MAX_STEPS + 1):
        task = tasks[(step - 1) % len(tasks)]
        prompt_ids = tokenizer.encode(task["prompt"]).ids
        if len(prompt_ids) > 360:
            continue

        model.eval()
        rollouts = []
        rewards = []
        for _ in range(GROUP_SIZE):
            # Sample rollout stopping on </action>
            curr = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            gen_ids = []
            with torch.no_grad():
                for _ in range(40):
                    logits, _ = model(curr)
                    probs = F.softmax(logits[0, -1, :].float() / 0.7, dim=-1)
                    nid = torch.multinomial(probs, 1).item()
                    gen_ids.append(nid)
                    curr = torch.cat([curr, torch.tensor([[nid]], device=device)], dim=1)
                    if "</action>" in tokenizer.decode(gen_ids):
                        break
            txt = tokenizer.decode(gen_ids)
            r = compute_reward(txt, task["prompt"], task["expected_action"])
            rollouts.append(gen_ids)
            rewards.append(r)

        running_r.append(sum(rewards) / len(rewards))
        r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        std_r = r_tensor.std()
        if std_r < 1e-5:
            if step % 15 == 0 or step == 1:
                print(f"Step {step:2d}/{MAX_STEPS} | MeanReward={running_r[-1]:+.2f} (uniform, skipped) | Elapsed={time.time()-t0:.1f}s", flush=True)
            continue

        advs = (r_tensor - r_tensor.mean()) / (std_r + 1e-6)

        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        for gen_ids, adv in zip(rollouts, advs):
            lp = get_seq_logprobs(model, prompt_ids, gen_ids, device)
            pg_loss = -(adv.detach() * lp.sum()) / GROUP_SIZE
            pg_loss.backward()
            total_loss += pg_loss.item()

        # L2 KL-anchor toward reference LoRA weights
        kl_loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if param.requires_grad and name in ref_lora:
                kl_loss = kl_loss + KL_BETA * F.mse_loss(param.float(), ref_lora[name].float())
        if kl_loss.item() > 0:
            kl_loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()

        if step % 15 == 0 or step == 1 or step == MAX_STEPS:
            avg_15 = sum(running_r[-15:]) / len(running_r[-15:])
            print(f"Step {step:2d}/{MAX_STEPS} | AvgReward(15)={avg_15:+.3f} | Loss={total_loss:.4f} | Elapsed={time.time()-t0:.1f}s", flush=True)

    best_lora_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(best_lora_state, OUT_LORA)

    print("\nFusing GRPO RL LoRA into standalone checkpoint...", flush=True)
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

