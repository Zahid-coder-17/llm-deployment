import os
import sys
import time
import math
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

current_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

from hypergqe.model_adapter.maria_model import MariaForCausalLM

def main():
    ckpt_path = os.path.join(workspace_root, "checkpoints", "maria_1b_20B_weights.pt")
    tok_path = os.path.join(workspace_root, "checkpoints", "tokenizer.json")

    print(f"Loading Tokenizer from: {tok_path}")
    sys.stdout.flush()
    tokenizer = Tokenizer.from_file(tok_path)
    
    print(f"Loading Maria-1B Base Weights from: {ckpt_path}")
    sys.stdout.flush()
    t0 = time.time()
    model = MariaForCausalLM.from_pretrained_checkpoint(ckpt_path, device="cpu", dtype=torch.bfloat16)
    model.eval()
    print(f"[OK] Model loaded in {time.time() - t0:.2f}s\n")
    sys.stdout.flush()

    # Curated real-world evaluation benchmarks across 7 key agent domains
    domain_benchmarks = {
        "1. Medical & Clinical QA": (
            "Patient Profile: A 62-year-old female presents to the outpatient clinic with a 3-month history of progressive exertional dyspnea, "
            "bilateral lower extremity edema, and orthopnea requiring three pillows. Physical examination reveals elevated jugular venous pressure, "
            "a third heart sound (S3 gallop), and bibasilar crackles on pulmonary auscultation. Echocardiography demonstrates a left ventricular ejection "
            "fraction of 32% with global hypokinesis. The initial management plan includes loop diuretics for volume overload, guideline-directed "
            "medical therapy consisting of a beta-blocker, an angiotensin receptor-neprilysin inhibitor, a mineralocorticoid receptor antagonist, and an SGLT2 inhibitor."
        ),
        "2. Code (Python Software Dev)": (
            "import os\n"
            "import json\n"
            "from typing import List, Dict, Any, Optional\n\n"
            "class TaskScheduler:\n"
            "    def __init__(self, max_concurrency: int = 4) -> None:\n"
            "        self.max_concurrency = max_concurrency\n"
            "        self.task_queue: List[Dict[str, Any]] = []\n"
            "        self.active_tasks: Dict[str, Any] = {}\n\n"
            "    def submit(self, task_id: str, payload: Dict[str, Any]) -> bool:\n"
            "        if task_id in self.active_tasks:\n"
            "            return False\n"
            "        self.task_queue.append({'id': task_id, 'data': payload, 'status': 'pending'})\n"
            "        return True\n\n"
            "    def process_next(self) -> Optional[Dict[str, Any]]:\n"
            "        if len(self.active_tasks) >= self.max_concurrency or not self.task_queue:\n"
            "            return None\n"
            "        task = self.task_queue.pop(0)\n"
            "        self.active_tasks[task['id']] = task\n"
            "        return task\n"
        ),
        "3. Structured JSON & Tool Calling": (
            '{\n'
            '  "agent_action": "execute_tool",\n'
            '  "tool_name": "patient_vital_recorder",\n'
            '  "parameters": {\n'
            '    "patient_id": "PT-94821",\n'
            '    "vitals": {\n'
            '      "heart_rate_bpm": 84,\n'
            '      "blood_pressure_systolic": 138,\n'
            '      "blood_pressure_diastolic": 88,\n'
            '      "oxygen_saturation_pct": 98.2,\n'
            '      "temperature_celsius": 37.1\n'
            '    },\n'
            '    "timestamp": "2026-11-10T08:15:00Z",\n'
            '    "urgency_level": "normal"\n'
            '  }\n'
            '}'
        ),
        "4. Math & Quantitative Reasoning": (
            "Problem: A pharmaceutical laboratory synthesizes an active antibiotic compound with a continuous half-life of 6 hours in human plasma. "
            "If an initial intravenous dose of 500 milligrams is administered at t = 0, and the elimination follows first-order exponential decay kinetics, "
            "the concentration remaining after t hours is given by C(t) = C0 * exp(-k * t), where the decay constant k = ln(2) / 6 ≈ 0.1155 per hour. "
            "To calculate the remaining amount after 18 hours: since 18 hours corresponds to exactly 3 half-lives, the remaining mass is 500 * (1/2)^3 = 500 / 8 = 62.5 milligrams."
        ),
        "5. Multi-turn Conversational Chat": (
            "User: Hi, I have been experiencing sharp headaches on the right side of my head for the past two days, especially when looking at bright screens.\n"
            "Assistant: Hello! Unilateral headaches associated with light sensitivity (photophobia) are frequently characteristic of migraine presentations, "
            "though other causes like eye strain or cluster headaches should also be considered. Have you noticed any other symptoms such as nausea, visual auras, or neck stiffness?\n"
            "User: Yes, I felt slightly nauseous this morning when waking up."
        ),
        "6. General Knowledge & Wikipedia": (
            "The Industrial Revolution was the transition to new manufacturing processes in Great Britain, continental Europe, and the United States, "
            "in the period from approximately 1760 to about 1840. This transition included going from hand production methods to machines, new chemical "
            "manufacturing and iron production processes, the increasing use of steam power and water power, the development of machine tools, and the rise of the mechanized factory system."
        ),
        "7. Scientific / Biology Foundations": (
            "Mitochondria are double-membrane-bound organelles found in most eukaryotic organisms. They generate most of the cell's supply of adenosine "
            "triphosphate (ATP), used as a source of chemical energy through oxidative phosphorylation. The outer mitochondrial membrane encloses the organelle, "
            "while the inner mitochondrial membrane contains numerous infoldings called cristae that expand the surface area for the electron transport chain."
        )
    }

    print("=" * 95)
    print(f"{'Domain / Concept Evaluated':<36} | {'Tokens':<6} | {'Min Logit':<9} | {'Max Logit':<9} | {'Loss (CE)':<10} | {'Perplexity':<10}")
    print("=" * 95)
    sys.stdout.flush()

    results = {}
    with torch.no_grad():
        for domain, text in domain_benchmarks.items():
            encoded = tokenizer.encode(text)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long)
            seq_len = input_ids.shape[1]

            logits, _ = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean").item()
            ppl = math.exp(loss)
            min_l = logits.min().item()
            max_l = logits.max().item()

            results[domain] = {
                "tokens": seq_len,
                "loss": loss,
                "ppl": ppl,
                "min_logit": min_l,
                "max_logit": max_l
            }

            print(f"{domain:<36} | {seq_len:<6d} | {min_l:<9.2f} | {max_l:<9.2f} | {loss:<10.4f} | {ppl:<10.2f}")
            sys.stdout.flush()

    print("=" * 95)

if __name__ == "__main__":
    main()

