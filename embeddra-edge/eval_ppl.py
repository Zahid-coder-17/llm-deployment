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
    checkpoint_path = os.path.join(workspace_root, "checkpoints", "maria_1b_20B_weights.pt")
    tokenizer_path = os.path.join(workspace_root, "checkpoints", "tokenizer.json")

    print(f"Loading Tokenizer from: {tokenizer_path}")
    sys.stdout.flush()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(f"[OK] Tokenizer loaded. Vocab size: {tokenizer.get_vocab_size():,}")
    sys.stdout.flush()

    print(f"Loading Model from: {checkpoint_path}")
    sys.stdout.flush()
    t0 = time.time()
    model = MariaForCausalLM.from_pretrained_checkpoint(checkpoint_path, device="cpu", dtype=torch.bfloat16)
    print(f"[OK] Model loaded in {time.time() - t0:.2f}s")
    sys.stdout.flush()

    test_corpora = {
        "PubMed_Biomedical": (
            "The human epidermal growth factor receptor 2 (HER2) is a member of the epidermal growth factor receptor family of receptor tyrosine kinases. "
            "Amplification or overexpression of this oncogene has been shown to play an important role in the development and progression of certain aggressive types of breast cancer. "
            "Trastuzumab is a monoclonal antibody that targets the extracellular domain of the HER2 protein, inhibiting cell proliferation and promoting antibody-dependent cellular cytotoxicity."
        ),
        "Clinical_QA": (
            "Question: A 54-year-old male presents with persistent cough for four weeks with hemoptysis and unexplained weight loss. What is the recommended urgent investigation?\n"
            "Answer: Under national guidelines, an urgent chest X-ray within two weeks is indicated for patients aged forty and over with unexplained hemoptysis or persistent cough."
        ),
        "ComputerScience_arXiv": (
            "Stochastic gradient descent is an iterative optimization algorithm commonly applied to objective functions with the structure of a sum of differentiable functions. "
            "Both modern statistical estimation and machine learning problems consider minimizing an objective function that represents an empirical risk over a given dataset. "
            "Hyperparameter tuning, including learning rate scheduling and weight decay regularization, significantly influences convergence rates and generalization properties."
        ),
        "General_Biology": (
            "Cellular respiration is a set of metabolic reactions and processes that take place in the cells of organisms to convert biochemical energy from nutrients into adenosine triphosphate. "
            "The catabolic reactions involved in respiration include glycolysis, the citric acid cycle, and oxidative phosphorylation occurring across the mitochondrial inner membrane."
        )
    }

    print("\n" + "=" * 80)
    print(f"{'Corpus / Domain':<35} | {'Tokens':<6} | {'Loss (CE)':<10} | {'Perplexity (PPL)':<16}")
    print("=" * 80)
    sys.stdout.flush()

    model.eval()
    with torch.no_grad():
        for name, text in test_corpora.items():
            encoded = tokenizer.encode(text)
            input_ids = torch.tensor([encoded.ids], dtype=torch.long)
            seq_len = input_ids.shape[1]

            logits, _ = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                reduction="mean"
            )
            ppl = math.exp(loss.item())
            print(f"{name:<35} | {seq_len:<6d} | {loss.item():<10.4f} | {ppl:<16.2f}")
            sys.stdout.flush()

    print("=" * 80)

if __name__ == "__main__":
    main()

