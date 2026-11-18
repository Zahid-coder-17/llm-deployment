import os
import sys
import types
import torch
from tokenizers import Tokenizer

BASE_DIR = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
sys.path.insert(0, os.path.join(BASE_DIR, "embeddra-edge"))
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
import verify_eer_smoke as v

m = types.ModuleType("model")
m.MariaConfig = MariaConfig
sys.modules["model"] = m

EER_PATH = os.path.join(BASE_DIR, "maria_1b_android_executive_merged.eer")
TOK_PATH = os.path.join(BASE_DIR, "checkpoints", "tokenizer.json")
from train_pristine_unified_yori import SYSTEM_CONTRACT

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_file(TOK_PATH)
    sd = v.load_state_dict_from_eer(EER_PATH)
    model = MariaForCausalLM(MariaConfig()).to(device=device, dtype=torch.float16)
    model.load_state_dict(sd, strict=False)
    model.eval()

    prompt = f"{SYSTEM_CONTRACT}<user>Give me 3 valuable pieces of life and career advice</user>\n<thought>\n"
    ids = tokenizer.encode(prompt).ids
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        logits, _ = model(inp)
        top_k = torch.topk(logits[0, -1, :], 10)
        print("Top 10 tokens at start of thought:")
        for idx, val in zip(top_k.indices, top_k.values):
            tok_str = tokenizer.decode([idx.item()])
            print(f"  token_id={idx.item():6d} | logit={val.item():.2f} | text='{tok_str}'")

if __name__ == "__main__":
    main()
