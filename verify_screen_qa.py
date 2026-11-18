import sys
import torch
sys.path.insert(0, r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment")
from verify_eer_smoke import load_state_dict_from_eer, generate_from_eer, EER_PATH, TOK_PATH
from hypergqe.model_adapter.maria_model import MariaForCausalLM, MariaConfig
from tokenizers import Tokenizer

sd = load_state_dict_from_eer(EER_PATH)
m = MariaForCausalLM(MariaConfig()).to("cuda", dtype=torch.float16)
m.load_state_dict(sd, strict=False)
m.eval()
tok = Tokenizer.from_file(TOK_PATH)

tests = [
    ("Summarize", '<user>summarize what is on my screen</user>\n<screen>\n[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] TextView "Hey, are we still meeting at 4 PM?"\n[3] EditText "Message" [Input]\n</screen>\n<thought>', True),
    ("WhatsHappening", '<user>tell me what\'s happening on screen</user>\n<screen>\n[SCREEN: com.google.android.youtube]\n[1] TextView "Agentic AI Lecture - Full Course"\n[2] Button "Subscribe"\n</screen>\n<thought>', True),
    ("Reply", '<user>reply I am on my way</user>\n<screen>\n[SCREEN: com.whatsapp]\n[1] TextView "Bilal"\n[2] EditText "Message" [Input]\n[3] Button "Send"\n</screen>\n<thought>', False),
    ("Scroll", '<user>scroll down</user>\n<thought>', False),
    ("Screenshot", '<user>take a screenshot</user>\n<thought>', False),
]
for name, p, s in tests:
    out = generate_from_eer(m, tok, p, "cuda", max_tokens=60, stop_on_thought=s).replace("\n", " ")
    print(f"[{name}] {out}")

