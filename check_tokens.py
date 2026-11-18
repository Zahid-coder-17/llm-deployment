from tokenizers import Tokenizer
tok = Tokenizer.from_file('checkpoints/tokenizer.json')
enc = tok.encode('Agentic AI lecture')
print('Tokens for Agentic AI lecture:', enc.tokens, enc.ids)
for t, i in zip(enc.tokens, enc.ids):
    print(f'{i}: {t!r}')

enc2 = tok.encode('stanford agentic ai lecture')
print('\nTokens for stanford agentic ai lecture:', enc2.tokens, enc2.ids)
for t, i in zip(enc2.tokens, enc2.ids):
    print(f'{i}: {t!r}')

