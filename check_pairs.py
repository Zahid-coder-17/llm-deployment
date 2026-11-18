import json

with open("hf_cache/yori_unified_train.jsonl", "r", encoding="utf-8") as f:
    for i in range(10):
        line = f.readline()
        if not line: break
        obj = json.loads(line)
        print(f"[{i}] CAT={obj.get('category')}")
        print(f"  P: {repr(obj['prompt'])}")
        print(f"  T: {repr(obj['target'][:80])}")
        print()
