import os
import json
import time
import random
import requests

OUT_DIR = r"D:\hf_datasets"
os.makedirs(OUT_DIR, exist_ok=True)
RAW_PATH = os.path.join(OUT_DIR, "rico_sca_raw.jsonl")
SFT_PATH = os.path.join(OUT_DIR, "yori_rico_label_ui.jsonl")

def extract_nodes(node, out_list):
    if not isinstance(node, dict):
        return
    vis = node.get("visibility", "visible")
    if vis != "visible":
        return

    text = (node.get("text") or "").strip()
    desc_raw = node.get("content-desc") or []
    desc = ""
    if isinstance(desc_raw, list):
        for d in desc_raw:
            if isinstance(d, str) and d.strip():
                desc = d.strip()
                break
    elif isinstance(desc_raw, str):
        desc = desc_raw.strip()

    label = text if text else desc
    cls_full = node.get("class") or (node.get("ancestors") or ["View"])[0]
    cls_name = str(cls_full).split(".")[-1]
    clickable = bool(node.get("clickable", False))
    checkable = "Switch" in cls_name or "CheckBox" in cls_name or "Toggle" in cls_name
    editable = "EditText" in cls_name or bool(node.get("editable", False))

    if label and 2 <= len(label) <= 40 and "\n" not in label and '"' not in label:
        if editable:
            t_str = "EditText"
            st = " [Input]"
        elif checkable:
            t_str = "Switch"
            st = " (ON)" if node.get("checked", False) else " (OFF)"
        elif clickable:
            t_str = "Button"
            st = ""
        else:
            t_str = cls_name
            st = ""
        out_list.append((t_str, label, st, editable, clickable or checkable))

    for ch in node.get("children") or []:
        extract_nodes(ch, out_list)

def main():
    print("Fetching 300 real Android screens from bevaya/RICO-SCA into D:\\hf_datasets...", flush=True)
    rows = []
    for offset in range(0, 300, 20):
        url = f"https://datasets-server.huggingface.co/rows?dataset=bevaya/RICO-SCA&config=default&split=train&offset={offset}&length=20"
        data = None
        for attempt in range(4):
            try:
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                data = resp.json().get("rows", [])
                break
            except Exception as e:
                print(f"  Retry {attempt+1} at offset={offset}: {e}", flush=True)
                time.sleep(2)
        if not data:
            print(f"  Skipping offset={offset} after 4 attempts.", flush=True)
            continue
        for item in data:
            r = item["row"]
            rows.append({
                "pkg": r.get("app_package_name", "com.android.app"),
                "vh": r.get("view_hierarchy", "")
            })
        print(f"  Fetched {len(rows)}/300 screens...", flush=True)

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    random.seed(42)
    sample_texts = ["Yori Lunch", "alex@email.com", "Hello there", "Bluetooth", "Meeting 3 PM", "Search notes"]
    click_verbs = ["Tap", "Click", "Open", "Select", "Press"]

    sft_examples = []
    for r in rows:
        try:
            vh = json.loads(r["vh"]) if isinstance(r["vh"], str) else r["vh"]
            root = vh.get("activity", {}).get("root")
            if not root:
                continue
            nodes = []
            extract_nodes(root, nodes)
            # Deduplicate by label
            seen = set()
            unique_nodes = []
            for n in nodes:
                if n[1].lower() not in seen:
                    seen.add(n[1].lower())
                    unique_nodes.append(n)
            if len(unique_nodes) < 4:
                continue
            unique_nodes = unique_nodes[:18]

            screen_lines = [f"[SCREEN: {r['pkg']}]"]
            for idx, (t_str, lbl, st, _, _) in enumerate(unique_nodes, 1):
                screen_lines.append(f'[{idx}] {t_str} "{lbl}"{st}')
            screen_str = "\n".join(screen_lines)

            # Pick 1 interactive target (prefer EditText or clickable)
            editables = [n for n in unique_nodes if n[3]]
            clickables = [n for n in unique_nodes if n[4] and not n[3]]

            if editables and random.random() < 0.4:
                _, lbl, _, _, _ = random.choice(editables)
                val = random.choice(sample_texts)
                intent = f"Type '{val}' into {lbl}"
                thought = f'I will type "{val}" into "{lbl}".'
                action = f'TYPE("{lbl}", "{val}")'
            elif clickables:
                _, lbl, _, _, _ = random.choice(clickables)
                intent = f"{random.choice(click_verbs)} {lbl}"
                thought = f'I will click "{lbl}".'
                action = f'CLICK("{lbl}")'
            else:
                _, lbl, _, _, _ = random.choice(unique_nodes)
                intent = f"Select {lbl}"
                thought = f'I will click "{lbl}".'
                action = f'CLICK("{lbl}")'

            sft_examples.append({
                "category": "ui_grounding",
                "prompt": f"<user>{intent}</user>\n<screen>\n{screen_str}\n</screen>\n<thought>",
                "target": f"{thought}</thought>\n<action>{action}</action>"
            })
        except Exception:
            continue

    with open(SFT_PATH, "w", encoding="utf-8") as f:
        for ex in sft_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Saved {len(sft_examples)} real Android RICO-SCA SFT examples to {SFT_PATH}!", flush=True)

if __name__ == "__main__":
    main()

