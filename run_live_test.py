import subprocess
import time
import sys
import urllib.parse

ADB = r"C:\Android\Sdk\platform-tools\adb.exe"
DEVICE = "RZCX61AJBKX"

def run_adb(args):
    cmd = [ADB, "-s", DEVICE] + args
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    return res.stdout.strip()

def clear_logcat():
    run_adb(["logcat", "-c"])

def send_query(query: str):
    print(f"\n=======================================================")
    print(f">> SENDING USER QUERY: '{query}'")
    print(f"=======================================================")
    clear_logcat()
    # In adb shell, wrapping with escaped quotes passes the entire string intact to Android am
    escaped = f"\\\"{query}\\\""
    run_adb(["shell", "am", "start", "-n", "com.embeddra.edge/.MainActivity", "--es", "query", escaped])

def stream_logs(timeout_sec=16):
    t0 = time.time()
    seen = set()
    while time.time() - t0 < timeout_sec:
        out = run_adb(["logcat", "-d", "-s", "MainActivity:I", "YoriAccessibilityService:I"])
        lines = out.splitlines()
        for line in lines:
            if line not in seen:
                seen.add(line)
                if any(k in line for k in [">> RECEIVED QUERY", ">> PROMPT TO LLM", "Turn=", "output="]):
                    print(f"  [LOG] {line}")
        time.sleep(0.8)

if __name__ == "__main__":
    test_queries = [
        "Open WhatsApp",
        "find Agentic AI lecture and play it from youtube",
        "Open ChatGPT and write a message about HyperGQE",
        "Open Wi-Fi Settings"
    ]
    if len(sys.argv) > 1:
        test_queries = [sys.argv[1]]
    for q in test_queries:
        send_query(q)
        stream_logs(timeout_sec=16)
        time.sleep(2)
