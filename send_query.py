import sys
import subprocess

query = sys.argv[1] if len(sys.argv) > 1 else "Give me 3 valuable pieces of life and career advice"
adb_path = r".\platform-tools\adb.exe"
device = "RZCX61AJBKX"

# In Android am start, escaping spaces with \ inside shell string passes the exact string
escaped = query.replace(" ", "\\ ")
cmd = f"am start -n com.embeddra.edge/.MainActivity --es query {escaped}"
print("Executing:", cmd)
res = subprocess.run([adb_path, "-s", device, "shell", cmd], capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("Stderr:", res.stderr)
