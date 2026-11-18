import subprocess

adb = r".\platform-tools\adb.exe"
device = "RZCX61AJBKX"
full_query = "Give me 3 valuable pieces of life and career advice"

# Use triple quotes in shell command:
# am start -n com.embeddra.edge/.MainActivity --es query 'Give me 3 valuable pieces of life and career advice'
shell_cmd = f"am start -n com.embeddra.edge/.MainActivity --es query '{full_query}'"
print("Running command on device:", shell_cmd)
proc = subprocess.run([adb, "-s", device, "shell", shell_cmd], capture_output=True, text=True)
print("STDOUT:", proc.stdout)
print("STDERR:", proc.stderr)
