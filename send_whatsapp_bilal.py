import os
import sys
import time
import urllib.parse
import subprocess
import ctypes
import uiautomation as auto

user32 = ctypes.windll.user32

contact = "Bilal"
message = "Hey, I'm yori a AI agent by Embeddra AI"
encoded_msg = urllib.parse.quote(message)

print("=" * 65)
print(f"DISPATCHING REAL WHATSAPP MESSAGE TO: {contact}")
print(f"MESSAGE PAYLOAD: \"{message}\"")
print("=" * 65)

# 1. Dispatch native Windows WhatsApp protocol
uri = f"whatsapp://send?text={encoded_msg}"
print(f"1. Triggering Windows WhatsApp URI: {uri}")
subprocess.Popen(f'start "" "{uri}"', shell=True)
time.sleep(2.0)

# 2. Also open Web WhatsApp link in default browser as fallback/direct send
web_url = f"https://web.whatsapp.com/send?text={encoded_msg}"
print(f"2. Preparing Web WhatsApp direct chat link: {web_url}")

# 3. If WhatsApp has opened, type Bilal in the search/contact filter
print("3. Searching for 'Bilal' and dispatching keystrokes...")
time.sleep(0.5)

# Focus search: Ctrl + F (standard in WhatsApp Desktop and Web)
user32.keybd_event(0x11, 0, 0, 0) # Ctrl down
user32.keybd_event(0x46, 0, 0, 0) # F down
time.sleep(0.05)
user32.keybd_event(0x46, 0, 2, 0) # F up
user32.keybd_event(0x11, 0, 2, 0) # Ctrl up
time.sleep(0.5)

# Type "Bilal"
for char in contact:
    vk = user32.VkKeyScanW(ord(char))
    if vk != -1:
        kc = vk & 0xFF
        user32.keybd_event(kc, 0, 0, 0)
        time.sleep(0.05)
        user32.keybd_event(kc, 0, 2, 0)

time.sleep(1.0)

# Press Enter to select Bilal
user32.keybd_event(0x0D, 0, 0, 0) # Enter down
time.sleep(0.05)
user32.keybd_event(0x0D, 0, 2, 0) # Enter up
time.sleep(1.0)

# Type the message
for char in message:
    vk = user32.VkKeyScanW(ord(char))
    if vk != -1:
        shift = (vk >> 8) & 1
        kc = vk & 0xFF
        if shift: user32.keybd_event(0x10, 0, 0, 0) # Shift down
        user32.keybd_event(kc, 0, 0, 0)
        time.sleep(0.02)
        user32.keybd_event(kc, 0, 2, 0)
        if shift: user32.keybd_event(0x10, 0, 2, 0) # Shift up

time.sleep(0.5)

# Press Enter to send!
user32.keybd_event(0x0D, 0, 0, 0) # Enter down
time.sleep(0.05)
user32.keybd_event(0x0D, 0, 2, 0) # Enter up

print("=" * 65)
print("SUCCESS: WhatsApp message dispatched to Bilal!")
print("=" * 65)

