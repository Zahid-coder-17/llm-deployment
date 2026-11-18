import os
import sys
import time
import subprocess
import ctypes
import urllib.request
import urllib.parse
import json
from typing import List, Dict, Any, Optional
from PIL import ImageGrab
import win32gui
import win32con
import win32process
import psutil
import uiautomation as auto

user32 = ctypes.windll.user32

class WindowsInspector:
    def __init__(self):
        self.last_screenshot_path = ""

    def get_foreground_window_info(self) -> Dict[str, Any]:
        try:
            hwnd = win32gui.GetForegroundWindow()
            if not hwnd or not win32gui.IsWindow(hwnd):
                return {"hwnd": 0, "title": "Windows Desktop", "class": "Progman", "rect": [0, 0, 1920, 1080], "width": 1920, "height": 1080}
            
            title = win32gui.GetWindowText(hwnd)
            class_name = win32gui.GetClassName(hwnd)
            rect = win32gui.GetWindowRect(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            return {
                "hwnd": hwnd,
                "title": title if title else "(Untitled Window)",
                "class": class_name,
                "pid": pid,
                "rect": [rect[0], rect[1], rect[2], rect[3]],
                "width": rect[2] - rect[0],
                "height": rect[3] - rect[1]
            }
        except Exception:
            return {"hwnd": 0, "title": "Windows Desktop", "class": "Progman", "rect": [0, 0, 1920, 1080], "width": 1920, "height": 1080}

    def list_visible_windows(self) -> List[Dict[str, Any]]:
        windows = []
        def enum_handler(hwnd, extra):
            try:
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd).strip()
                    rect = win32gui.GetWindowRect(hwnd)
                    w = rect[2] - rect[0]
                    h = rect[3] - rect[1]
                    if title and w > 100 and h > 100:
                        windows.append({
                            "hwnd": hwnd,
                            "title": title,
                            "class": win32gui.GetClassName(hwnd),
                            "rect": [rect[0], rect[1], rect[2], rect[3]]
                        })
            except Exception:
                pass
        win32gui.EnumWindows(enum_handler, None)
        return windows

    def dump_screen_elements(self) -> str:
        """
        Dumps the active desktop hierarchy into Maria-1B prompt format.
        """
        fg = self.get_foreground_window_info()
        wins = self.list_visible_windows()
        
        lines = [f"[SCREEN: Windows Desktop | Active: {fg['title']}]"]
        lines.append(f"[1] ActiveWindow: \"{fg['title']}\" rect={fg['rect']}")
        
        idx = 2
        for w in wins[:8]:
            if w["title"] != fg["title"]:
                lines.append(f"[{idx}] Window: \"{w['title']}\" rect={w['rect']}")
                idx += 1
                
        return "\n".join(lines)

    def capture_screenshot(self, output_path: str) -> str:
        try:
            img = ImageGrab.grab()
            img.save(output_path, "PNG")
            self.last_screenshot_path = output_path
            return output_path
        except Exception as e:
            print(f"Screenshot error: {e}")
            return ""

    # === Real Multi-Domain Actuation & Executive Actions ===

    def search_web_and_extract(self, query: str) -> Dict[str, Any]:
        """
        1. Launches live browser search (Google).
        2. Queries live encyclopedic knowledge via Wikipedia API for instant facts.
        """
        q_encoded = urllib.parse.quote(query.strip())
        search_url = f"https://www.google.com/search?q={q_encoded}"
        subprocess.Popen(f"start {search_url}", shell=True)

        snippets = []
        try:
            wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={q_encoded}&format=json"
            req = urllib.request.Request(wiki_url, headers={"User-Agent": "YoriDesktopAssistant/2.0"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode())
                for item in data.get("query", {}).get("search", [])[:3]:
                    clean_snip = item["snippet"].replace('<span class="searchmatch">', "").replace("</span>", "").replace("&quot;", '"')
                    snippets.append(f"• {item['title']}: {clean_snip}")
        except Exception as e:
            snippets.append(f"Live web search launched in your default browser: {search_url}")

        return {
            "query": query,
            "browser_url": search_url,
            "facts": snippets
        }

    def whatsapp_send(self, contact: str, message: str) -> Dict[str, Any]:
        """
        Launches WhatsApp message directly via URI protocol.
        """
        msg_encoded = urllib.parse.quote(message.strip())
        wa_uri = f"whatsapp://send?text={msg_encoded}"
        web_url = f"https://web.whatsapp.com/send?text={msg_encoded}"
        try:
            subprocess.Popen(["cmd.exe", "/c", "start", "", wa_uri], shell=False)
        except Exception:
            pass
        return {
            "status": "OPENED_WHATSAPP",
            "contact": contact if contact else "Active/Selected Chat",
            "message": message,
            "uri": wa_uri,
            "web_url": web_url
        }

    def chatgpt_image_create(self, prompt: str) -> Dict[str, Any]:
        """
        Launches ChatGPT or Bing Image Creator with a detailed generative prompt.
        """
        p_encoded = urllib.parse.quote(prompt.strip())
        bing_url = f"https://www.bing.com/images/create?q={p_encoded}"
        subprocess.Popen(f"start {bing_url}", shell=True)
        return {
            "status": "OPENED_GENERATOR",
            "target": "Bing Image Creator / DALL-E",
            "prompt": prompt,
            "url": bing_url
        }

    def get_system_telemetry(self) -> Dict[str, Any]:
        """
        Gathers live laptop hardware telemetry: CPU, RAM, Battery, Disk.
        """
        ram = psutil.virtual_memory()
        cpu_pct = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage("C:")
        
        battery_info = "Desktop / No Battery"
        battery = psutil.sensors_battery()
        if battery:
            plugged_str = "Plugged In" if battery.power_plugged else "On Battery"
            battery_info = f"{battery.percent}% ({plugged_str})"

        return {
            "cpu": f"{cpu_pct}%",
            "ram_used_gb": f"{ram.used / (1024**3):.1f} GB / {ram.total / (1024**3):.1f} GB ({ram.percent}%)",
            "battery": battery_info,
            "disk_free_gb": f"{disk.free / (1024**3):.1f} GB free of {disk.total / (1024**3):.1f} GB",
            "os": "Windows 11 (build 26100)"
        }

    def create_note_file(self, filename: str, content: str) -> str:
        """
        Creates a text file and opens it in Notepad.
        """
        target_dir = r"c:\Users\YADKI ZAHID AHMED\Desktop\llm_deployment"
        file_path = os.path.join(target_dir, filename if filename.endswith(".txt") else f"{filename}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        subprocess.Popen(f"notepad.exe \"{file_path}\"", shell=True)
        return file_path

    def launch_app(self, app_name: str) -> bool:
        app_name_lower = app_name.lower().strip()
        
        # System URI Mapping
        uri_map = {
            "settings": "ms-settings:",
            "wifi": "ms-settings:network-wifi",
            "wi-fi": "ms-settings:network-wifi",
            "bluetooth": "ms-settings:bluetooth",
            "display": "ms-settings:display",
            "sound": "ms-settings:sound",
            "volume": "ms-settings:sound",
            "battery": "ms-settings:batterysaver",
            "apps": "ms-settings:appsfeatures"
        }
        
        for k, uri in uri_map.items():
            if k in app_name_lower:
                subprocess.Popen(f"start {uri}", shell=True)
                return True
                
        # Known Windows Applications
        app_exec_map = {
            "chrome": "start chrome",
            "edge": "start msedge",
            "browser": "start msedge",
            "whatsapp": "start whatsapp:",
            "spotify": "start spotify:",
            "notepad": "notepad.exe",
            "calculator": "calc.exe",
            "calc": "calc.exe",
            "terminal": "wt.exe",
            "cmd": "cmd.exe",
            "powershell": "powershell.exe",
            "explorer": "explorer.exe",
            "files": "explorer.exe",
            "vscode": "code",
            "paint": "mspaint.exe",
            "taskmgr": "taskmgr.exe",
            "task manager": "taskmgr.exe"
        }
        
        for k, cmd in app_exec_map.items():
            if k in app_name_lower:
                subprocess.Popen(cmd, shell=True)
                return True
                
        # Fallback to direct Windows start
        subprocess.Popen(f"start {app_name}", shell=True)
        return True

    def close_active_window(self) -> bool:
        hwnd = win32gui.GetForegroundWindow()
        if hwnd:
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            return True
        return False

    def minimize_all_show_desktop(self) -> bool:
        # Win+D
        user32.keybd_event(0x5B, 0, 0, 0) # Win
        user32.keybd_event(0x44, 0, 0, 0) # D
        time.sleep(0.05)
        user32.keybd_event(0x44, 0, 2, 0)
        user32.keybd_event(0x5B, 0, 2, 0)
        return True

    def lock_workstation(self) -> bool:
        ctypes.windll.user32.LockWorkStation()
        return True

    def click_at(self, x: int, y: int) -> bool:
        user32.SetCursorPos(x, y)
        time.sleep(0.05)
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
        time.sleep(0.05)
        user32.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        return True

    def type_text(self, text: str) -> bool:
        for char in text:
            vk = user32.VkKeyScanW(ord(char))
            if vk != -1:
                shift = (vk >> 8) & 1
                key_code = vk & 0xFF
                if shift:
                    user32.keybd_event(win32con.VK_SHIFT, 0, 0, 0)
                user32.keybd_event(key_code, 0, 0, 0)
                time.sleep(0.02)
                user32.keybd_event(key_code, 0, win32con.KEYEVENTF_KEYUP, 0)
                if shift:
                    user32.keybd_event(win32con.VK_SHIFT, 0, win32con.KEYEVENTF_KEYUP, 0)
        return True
