import os
import sys
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Ensure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from desktop_agent import YoriDesktopAgent

app = FastAPI(title="YORI - Desktop Executive Assistant", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Desktop Agent
agent = YoriDesktopAgent()

@app.post("/api/execute")
async def execute(req: Request):
    data = await req.json()
    query = data.get("query", "").strip()
    if not query:
        return {"error": "Empty query", "summary": "No query provided"}
    return agent.process_query(query)

@app.get("/api/state")
async def get_state():
    fg = agent.inspector.get_foreground_window_info()
    wins = agent.inspector.list_visible_windows()
    return {
        "status": "ready",
        "active_window": fg,
        "visible_windows_count": len(wins),
        "device": str(agent.device),
        "model": "Maria-1B-Executive"
    }

# Mount static web UI
static_dir = os.path.join(SCRIPT_DIR, "static")
app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8091, reload=False)
