"""
ResearchIQ — Multi-Agent Research Assistant
3 agents: Researcher → Critic → Writer
Powered by Groq (free)
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ── Groq ──────────────────────────────────────────────────────────────────────
try:
    from groq import Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    groq_client = Groq(api_key=GROQ_API_KEY)
    GROQ_AVAILABLE = True
except Exception as e:
    print(f"Groq not available: {e}")
    GROQ_AVAILABLE = False
    groq_client = None

# ── Web search (optional) ─────────────────────────────────────────────────────
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

app = FastAPI(title="ResearchIQ API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── In-memory store ───────────────────────────────────────────────────────────
research_sessions: dict[str, dict] = {}
connected_clients: dict[str, list[WebSocket]] = {}

# ── Pydantic ──────────────────────────────────────────────────────────────────
class ResearchRequest(BaseModel):
    topic: str
    depth: str = "standard"  # quick | standard | deep

# ── Groq helper ───────────────────────────────────────────────────────────────
def call_groq(system: str, user: str, max_tokens: int = 1500) -> str:
    if not GROQ_AVAILABLE or not groq_client:
        return "AI not available — set GROQ_API_KEY"
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# ── AGENT 1: Researcher ───────────────────────────────────────────────────────
async def researcher_agent(topic: str, depth: str) -> dict:
    """Gathers key facts, angles, and subtopics on the given topic."""
    detail = {"quick": "3-4", "standard": "5-6", "deep": "7-8"}[depth]

    system = """You are a Research Agent. Your job is to gather comprehensive, 
accurate information on any topic. You identify key facts, important angles, 
current developments, and relevant subtopics. Be thorough and factual."""

    user = f"""Research this topic thoroughly: {topic}

Provide:
1. Overview (2-3 sentences)
2. {detail} key facts or findings
3. {detail} important subtopics or angles to explore
4. Current relevance and why this matters
5. Any controversies or debates in this space

Be specific, factual, and comprehensive."""

    result = call_groq(system, user, max_tokens=1200)
    return {
        "agent": "Researcher",
        "status": "complete",
        "output": result,
        "timestamp": datetime.now().isoformat()
    }

# ── AGENT 2: Critic ───────────────────────────────────────────────────────────
async def critic_agent(topic: str, research: str) -> dict:
    """Reviews research for gaps, biases, and missing angles."""

    system = """You are a Critical Analysis Agent. Your job is to review research 
and identify gaps, potential biases, missing perspectives, and areas that need 
deeper exploration. You ensure balanced, complete coverage."""

    user = f"""Topic: {topic}

Research to review:
{research}

Critically analyze this research:
1. What important angles or perspectives are MISSING?
2. Are there any potential biases in the framing?
3. What counterarguments or opposing views should be included?
4. What additional context would strengthen this?
5. Rate completeness 1-10 and explain why
6. List 3 specific improvements needed

Be constructively critical and specific."""

    result = call_groq(system, user, max_tokens=900)
    return {
        "agent": "Critic",
        "status": "complete",
        "output": result,
        "timestamp": datetime.now().isoformat()
    }

# ── AGENT 3: Writer ───────────────────────────────────────────────────────────
async def writer_agent(topic: str, research: str, critique: str, depth: str) -> dict:
    """Synthesizes research and critique into a polished report."""

    length = {"quick": "400-500", "standard": "600-800", "deep": "900-1100"}[depth]

    system = """You are a Professional Writer Agent. Your job is to synthesize 
research and critical feedback into clear, compelling, well-structured reports. 
You write for an intelligent general audience — clear, insightful, and engaging."""

    user = f"""Write a comprehensive research report on: {topic}

Based on this research:
{research}

Incorporating these critical improvements:
{critique}

Write a {length} word report with:
- Compelling title
- Executive Summary (2-3 sentences)
- Main Body with clear sections and headers
- Key Takeaways (3-5 bullet points)
- Conclusion

Make it professional, clear, and genuinely useful. Use markdown formatting."""

    result = call_groq(system, user, max_tokens=1500)
    return {
        "agent": "Writer",
        "status": "complete",
        "output": result,
        "timestamp": datetime.now().isoformat()
    }

# ── Broadcast ─────────────────────────────────────────────────────────────────
async def broadcast(session_id: str, event: str, data: dict):
    dead = []
    for ws in connected_clients.get(session_id, []):
        try:
            await ws.send_json({"event": event, "data": data})
        except Exception:
            dead.append(ws)
    for ws in dead:
        connected_clients[session_id].remove(ws)

# ── Main pipeline ─────────────────────────────────────────────────────────────
async def run_research_pipeline(session_id: str, topic: str, depth: str):
    session = research_sessions[session_id]
    session["status"] = "running"

    try:
        # Stage 1: Researcher
        await broadcast(session_id, "agent_start", {
            "agent": "Researcher",
            "message": f"Researching: {topic}...",
            "step": 1, "total": 3
        })
        research = await researcher_agent(topic, depth)
        session["research"] = research
        await broadcast(session_id, "agent_complete", research)
        await asyncio.sleep(0.5)

        # Stage 2: Critic
        await broadcast(session_id, "agent_start", {
            "agent": "Critic",
            "message": "Analyzing research for gaps and biases...",
            "step": 2, "total": 3
        })
        critique = await critic_agent(topic, research["output"])
        session["critique"] = critique
        await broadcast(session_id, "agent_complete", critique)
        await asyncio.sleep(0.5)

        # Stage 3: Writer
        await broadcast(session_id, "agent_start", {
            "agent": "Writer",
            "message": "Synthesizing final report...",
            "step": 3, "total": 3
        })
        report = await writer_agent(topic, research["output"], critique["output"], depth)
        session["report"] = report
        await broadcast(session_id, "agent_complete", report)

        session["status"] = "complete"
        session["completed_at"] = datetime.now().isoformat()
        await broadcast(session_id, "pipeline_complete", {
            "session_id": session_id,
            "topic": topic,
            "report": report["output"]
        })

    except Exception as e:
        session["status"] = "error"
        await broadcast(session_id, "error", {"message": str(e)})

# ── REST Endpoints ─────────────────────────────────────────────────────────────
@app.post("/api/research")
async def start_research(req: ResearchRequest):
    session_id = str(uuid.uuid4())
    research_sessions[session_id] = {
        "session_id": session_id,
        "topic": req.topic,
        "depth": req.depth,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "research": None,
        "critique": None,
        "report": None,
    }
    connected_clients[session_id] = []
    asyncio.create_task(run_research_pipeline(session_id, req.topic, req.depth))
    return {"session_id": session_id, "topic": req.topic}

@app.get("/api/research/{session_id}")
async def get_research(session_id: str):
    if session_id not in research_sessions:
        return {"error": "Session not found"}
    return research_sessions[session_id]

@app.get("/api/research")
async def list_research():
    return list(research_sessions.values())

@app.get("/api/health")
async def health():
    return {"status": "ok", "groq_available": GROQ_AVAILABLE, "agents": ["Researcher", "Critic", "Writer"]}

# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    connected_clients.setdefault(session_id, []).append(websocket)
    try:
        await websocket.send_json({"event": "connected", "data": {"session_id": session_id}})
        while True:
            try:
                await asyncio.wait_for(websocket.receive(), timeout=60.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"event": "ping", "data": {}})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")
    finally:
        clients = connected_clients.get(session_id, [])
        if websocket in clients:
            clients.remove(websocket)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
