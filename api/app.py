"""
api/app.py — FastAPI application entry point for PRISM AI Module.

Starts the server, connects to MongoDB, registers all route files.

Run:
    uvicorn api.app:app --host 0.0.0.0 --port 8001 --reload
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from db.mongo_client import connect, close

# Route imports
from api.routes import (
    batches,
    records,
    sentiment,
    engagement,
    crisis,
    reports,
    alerts,
    jobs,
)

load_dotenv()


# ── Lifespan: startup + shutdown ────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to MongoDB
    await connect()
    print("✅ Connected to MongoDB")
    yield
    # Shutdown: close MongoDB connection
    await close()
    print("🛑 MongoDB connection closed")


# ── App init ────────────────────────────────────────────────────────

app = FastAPI(
    title="PRISM AI Module",
    description="Sentiment intelligence pipeline for the Indian film/celebrity ecosystem. "
                "Receives raw data from Team 1, processes through a 5-phase pipeline, "
                "and exposes analyzed results via internal API.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── CORS ────────────────────────────────────────────────────────────
# POC: allow all origins. Restrict in production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Register routers ───────────────────────────────────────────────

app.include_router(batches.router,     prefix="/api/v1", tags=["Batches"])
app.include_router(records.router,     prefix="/api/v1", tags=["Records"])
app.include_router(sentiment.router,   prefix="/api/v1", tags=["Sentiment"])
app.include_router(engagement.router,  prefix="/api/v1", tags=["Engagement"])
app.include_router(crisis.router,      prefix="/api/v1", tags=["Crisis"])
app.include_router(reports.router,     prefix="/api/v1", tags=["Reports"])
app.include_router(alerts.router,      prefix="/api/v1", tags=["Alerts"])
app.include_router(jobs.router,        prefix="/api/v1", tags=["Jobs"])


# ── Health check ────────────────────────────────────────────────────

@app.get("/api/v1/health", tags=["Health"])
async def health_check():
    """Basic health check — confirms API and DB are running."""
    from db.mongo_client import get_db
    try:
        db = get_db()
        # Ping MongoDB to verify connection is alive
        await db.command("ping")
        return {"status": "ok", "db": "ok"}
    except Exception:
        return {"status": "ok", "db": "error"}