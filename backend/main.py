"""FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

from .core.database import run_migrations
from .routers import admin, auth, chat, conversations, feedback, files, memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000",
).split(",")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run DB migrations on startup."""
    await run_migrations()
    logger.info("FastAPI backend ready")
    yield


app = FastAPI(
    title="AI Financial Analyst API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _ALLOWED_ORIGINS],
    allow_credentials=True,  # Required for httpOnly cookie auth
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(memory.router)
app.include_router(feedback.router)
app.include_router(files.router)
app.include_router(admin.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
