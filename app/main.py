"""
FastAPI application entry point.

Run with:
    uvicorn app.main:app --reload
"""

from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

# Load .env BEFORE any other imports that read env vars
load_dotenv()

from app.api.routes import router  # noqa: E402
from app.db.neo4j_client import close_driver, get_driver  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle for the FastAPI application."""
    # Startup: validate Neo4j connection
    driver = get_driver()
    driver.verify_connectivity()
    print("✅  Connected to Neo4j")
    yield
    # Shutdown: close Neo4j driver
    close_driver()
    print("🛑  Neo4j driver closed")


app = FastAPI(
    title="Stock Knowledge Graph API",
    description=(
        "Conversational API powered by LangGraph + Groq LLM + Neo4j.\n\n"
        "• **POST /collect** – Multi-turn conversation to gather stock info and store it in the graph.\n"
        "• **POST /query** – Ask questions about stocks; Cypher is generated automatically.\n"
        "• **GET /stocks** – List all stored stocks."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
