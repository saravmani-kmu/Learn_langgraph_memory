"""FastAPI routes for the Stock Knowledge Graph API."""

from __future__ import annotations

from collections import defaultdict

from fastapi import APIRouter
from langchain_core.messages import AIMessage, HumanMessage

from app.db.neo4j_client import run_cypher
from app.graph.collect_graph import collect_graph
from app.graph.query_graph import query_graph
from app.models.schemas import (
    CollectRequest,
    CollectResponse,
    QueryRequest,
    QueryResponse,
    StockInfo,
)

router = APIRouter()

# ── In-memory session store (maps session_id → message history) ─────────────
_sessions: dict[str, list] = defaultdict(list)


# ── POST /collect ───────────────────────────────────────────────────────────────

@router.post("/collect", response_model=CollectResponse)
async def collect_stock(req: CollectRequest):
    """
    Multi-turn conversational endpoint to collect stock information.

    Send successive messages with the same `session_id` to continue the
    conversation.  When all details are gathered the data is automatically
    stored in Neo4j.
    """
    # Append the user message to session history
    _sessions[req.session_id].append(HumanMessage(content=req.user_message))

    # Invoke the collect graph
    state = collect_graph.invoke({
        "messages": list(_sessions[req.session_id]),
        "extracted": None,
        "stored": False,
    })

    # Capture the latest assistant reply
    assistant_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
    assistant_reply = assistant_msgs[-1].content if assistant_msgs else ""

    # Persist assistant reply in session
    _sessions[req.session_id].append(AIMessage(content=assistant_reply))

    # If stored, clear the session so the user can start a new collection
    extracted: StockInfo | None = state.get("extracted")
    stored: bool = state.get("stored", False)

    if stored:
        del _sessions[req.session_id]

    return CollectResponse(
        assistant_message=assistant_reply,
        extracted=extracted,
        stored=stored,
    )


# ── POST /query ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_stocks(req: QueryRequest):
    """
    Ask a natural-language question about the stock knowledge graph.
    The LLM generates a Cypher query, executes it, and returns a
    synthesised answer.
    """
    state = query_graph.invoke({
        "question": req.question,
        "cypher": "",
        "result": "",
        "answer": "",
    })

    return QueryResponse(
        answer=state.get("answer", ""),
        cypher_used=state.get("cypher", ""),
    )


# ── GET /stocks ─────────────────────────────────────────────────────────────────

@router.get("/stocks")
async def list_stocks():
    """Return all stocks with their sectors and linked stocks from Neo4j."""
    query = """
    MATCH (s:Stock)
    OPTIONAL MATCH (s)-[:BELONGS_TO]->(sec:Sector)
    OPTIONAL MATCH (s)-[:LINKED_TO]->(ls:Stock)
    RETURN s.name AS stock,
           sec.name AS sector,
           collect(DISTINCT ls.name) AS linked_stocks
    ORDER BY s.name
    """
    records = run_cypher(query)
    return {"stocks": records}
