"""
LangGraph workflow – Collect stock information through conversation.

Flow: chat → extract → (store | END)
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.db.neo4j_client import upsert_stock
from app.models.schemas import StockInfo

# ── LLM setup ──────────────────────────────────────────────────────────────────

_llm = None


def _get_llm():
    """Lazily initialise ChatGroq so env vars are loaded first."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm

SYSTEM_PROMPT = """\
You are a helpful financial assistant that collects stock information from the user.
Your goal is to gather THREE pieces of information:
1. **Stock Name** – the name or ticker symbol (e.g. TCS, RELIANCE, INFY).
2. **Sector Name** – the sector it belongs to (e.g. IT, Banking, Pharma, Auto).
3. **Linked Stock Names** – other stocks that are related or competitors (at least one).

Ask the user conversationally. Ask for one piece at a time if needed.
When you believe you have ALL three pieces, confirm them with the user in your reply.
Always be concise and friendly.
"""

# ── State ───────────────────────────────────────────────────────────────────────


class CollectState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    extracted: StockInfo | None
    stored: bool


# ── Nodes ───────────────────────────────────────────────────────────────────────


def chat_node(state: CollectState) -> dict:
    """Send the conversation so far to the LLM and get an assistant reply."""
    messages = state["messages"]
    llm = _get_llm()

    # Prepend system prompt if this is the first turn
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

    response: AIMessage = llm.invoke(messages)
    return {"messages": [response]}


def extract_node(state: CollectState) -> dict:
    """Attempt to extract StockInfo from the full conversation."""
    structured_llm = _get_llm().with_structured_output(StockInfo)

    extraction_prompt = [
        SystemMessage(
            content=(
                "Extract stock information from the conversation below. "
                "Return the stock_name, sector, and linked_stocks if ALL are present. "
                "If any piece is missing, return stock_name as empty string to signal incomplete."
            )
        ),
        *state["messages"],
    ]

    try:
        info: StockInfo = structured_llm.invoke(extraction_prompt)
        # Treat empty stock_name as "not enough info yet"
        if info.stock_name.strip() and info.sector.strip() and len(info.linked_stocks) > 0:
            return {"extracted": info}
    except Exception:
        pass

    return {"extracted": None}


def store_node(state: CollectState) -> dict:
    """Persist extracted stock data into Neo4j."""
    info: StockInfo = state["extracted"]  # type: ignore[assignment]
    upsert_stock(
        stock_name=info.stock_name,
        sector=info.sector,
        linked_stocks=info.linked_stocks,
    )
    return {"stored": True}


# ── Edges ───────────────────────────────────────────────────────────────────────


def should_store(state: CollectState) -> str:
    """Route after extraction: store if data is complete, else end (wait for more input)."""
    if state.get("extracted") is not None:
        return "store"
    return END


# ── Graph assembly ──────────────────────────────────────────────────────────────

def build_collect_graph():
    """Build and compile the collect LangGraph."""
    graph = StateGraph(CollectState)

    graph.add_node("chat", chat_node)
    graph.add_node("extract", extract_node)
    graph.add_node("store", store_node)

    graph.set_entry_point("chat")
    graph.add_edge("chat", "extract")
    graph.add_conditional_edges("extract", should_store, {"store": "store", END: END})
    graph.add_edge("store", END)

    return graph.compile()


# Pre-compiled graph instance
collect_graph = build_collect_graph()
