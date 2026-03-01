"""
LangGraph workflow – Query the stock knowledge graph using natural language.

Flow: generate_cypher → execute_cypher → synthesize → END
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from app.db.neo4j_client import run_cypher
from app.models.schemas import CypherOutput

# ── LLM setup ──────────────────────────────────────────────────────────────────

_llm = None


def _get_llm():
    """Lazily initialise ChatGroq so env vars are loaded first."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    return _llm

NEO4J_SCHEMA = """\
Node labels and properties:
  - (:Stock {name: STRING, sector: STRING})
  - (:Sector {name: STRING})

Relationship types:
  - (:Stock)-[:BELONGS_TO]->(:Sector)
  - (:Stock)-[:LINKED_TO]->(:Stock)   (bidirectional)

All stock and sector names are stored in UPPERCASE.
"""

CYPHER_SYSTEM_PROMPT = f"""\
You are a Neo4j Cypher expert. Given the graph schema below and a user question,
generate a SINGLE valid Cypher query to answer the question.

**Graph Schema:**
{NEO4J_SCHEMA}

Rules:
- Use UPPER CASE for any stock or sector names in the query.
- Return meaningful columns with aliases.
- Do NOT use APOC or custom procedures.
- Keep the query as simple as possible.
"""

SYNTHESIZE_SYSTEM_PROMPT = """\
You are a helpful financial assistant. The user asked a question about stocks.
A Cypher query was run against the knowledge graph and returned some results.
Use those results to give a clear, concise, natural-language answer.
If the results are empty, say you don't have that information yet.
"""

# ── State ───────────────────────────────────────────────────────────────────────


class QueryState(TypedDict):
    question: str
    cypher: str
    result: str
    answer: str


# ── Nodes ───────────────────────────────────────────────────────────────────────


def generate_cypher_node(state: QueryState) -> dict:
    """Ask the LLM to produce a Cypher query for the user's question."""
    structured_llm = _get_llm().with_structured_output(CypherOutput)

    messages = [
        SystemMessage(content=CYPHER_SYSTEM_PROMPT),
        HumanMessage(content=state["question"]),
    ]

    output: CypherOutput = structured_llm.invoke(messages)
    return {"cypher": output.cypher_query}


def execute_cypher_node(state: QueryState) -> dict:
    """Run the generated Cypher against Neo4j."""
    try:
        records = run_cypher(state["cypher"])
        result_text = str(records) if records else "No results found."
    except Exception as exc:
        result_text = f"Error executing Cypher: {exc}"
    return {"result": result_text}


def synthesize_node(state: QueryState) -> dict:
    """Turn the raw Cypher results into a human-friendly answer."""
    messages = [
        SystemMessage(content=SYNTHESIZE_SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"User question: {state['question']}\n\n"
                f"Cypher query used:\n{state['cypher']}\n\n"
                f"Query results:\n{state['result']}"
            )
        ),
    ]
    response = _get_llm().invoke(messages)
    return {"answer": response.content}


# ── Graph assembly ──────────────────────────────────────────────────────────────

def build_query_graph():
    """Build and compile the query LangGraph."""
    graph = StateGraph(QueryState)

    graph.add_node("generate_cypher", generate_cypher_node)
    graph.add_node("execute_cypher", execute_cypher_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("generate_cypher")
    graph.add_edge("generate_cypher", "execute_cypher")
    graph.add_edge("execute_cypher", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


# Pre-compiled graph instance
query_graph = build_query_graph()
