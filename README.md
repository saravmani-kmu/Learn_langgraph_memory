# Stock Knowledge Graph API

A conversational REST API built with **FastAPI**, **LangGraph**, **Groq LLM**, and **Neo4j** that:

1. **Collects** stock information (name, sector, linked stocks) through multi-turn conversation.
2. **Stores** it as a knowledge graph in Neo4j.
3. **Answers** natural-language queries by auto-generating Cypher.

## Quick Start

### 1. Prerequisites
- Python 3.11+
- Neo4j (running at `bolt://localhost:7687`)
- Groq API key

### 2. Install

```bash
pip install -r requirements.txt
```

### 3. Configure

Copy `.env.example` → `.env` and fill in your credentials:

```
GROQ_API_KEY=gsk_...
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
```

### 4. Run

```bash
uvicorn app.main:app --reload
```

Server starts at `http://127.0.0.1:8000`. Docs at `/docs`.

## API Endpoints

### `POST /collect` – Collect Stock Info (Multi-turn)

```bash
# Turn 1 – Start conversation
curl -X POST http://localhost:8000/collect \
  -H "Content-Type: application/json" \
  -d '{"user_message": "I want to add TCS stock", "session_id": "s1"}'

# Turn 2 – Provide sector
curl -X POST http://localhost:8000/collect \
  -H "Content-Type: application/json" \
  -d '{"user_message": "It belongs to IT sector", "session_id": "s1"}'

# Turn 3 – Provide linked stocks
curl -X POST http://localhost:8000/collect \
  -H "Content-Type: application/json" \
  -d '{"user_message": "Linked stocks are INFY and WIPRO", "session_id": "s1"}'
```

### `POST /query` – Ask Questions

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which stocks are linked to TCS?"}'
```

### `GET /stocks` – List All Stocks

```bash
curl http://localhost:8000/stocks
```

## Graph Schema

```
(:Stock {name, sector})-[:BELONGS_TO]->(:Sector {name})
(:Stock)-[:LINKED_TO]->(:Stock)
```
