"""Neo4j database client – singleton driver + helper functions."""

import os
from neo4j import GraphDatabase

_driver = None


def get_driver():
    """Return (and lazily create) the singleton Neo4j driver."""
    global _driver
    if _driver is None:
        uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD", "")
        _driver = GraphDatabase.driver(uri, auth=(user, password))
    return _driver


def close_driver():
    """Shut down the Neo4j driver (call on app shutdown)."""
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def run_cypher(query: str, params: dict | None = None) -> list[dict]:
    """Execute *query* with *params* and return all records as dicts."""
    driver = get_driver()
    with driver.session() as session:
        result = session.run(query, parameters=params or {})
        return [record.data() for record in result]


def upsert_stock(stock_name: str, sector: str, linked_stocks: list[str]) -> None:
    """
    MERGE a Stock node, its Sector, and LINKED_TO relationships.

    Graph model
    -----------
    (:Stock {name})-[:BELONGS_TO]->(:Sector {name})
    (:Stock)-[:LINKED_TO]->(:Stock)
    """
    stock_name = stock_name.strip().upper()
    sector = sector.strip().upper()
    linked_stocks = [s.strip().upper() for s in linked_stocks if s.strip()]

    query = """
    MERGE (s:Stock {name: $stock_name})
    SET s.sector = $sector
    MERGE (sec:Sector {name: $sector})
    MERGE (s)-[:BELONGS_TO]->(sec)
    WITH s
    UNWIND $linked AS linked_name
    MERGE (ls:Stock {name: linked_name})
    MERGE (s)-[:LINKED_TO]->(ls)
    MERGE (ls)-[:LINKED_TO]->(s)
    """
    driver = get_driver()
    with driver.session() as session:
        session.run(query, stock_name=stock_name, sector=sector, linked=linked_stocks)
