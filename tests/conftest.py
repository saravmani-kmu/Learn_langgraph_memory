"""Shared pytest fixtures for the Stock Knowledge Graph API tests."""

import os
import pytest
from unittest.mock import patch, MagicMock

# Set dummy env vars BEFORE importing the app (prevents Groq init errors)
os.environ.setdefault("GROQ_API_KEY", "test_key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USERNAME", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test")

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import app
from app.api.routes import _sessions


@pytest.fixture()
def client():
    """FastAPI test client with mocked Neo4j connectivity check."""
    with patch("app.main.get_driver") as mock_driver, \
         patch("app.main.close_driver"):
        mock_driver.return_value = MagicMock()
        with TestClient(app) as c:
            yield c
    # Clear sessions between tests
    _sessions.clear()


@pytest.fixture()
def mock_collect_graph():
    """Patch the collect_graph.invoke to return a controlled state."""
    with patch("app.api.routes.collect_graph") as mock_graph:
        yield mock_graph


@pytest.fixture()
def mock_query_graph():
    """Patch the query_graph.invoke to return a controlled state."""
    with patch("app.api.routes.query_graph") as mock_graph:
        yield mock_graph


@pytest.fixture()
def mock_run_cypher():
    """Patch run_cypher to return controlled data."""
    with patch("app.api.routes.run_cypher") as mock_cypher:
        yield mock_cypher
