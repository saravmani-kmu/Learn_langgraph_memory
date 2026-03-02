"""Tests for POST /collect endpoint."""

from langchain_core.messages import AIMessage, HumanMessage

from app.models.schemas import StockInfo


class TestCollectEndpoint:
    """Test the multi-turn stock collection conversation flow."""

    def test_collect_first_turn_asks_question(self, client, mock_collect_graph):
        """First message should trigger the bot to ask for stock details."""
        mock_collect_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="I want to add a stock"),
                AIMessage(content="Sure! What is the stock name?"),
            ],
            "extracted": None,
            "stored": False,
        }

        resp = client.post("/collect", json={
            "user_message": "I want to add a stock",
            "session_id": "test-1",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert "stock name" in data["assistant_message"].lower()
        assert data["extracted"] is None
        assert data["stored"] is False

    def test_collect_returns_extracted_when_complete(self, client, mock_collect_graph):
        """When all info is gathered, response should contain extracted data and stored=True."""
        mock_collect_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="TCS, IT sector, linked to INFY and WIPRO"),
                AIMessage(content="Got it! I've stored TCS in the IT sector linked to INFY and WIPRO."),
            ],
            "extracted": StockInfo(
                stock_name="TCS",
                sector="IT",
                linked_stocks=["INFY", "WIPRO"],
            ),
            "stored": True,
        }

        resp = client.post("/collect", json={
            "user_message": "TCS, IT sector, linked to INFY and WIPRO",
            "session_id": "test-2",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["stored"] is True
        assert data["extracted"]["stock_name"] == "TCS"
        assert data["extracted"]["sector"] == "IT"
        assert "INFY" in data["extracted"]["linked_stocks"]
        assert "WIPRO" in data["extracted"]["linked_stocks"]

    def test_collect_session_cleared_after_store(self, client, mock_collect_graph):
        """Session should be cleared after successful storage so a new collection can start."""
        mock_collect_graph.invoke.return_value = {
            "messages": [
                AIMessage(content="Stored successfully!"),
            ],
            "extracted": StockInfo(stock_name="RELIANCE", sector="OIL", linked_stocks=["BPCL"]),
            "stored": True,
        }

        resp = client.post("/collect", json={
            "user_message": "Add RELIANCE in OIL sector linked to BPCL",
            "session_id": "test-3",
        })
        assert resp.json()["stored"] is True

        # A second call with the same session_id should start fresh
        mock_collect_graph.invoke.return_value = {
            "messages": [
                HumanMessage(content="Add another"),
                AIMessage(content="Sure! What is the stock name?"),
            ],
            "extracted": None,
            "stored": False,
        }

        resp2 = client.post("/collect", json={
            "user_message": "Add another",
            "session_id": "test-3",
        })
        assert resp2.json()["stored"] is False

    def test_collect_missing_user_message(self, client):
        """Request without user_message should return 422."""
        resp = client.post("/collect", json={"session_id": "x"})
        assert resp.status_code == 422

    def test_collect_default_session_id(self, client, mock_collect_graph):
        """If session_id is omitted, it defaults to 'default'."""
        mock_collect_graph.invoke.return_value = {
            "messages": [AIMessage(content="Hello!")],
            "extracted": None,
            "stored": False,
        }

        resp = client.post("/collect", json={"user_message": "hi"})
        assert resp.status_code == 200
