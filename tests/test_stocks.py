"""Tests for GET /stocks endpoint."""


class TestStocksEndpoint:
    """Test the stock listing endpoint."""

    def test_list_stocks_returns_data(self, client, mock_run_cypher):
        """Should return all stocks from Neo4j."""
        mock_run_cypher.return_value = [
            {"stock": "TCS", "sector": "IT", "linked_stocks": ["INFY", "WIPRO"]},
            {"stock": "RELIANCE", "sector": "OIL", "linked_stocks": ["BPCL"]},
        ]

        resp = client.get("/stocks")

        assert resp.status_code == 200
        data = resp.json()
        assert "stocks" in data
        assert len(data["stocks"]) == 2
        assert data["stocks"][0]["stock"] == "TCS"
        assert data["stocks"][1]["stock"] == "RELIANCE"

    def test_list_stocks_empty(self, client, mock_run_cypher):
        """Should return an empty list when no stocks exist."""
        mock_run_cypher.return_value = []

        resp = client.get("/stocks")

        assert resp.status_code == 200
        data = resp.json()
        assert data["stocks"] == []

    def test_list_stocks_with_no_linked(self, client, mock_run_cypher):
        """Stock with no linked stocks should show an empty list."""
        mock_run_cypher.return_value = [
            {"stock": "HDFC", "sector": "BANKING", "linked_stocks": []},
        ]

        resp = client.get("/stocks")

        assert resp.status_code == 200
        stocks = resp.json()["stocks"]
        assert len(stocks) == 1
        assert stocks[0]["linked_stocks"] == []
