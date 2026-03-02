"""Tests for POST /query endpoint."""


class TestQueryEndpoint:
    """Test the natural-language → Cypher → answer query flow."""

    def test_query_returns_answer_and_cypher(self, client, mock_query_graph):
        """Successful query should return a natural-language answer and the Cypher used."""
        mock_query_graph.invoke.return_value = {
            "question": "Which stocks are linked to TCS?",
            "cypher": "MATCH (s:Stock {name: 'TCS'})-[:LINKED_TO]->(ls) RETURN ls.name",
            "result": "[{'ls.name': 'INFY'}, {'ls.name': 'WIPRO'}]",
            "answer": "TCS is linked to INFY and WIPRO.",
        }

        resp = client.post("/query", json={
            "question": "Which stocks are linked to TCS?",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert "TCS" in data["answer"]
        assert "MATCH" in data["cypher_used"]

    def test_query_no_results(self, client, mock_query_graph):
        """When no data is found, the answer should still be returned gracefully."""
        mock_query_graph.invoke.return_value = {
            "question": "Tell me about XYZ stock",
            "cypher": "MATCH (s:Stock {name: 'XYZ'}) RETURN s",
            "result": "No results found.",
            "answer": "I don't have any information about XYZ stock yet.",
        }

        resp = client.post("/query", json={
            "question": "Tell me about XYZ stock",
        })

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"]  # should not be empty
        assert data["cypher_used"]

    def test_query_missing_question(self, client):
        """Request without question should return 422."""
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_query_sector_question(self, client, mock_query_graph):
        """Should handle sector-related queries."""
        mock_query_graph.invoke.return_value = {
            "question": "Which stocks are in the IT sector?",
            "cypher": "MATCH (s:Stock)-[:BELONGS_TO]->(:Sector {name: 'IT'}) RETURN s.name",
            "result": "[{'s.name': 'TCS'}, {'s.name': 'INFY'}, {'s.name': 'WIPRO'}]",
            "answer": "The IT sector contains TCS, INFY, and WIPRO.",
        }

        resp = client.post("/query", json={
            "question": "Which stocks are in the IT sector?",
        })

        assert resp.status_code == 200
        assert "IT" in resp.json()["answer"]
