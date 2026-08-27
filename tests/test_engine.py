import pytest
from fastapi.testclient import TestClient
from core.retriever import LegalClauseRetriever
from core.auditor import LegalHallucinationAuditor
from server.main import app


@pytest.fixture(scope="module")
def retriever():
    return LegalClauseRetriever()


@pytest.fixture(scope="module")
def auditor():
    return LegalHallucinationAuditor()


def test_retriever_top_k(retriever):
    hits = retriever.retrieve("termination notice period", top_k=2)
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]
    assert "clause_id" in hits[0]
    assert "text" in hits[0]


def test_auditor_verification(auditor):
    premise = "Either party may terminate this agreement upon 30 days prior written notice."
    safe_claim = "The agreement permits termination given 30 days notice."
    contradictory_claim = "The agreement cannot be terminated under any circumstances."

    safe_res = auditor.audit_claim(premise, safe_claim)
    assert safe_res["status"] in ["VERIFIED_SAFE", "UNGROUNDED_NEUTRAL"]

    contra_res = auditor.audit_claim(premise, contradictory_claim)
    assert contra_res["is_safe"] is False
    assert contra_res["status"] == "HALLUCINATION_CONTRADICTION"


def test_api_query_endpoint():
    with TestClient(app) as client:
        payload = {
            "query": "termination for cause",
            "top_k": 1,
            "simulated_claims": [
                "Either party may terminate upon specified conditions.",
                "Distributor must pay $1,000,000 penalty immediately."
            ]
        }
        response = client.post("/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        
        assert "top_clause" in data
        assert len(data["audit_results"]) == 2
        assert len(data["flagged_claims"]) >= 1
