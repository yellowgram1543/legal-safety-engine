from typing import List, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Natural language legal query")
    top_k: int = Field(default=2, ge=1, le=5, description="Number of clauses to retrieve")
    simulated_claims: Optional[List[str]] = Field(
        default=None,
        description="Optional list of claims to audit against the top retrieved clause"
    )


class ClauseCitation(BaseModel):
    clause_id: str
    rank: int
    score: float
    header: str
    text: str
    start_char: int
    end_char: int


class ClaimAuditVerdict(BaseModel):
    claim: str
    status: str
    is_safe: bool
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    latency_ms: float


class QueryResponse(BaseModel):
    query: str
    top_clause: ClauseCitation
    audit_results: List[ClaimAuditVerdict]
    verified_safe_claims: List[str]
    flagged_claims: List[str]
