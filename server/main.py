import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from core.retriever import LegalClauseRetriever
from core.auditor import LegalHallucinationAuditor
from core.generator import LocalLegalGenerator
from server.schemas import QueryRequest, QueryResponse, ClauseCitation, ClaimAuditVerdict


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Initializing FAISS Retriever, NLI Auditor, and Local Qwen Generator...")
    app.state.retriever = LegalClauseRetriever()
    app.state.auditor = LegalHallucinationAuditor()
    app.state.generator = LocalLegalGenerator()
    print("All models and indices loaded successfully.")
    yield
    print("Shutting down engine...")


app = FastAPI(
    title="Deterministic Legal Safety Engine",
    version="1.0.0",
    description="Zero-trust RAG backend with AST clause retrieval, local LLM generation, and NLI hallucination auditing.",
    lifespan=lifespan,
)

os.makedirs("server/static", exist_ok=True)
app.mount("/static", StaticFiles(directory="server/static"), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse("server/static/index.html")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> dict:
    return {"status": "healthy", "service": "legal-safety-engine"}


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_legal_query(payload: QueryRequest) -> QueryResponse:
    retriever: LegalClauseRetriever = app.state.retriever
    auditor: LegalHallucinationAuditor = app.state.auditor
    generator: LocalLegalGenerator = app.state.generator

    # 1. Retrieve Ground Truth Context
    hits = retriever.retrieve(payload.query, top_k=payload.top_k)
    if not hits:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant clauses found in the indexed contract.",
        )

    best_hit = hits[0]
    top_clause = ClauseCitation(**best_hit)

    # 2. Generate Real LLM Claims (or use custom payload claims if provided)
    if payload.simulated_claims and len(payload.simulated_claims) > 0:
        claims_to_audit = payload.simulated_claims
    else:
        gen_result = generator.generate_answer(
            query=payload.query,
            context=best_hit["text"]
        )
        claims_to_audit = gen_result["claims"]

    # 3. Audit each LLM-generated claim against the raw clause
    audit_results = []
    verified_claims = []
    flagged_claims = []

    for claim in claims_to_audit:
        result = auditor.audit_claim(premise=best_hit["text"], claim=claim)
        verdict = ClaimAuditVerdict(**result)
        audit_results.append(verdict)

        if verdict.is_safe:
            verified_claims.append(verdict.claim)
        else:
            flagged_claims.append(verdict.claim)

    return QueryResponse(
        query=payload.query,
        top_clause=top_clause,
        audit_results=audit_results,
        verified_safe_claims=verified_claims,
        flagged_claims=flagged_claims,
    )
