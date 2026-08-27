import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from core.retriever import LegalClauseRetriever
from core.auditor import LegalHallucinationAuditor
from core.generator import LocalLegalGenerator
from core.parser import ContractIngestionPipeline
from server.schemas import QueryRequest, QueryResponse, ClauseCitation, ClaimAuditVerdict


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    print("Initializing FAISS Retriever, NLI Auditor, and Local Qwen Generator...")
    app.state.retriever = LegalClauseRetriever()
    app.state.auditor = LegalHallucinationAuditor()
    app.state.generator = LocalLegalGenerator()
    app.state.ingestion = ContractIngestionPipeline()
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


@app.post("/upload", status_code=status.HTTP_200_OK)
async def upload_contract(file: UploadFile = File(...)) -> dict:
    ingestion: ContractIngestionPipeline = app.state.ingestion
    
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in [".pdf", ".txt"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a .pdf or .txt contract."
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        doc_id = os.path.splitext(file.filename)[0]
        result = ingestion.process_and_index(tmp_path, doc_id=doc_id)
        # Hot-reload retriever with new serialized index
        app.state.retriever = LegalClauseRetriever()
        return {
            "status": "success",
            "filename": file.filename,
            "clauses_indexed": result["clauses_indexed"],
            "total_chars": result["total_chars"]
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def process_legal_query(payload: QueryRequest) -> QueryResponse:
    retriever: LegalClauseRetriever = app.state.retriever
    auditor: LegalHallucinationAuditor = app.state.auditor
    generator: LocalLegalGenerator = app.state.generator

    hits = retriever.retrieve(payload.query, top_k=payload.top_k)
    if not hits:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant clauses found in the indexed contract.",
        )

    best_hit = hits[0]
    
    # Retrieval Score Floor / Rejection Gate
    if best_hit.get("score", 1.0) < 0.60:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Rejection Gate: Top clause score ({best_hit.get('score', 0):.2f}) is below confidence threshold (0.60). No relevant contract clause found for this query.",
        )

    top_clause = ClauseCitation(**best_hit)

    if payload.simulated_claims and len(payload.simulated_claims) > 0:
        claims_to_audit = payload.simulated_claims
    else:
        gen_result = generator.generate_answer(
            query=payload.query,
            context=best_hit["text"]
        )
        claims_to_audit = gen_result["claims"]

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
