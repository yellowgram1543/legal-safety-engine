import os
import re
import json
import faiss
import numpy as np
import torch
from typing import List, Dict, Any
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


class StructureAwareLegalParser:
    def __init__(self, min_clause_length: int = 40):
        self.min_clause_length = min_clause_length
        self.section_pattern = re.compile(
            r"(?:\n\s*)"
            r"(?:"
            r"(?:ARTICLE|SECTION|CLAUSE)\s+[0-9IVXLCDM]+(?:\.[0-9]+)*[^\n]*|"
            r"(?:[0-9]{1,2}\.[0-9]{1,2}(?:\.[0-9]+)*)\s+[^\n]+|"
            r"(?:[A-Z]\.\s+[^\n]+)|"
            r"(?:RECITALS|DEFINITIONS|MISCELLANEOUS|TERMINATION|CONFIDENTIALITY|INDEMNIFICATION)"
            r")",
            re.IGNORECASE
        )

    def parse(self, text: str, doc_id: str = "uploaded_doc") -> List[Dict[str, Any]]:
        matches = list(self.section_pattern.finditer(text))
        clauses = []

        if not matches:
            return [{
                "clause_id": f"{doc_id}_clause_0",
                "header": "FULL_DOCUMENT",
                "text": text.strip(),
                "start_char": 0,
                "end_char": len(text)
            }]

        if matches[0].start() > 0:
            preamble_text = text[:matches[0].start()].strip()
            if len(preamble_text) >= self.min_clause_length:
                clauses.append({
                    "clause_id": f"{doc_id}_clause_0",
                    "header": "PREAMBLE",
                    "text": preamble_text,
                    "start_char": 0,
                    "end_char": matches[0].start()
                })

        for i, match in enumerate(matches):
            start_pos = match.start()
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            clause_raw = text[start_pos:end_pos].strip()
            header_line = match.group(0).strip()
            body_text = clause_raw[len(header_line):].strip()
            full_clause_text = f"{header_line}\n{body_text}".strip()

            if len(full_clause_text) >= self.min_clause_length:
                clauses.append({
                    "clause_id": f"{doc_id}_clause_{len(clauses)}",
                    "header": header_line,
                    "text": full_clause_text,
                    "start_char": start_pos,
                    "end_char": end_pos
                })

        return clauses


class ContractIngestionPipeline:
    """End-to-end ingestion pipeline for parsing, embedding, and storing new contracts."""

    def __init__(
        self,
        index_path: str = "artifacts/contract_index.faiss",
        meta_path: str = "artifacts/clauses_metadata.json",
        model_name: str = "BAAI/bge-small-en-v1.5"
    ):
        self.index_path = index_path
        self.meta_path = meta_path
        self.parser = StructureAwareLegalParser()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(model_name, device=self.device)

    def extract_text(self, file_path: str) -> str:
        """Extract text from PDF or TXT file."""
        if file_path.lower().endswith(".pdf"):
            reader = PdfReader(file_path)
            pages_text = [p.extract_text() or "" for p in reader.pages]
            return "\n\n".join(pages_text)
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def process_and_index(self, file_path: str, doc_id: str = "custom_contract") -> Dict[str, Any]:
        """Extract text, chunk into clauses, embed, and serialize index to disk."""
        text = self.extract_text(file_path)
        if not text.strip():
            raise ValueError("Extracted document text is empty.")

        clauses = self.parser.parse(text, doc_id=doc_id)
        if not clauses:
            raise ValueError("No valid clauses could be extracted from document.")

        # Embed passages
        passage_texts = [f"passage: {c['text']}" for c in clauses]
        embeddings = self.embedder.encode(
            passage_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Ensure directory and serialize
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(clauses, f, indent=2)

        return {
            "doc_id": doc_id,
            "clauses_indexed": len(clauses),
            "total_chars": len(text)
        }
