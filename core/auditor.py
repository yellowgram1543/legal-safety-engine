import time
from typing import Dict, Any
import numpy as np
import torch
from sentence_transformers import CrossEncoder


class LegalHallucinationAuditor:
    """NLI-based fact-checking auditor to detect hallucinations and enforce grounding."""

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        entailment_threshold: float = 0.70,
        contradiction_threshold: float = 0.40,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=self.device)
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold

    def audit_claim(self, premise: str, claim: str) -> Dict[str, Any]:
        """Audit a single generated claim against a source premise text."""
        t0 = time.perf_counter()

        # CrossEncoder returns raw logits for [contradiction, entailment, neutral]
        scores = self.model.predict([(premise, claim)], apply_softmax=True)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        prob_contradiction = float(scores[0])
        prob_entailment = float(scores[1])
        prob_neutral = float(scores[2])

        if prob_entailment >= self.entailment_threshold:
            status = "VERIFIED_SAFE"
            is_safe = True
        elif prob_contradiction > self.contradiction_threshold:
            status = "HALLUCINATION_CONTRADICTION"
            is_safe = False
        else:
            status = "UNGROUNDED_NEUTRAL"
            is_safe = False

        return {
            "claim": claim,
            "status": status,
            "is_safe": is_safe,
            "entailment_score": prob_entailment,
            "contradiction_score": prob_contradiction,
            "neutral_score": prob_neutral,
            "latency_ms": latency_ms,
        }
