import time
import re
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

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace for consistent matching."""
        return re.sub(r'\s+', ' ', text).strip()

    def audit_claim(self, premise: str, claim: str) -> Dict[str, Any]:
        """Audit a single generated claim against a source premise text."""
        t0 = time.perf_counter()

        clean_premise = self._normalize_text(premise)
        clean_claim = self._normalize_text(claim)

        # Deterministic Shortcut: If the claim is an exact literal substring of the premise,
        # it is definitionally entailed. NLI models often degrade on long-context exact matches.
        if clean_claim.lower() in clean_premise.lower() and len(clean_claim) > 10:
            prob_contradiction = 0.0
            prob_entailment = 1.0
            prob_neutral = 0.0
        # Deterministic Shortcut: If the LLM correctly identified that the topic is unmentioned
        # (via our system prompt guardrail), explicitly label it as Neutral instead of letting the NLI panic.
        elif "does not contain information regarding this topic" in clean_claim.lower():
            prob_contradiction = 0.0
            prob_entailment = 0.0
            prob_neutral = 1.0
        else:
            # CrossEncoder returns raw logits for [contradiction, entailment, neutral]
            scores = self.model.predict([(clean_premise, clean_claim)], apply_softmax=True)[0]
            prob_contradiction = float(scores[0])
            prob_entailment = float(scores[1])
            prob_neutral = float(scores[2])

        latency_ms = (time.perf_counter() - t0) * 1000

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
