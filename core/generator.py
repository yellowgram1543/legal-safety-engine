import re
from typing import Dict, Any, List
import torch
from transformers import pipeline


class LocalLegalGenerator:
    """Local generative model (Qwen2.5-0.5B-Instruct) for answering legal queries."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float32,
        )

    def generate_answer(self, query: str, context: str) -> Dict[str, Any]:
        """Generate an answer from context and split into distinct claims."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a legal assistant. Answer the user question concisely using "
                    "ONLY the provided contract excerpt. Write 2 or 3 complete sentences."
                ),
            },
            {
                "role": "user",
                "content": f"Contract Excerpt:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        output = self.pipe(
            messages,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            return_full_text=False,
        )

        response_text = output[0]["generated_text"].strip()

        # Split into distinct sentences (claims) for NLI auditing
        raw_claims = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", response_text)
            if len(s.strip()) > 10
        ]

        if not raw_claims:
            raw_claims = [response_text]

        return {
            "full_response": response_text,
            "claims": raw_claims,
        }
