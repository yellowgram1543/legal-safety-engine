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
        """Generate a strictly grounded answer from context and split into claims."""
        system_instruction = (
            "You are a strict, faithful legal contract analyst. "
            "Answer the question concisely in 1 to 3 sentences using ONLY the provided excerpt. "
            "NEVER start your answer with 'Yes', 'No', or negative phrases. Instead, state affirmatively exactly what the contract says. "
            "For example, instead of saying 'No, it does not say 90 days', say 'The contract states it is 30 days'. "
            "If the excerpt does not mention the topic or entity in the question, state: "
            "'The provided contract clause does not contain information regarding this topic.'"
        )

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Contract Excerpt:\n{context}\n\nQuestion: {query}\n\nAnswer:"},
        ]

        output = self.pipe(
            messages,
            max_new_tokens=150,
            temperature=0.1,
            do_sample=False,
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
