"""Grounded generation prompt templates.

Each template instructs the model to answer ONLY from the supplied evidence,
cite its sources, and refuse when evidence is insufficient.
"""

from __future__ import annotations

from app.retrieval.models import RetrievalResult

# ---------------------------------------------------------------------------
# Evidence formatting
# ---------------------------------------------------------------------------


def format_evidence(candidates: list[RetrievalResult]) -> str:
    """Format a list of retrieval results as a citation context block.

    Produces a numbered list of chunks prefixed with ``[C1]``, ``[C2]``, … so
    the model can reference them in its answer::

        [C1] document: authentication-guide
        section: Token Lifecycle
        text: Access tokens expire after 60 minutes.

        [C2] document: rate-limits
        section: Overview
        text: The API allows 1000 requests per minute per client.
    """
    if not candidates:
        return "No evidence provided."

    lines: list[str] = []
    for idx, result in enumerate(candidates, start=1):
        citation_id = f"[C{idx}]"
        doc_id = result.document_id or result.source or "unknown"
        section = result.section or ""
        page = result.page_number
        text = result.text

        header = f"{citation_id} document: {doc_id}"
        if section:
            header += f"\nsection: {section}"
        if page is not None:
            header += f"\npage: {page}"
        header += f"\ntext: {text}"

        lines.append(header)
        lines.append("")  # blank line between entries

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """You are a precise technical assistant.
Your task is to answer user questions using ONLY the evidence provided below.

## Rules

1. Answer based ONLY on the supplied evidence.
   Do NOT invent, extrapolate, or assume information not present in the evidence.
2. If the evidence is insufficient to answer the question, respond with a refusal. Do NOT make up an answer.
3. Every non-trivial claim MUST be accompanied by a citation marker such as [C1], [C2], etc.
   Place the marker immediately AFTER the specific claim it supports — never
   group all citations at the very end of the answer. If your answer states
   several facts in separate sentences, put a marker at the end of EACH of
   those sentences. Only short connective or explanatory phrasing (e.g. "In
   summary") may omit a marker.
4. Use the EXACT citation IDs from the evidence (e.g. [C1], [C2]; not [C3]
   unless that ID actually exists in the evidence).
5. Keep citations attached to specific claims, not just at the end of a paragraph.
   The FIRST sentence of your answer often restates the core fact — if that
   fact comes from the evidence, it must carry its own citation marker too.
6. If you cannot answer using the provided evidence, respond with a refusal message and set "refused" to true.
7. Never invent citation IDs. Only use citation IDs that appear in the provided evidence.

## Response Format

You MUST respond with a valid JSON object (no markdown, no code fences, no extra text)
containing these exact fields:

- "answer": Your full answer text with inline citation markers.
- "citations": A list of citation objects, each with:
    - "citation_id": The marker used in the answer (e.g. "[C1]")
    - "chunk_id": The chunk ID from the evidence (e.g. "authentication-guide:1")
    - "text": The exact text span you cited (up to you)
- "confidence": A number between 0.0 and 1.0 indicating how confident you are.
- "refused": false if you answered, true if you refused.
- "refused_reason": null if you answered, or a brief reason string if you refused.

Example answer response:
{
  "answer": "Access tokens expire after 60 minutes. [C1] Tokens can be rotated from the Admin Console under the Keys tab. [C2]",
  "citations": [
    {
      "citation_id": "[C1]",
      "chunk_id": "authentication-guide:1",
      "text": "Access tokens expire after 60 minutes."
    },
    {
      "citation_id": "[C2]",
      "chunk_id": "authentication-guide:3",
      "text": "Tokens can be rotated from the Admin Console under the Keys tab."
    }
  ],
  "confidence": 0.95,
  "refused": false,
  "refused_reason": null
}

Example refusal response:
{
  "answer": "I don't have enough information to answer this question.",
  "citations": [],
  "confidence": 0.1,
  "refused": true,
  "refused_reason": "No supporting evidence was found."
}
"""


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """## Evidence

{evidence}

---

## Question

{question}

---

Produce your JSON response now."""


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def build_prompts(
    question: str,
    evidence: list[RetrievalResult],
) -> tuple[str, str]:
    """Build (system_prompt, user_prompt) for the given question and evidence.

    Parameters
    ----------
    question:
        The user's natural-language question.
    evidence:
        List of retrieval results to use as grounding evidence.

    Returns
    -------
    tuple[str, str]
        The system prompt and user prompt strings.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE
    evidence_text = format_evidence(evidence)
    user_prompt = USER_PROMPT_TEMPLATE.format(
        evidence=evidence_text,
        question=question,
    )
    return system_prompt, user_prompt
