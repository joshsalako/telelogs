"""Prompt templates for diverse reasoning and final trace synthesis."""

from __future__ import annotations

from .models import Strategy


COMMON_RULES = """
Use only the evidence in the question. The candidate identifiers have been
randomized, so preserve their R1-R8 labels exactly. Perform the technical RCA
carefully and end with exactly one final prediction written as \\boxed{R#}.
""".strip()

ELIMINATION_SYSTEM_PROMPT = f"""You are a senior 5G radio-network RCA engineer.
Systematically evaluate every candidate root cause against the drive-test and
engineering-parameter evidence. Explicitly rule out implausible candidates,
then select the strongest remaining explanation.

{COMMON_RULES}"""

CONTRADICTION_SYSTEM_PROMPT = f"""You are a senior 5G radio-network RCA engineer.
For each candidate root cause, temporarily assume it is correct, derive what
the tables should show under that assumption, and compare those implications
with the actual observations. Discard assumptions that create contradictions,
then select the candidate that remains consistent.

{COMMON_RULES}"""

FORMATTER_SYSTEM_PROMPT = """You are an RCA reasoning editor. Synthesize the
provided validated trajectory into a concise, self-contained answer grounded
in the randomized question. Remove repetition, backtracking, and unsupported
claims. Do not rename candidate identifiers or change the supplied validated
answer.

Return exactly these four sections in this order:
Task 1: Data analysis
Task 2: Root cause analysis
Task 3: Root cause identification
Summary

The Summary must end with the validated answer in \\boxed{R#}. Do not add
another boxed expression anywhere else."""


def reasoning_messages(question: str, strategy: Strategy) -> list[dict[str, str]]:
    system_prompt = (
        ELIMINATION_SYSTEM_PROMPT
        if strategy is Strategy.ELIMINATION
        else CONTRADICTION_SYSTEM_PROMPT
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def formatter_messages(
    question: str, trajectory: str, validated_answer: str
) -> list[dict[str, str]]:
    content = f"""Randomized question:
<question>
{question}
</question>

Validated reasoning trajectory:
<trajectory>
{trajectory}
</trajectory>

Validated final identifier: {validated_answer}
"""
    return [
        {"role": "system", "content": FORMATTER_SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
