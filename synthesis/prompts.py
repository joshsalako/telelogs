"""Prompt templates for diverse reasoning and final trace synthesis."""

from __future__ import annotations

from .models import Strategy


COMMON_RULES = """
Use only the evidence in the question. The candidate identifiers have been
randomized, so preserve their R1-R8 labels exactly. Perform the technical RCA
carefully and end with exactly one final prediction written as \\boxed{R#}.
""".strip()

DOMAIN_KNOWLEDGE_AND_HEURISTICS = """Domain Knowledge & Evaluation Heuristics

The C1-C8 names below identify canonical root-cause meanings, not the labels in
the question. The question randomizes those candidates to R1-R8. Locate each
cause by its description, carry out the required check, and report the result
using the corresponding R label. Never assume that C1 maps to R1, C2 to R2,
and so on.

You MUST explicitly calculate or verify every check below in your
Chain-of-Thought reasoning before reaching a conclusion. Show the relevant
values, arithmetic, threshold comparison, and resulting candidate decision;
do not rely on a qualitative impression alone.

1. C1 — Excessive Downtilt: Read Mechanical Downtilt, Digital Tilt, and Beam
   Scenario for the serving cell from the Engineering Parameters table.
   Convert the encoded Digital Tilt value 255 to 6 degrees, then calculate
   Total Downtilt = Mechanical Downtilt + Digital Tilt (in degrees). Compare
   the total with the vertical beamwidth: DEFAULT and SCENARIO_1 through
   SCENARIO_5 = 6 degrees; SCENARIO_6 through SCENARIO_11 = 12 degrees; and
   SCENARIO_12 or higher = 25 degrees. High total downtilt combined with a
   narrow vertical beamwidth makes far-end coverage weak.
2. C6 — PCI Mod 30 Conflict: Calculate Serving_PCI % 30 and separately
   calculate Neighbor_PCI % 30 for every available top neighbor. Display the
   remainders. An equal remainder indicates severe interference. If none of
   the neighbor remainders equals the serving remainder, C6 MUST be eliminated.
3. C4 — Non-colocated Overlapping Coverage: Identify the relevant serving and
   interfering-neighbor cells by PCI, then compare their `gNodeB ID` values in
   the Engineering Parameters table. If the IDs are exactly equal, the cells
   are colocated; C4 is then mathematically impossible and MUST be ruled out.
4. C7 — Test Vehicle Speed and C8 — Average RBs: Scan every User Plane row.
   State max(`GPS Speed (km/h)`) and whether it is greater than 40 km/h for C7.
   Calculate and state the arithmetic mean of the `DL RB Num` column and
   whether it is below 160 for C8. Use the full column whose header contains
   `DL RB Num` when the table uses a longer KPI name.
5. C3 — Neighbor Cell Higher Throughput: Reconstruct chronological order from
   the Timestamp column because rows may be presented out of order. Find each
   handover where the Serving PCI changes to a PCI previously observed as a
   neighbor. Compare throughput in Mbps immediately before and after the
   handover and state whether it jumps significantly. A significant immediate
   increase supports C3 because the neighbor provides higher throughput.
""".strip()

ELIMINATION_SYSTEM_PROMPT = f"""You are a senior 5G radio-network RCA engineer.
Systematically evaluate every candidate root cause against the drive-test and
engineering-parameter evidence. Explicitly rule out implausible candidates,
then select the strongest remaining explanation. Incorporate every mandatory
calculation below into this elimination routine and use each result to retain
or eliminate the corresponding randomized candidate.

{DOMAIN_KNOWLEDGE_AND_HEURISTICS}

{COMMON_RULES}"""

CONTRADICTION_SYSTEM_PROMPT = f"""You are a senior 5G radio-network RCA engineer.
For each candidate root cause, temporarily assume it is correct, derive what
the tables should show under that assumption, and compare those implications
with the actual observations. Discard assumptions that create contradictions,
then select the candidate that remains consistent. Incorporate every mandatory
calculation below into this contradiction routine: state what each candidate
requires, compute the observed result, and identify whether the result supports
or contradicts the corresponding randomized candidate.

{DOMAIN_KNOWLEDGE_AND_HEURISTICS}

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
