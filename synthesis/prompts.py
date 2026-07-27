"""Prompt templates for diverse reasoning and final trace synthesis."""

from __future__ import annotations

from .models import Strategy


COMMON_RULES = """
Use only the evidence in the question. The candidate identifiers have been
randomized, so preserve their R1-R8 labels exactly. Evaluate each candidate
once, record the result in the required table, and do not reopen or revise a
completed row. Do not use reconsideration language such as "wait", "check
again", or "might have missed". End with exactly one final prediction written
as \\boxed{R#}, with no text after it.
""".strip()

DOMAIN_KNOWLEDGE_AND_HEURISTICS = """### DOMAIN KNOWLEDGE & EVALUATION HEURISTICS

Evaluate all eight semantic causes exactly once. Put the required calculation
and observed result in the corresponding randomized R1-R8 row of the Task 2
table before selecting a cause.

1. **Evaluating C1 (Excessive Downtilt):** Check the serving cell's "Digital Tilt" value. If it is exactly 255, treat it as 6 degrees; never add 255 mathematically. Show "Mechanical Downtilt + normalized Digital Tilt = Total Downtilt" and compare the total with the vertical beamwidth (DEFAULT/SCENARIO_1-5 = 6°, SCENARIO_6-11 = 12°, SCENARIO_12+ = 25°). Use the serving-cell geometry and RSRP/SINR behavior to decide whether the narrow, downward beam plausibly causes weak coverage on the degraded road section. A distance below 1 km rules out C2, but it does not by itself rule out C1.
2. **Evaluating C2 (Coverage distance > 1km):** Compare the Longitude/Latitude in the User Plane data against the Longitude/Latitude of the Serving Cell in the Engineering Parameters. *Rule of thumb:* 0.01 degrees is roughly 1 km. If BOTH the latitude difference AND the longitude difference are less than 0.009, the distance is guaranteed to be under 1km, and you MUST explicitly rule out C2. If EITHER difference is > 0.01, the distance exceeds 1km, and C2 is the root cause.
3. **Evaluating C3 (Neighbor Cell Higher Throughput):** Find every serving-PCI change and write the timestamped throughput immediately before and immediately after the handover. Select C3 only when throughput is below 600 Mbps before the handover and immediately recovers above 600 Mbps on the former neighbor. If throughput stays below 600 Mbps after the handover, explicitly rule out C3.
4. **Evaluating C4 (Non-colocated Overlapping Coverage):** During the throughput-drop rows, map the serving PCI and Top 1/Top 2 neighbor PCIs to their gNodeB IDs and compare serving RSRP/SINR with neighbor BRSRP. A different gNodeB establishes that non-colocated overlap is present, but it is not sufficient by itself to make C4 the primary cause. Select C4 only when the non-colocated neighbor is strong and persistent during the drop, its presence correlates with degraded radio quality, and no stronger direct threshold or handover explanation fits. If the neighbor is weak, transient, or merely a background condition, mark C4 present-but-secondary rather than selected.
5. **Evaluating C5 (Frequent Handovers):** Scan the `5G KPI PCell RF Serving PCI` column from top to bottom. Count every time the value changes from one row to the next. If it changes 2 or more times back-and-forth (e.g., PCI A -> PCI B -> PCI A), frequent handovers are occurring, and C5 is the root cause. If it changes 0 or 1 time, it is NOT frequent, and you MUST explicitly rule out C5.
6. **Evaluating C6 (PCI Mod 30 Conflict):** For each relevant serving PCI, show `serving PCI % 30`. Independently show `neighbor PCI % 30` for Top 1, Top 2, and Top 3 during the drop. Select C6 only when at least one neighbor remainder exactly equals the serving remainder; equivalently, their non-zero absolute difference is a multiple of 30. Never select C6 after calculating unequal remainders. Apply the C6-over-C3 priority rule only after a real modulo-30 conflict has been numerically verified.
7. **Evaluating C7 (Test Vehicle Speed > 40km/h):** Scan the `GPS Speed (km/h)` column ONLY during the rows where the throughput drops. If the speed is consistently above 40 km/h (or has peaks exceeding 45 km/h) DURING the throughput drop section, C7 is the cause. Do not trigger C7 for brief 1-second speed bumps outside the degradation window.
8. **Evaluating C8 (Average RBs < 160):** Do not calculate an average. Scan the `5G KPI PCell Layer1 DL RB Num (Including 0)` column. If there are multiple rows where the RB value is strictly less than 150 (e.g., 130, 135, 145), C8 is the cause. Do not debate if 140+ is sufficient; any sustained drop below 150 validates C8. If all values consistently stay above 150, explicitly rule out C8.

The Task 2 table must contain exactly one row for each randomized candidate and
use these columns:
| Candidate | Required check | Observed evidence/calculation | Verdict |

Use only `Supported`, `Ruled out`, or `Present but secondary` in the Verdict
column, mark exactly one candidate as `Supported`, and use `abs(a-b)` rather
than pipe characters inside table cells. Complete the table in one pass.
Task 3 must select the single strongest supported candidate without repeating
all calculations.
"""

ELIMINATION_FEW_SHOT_EXAMPLE = r"""### EXAMPLE OF A PERFECT REASONING TRAJECTORY:

**User Prompt Context (Simulated):**
Throughput drops below 600 Mbps on Serving PCI 919. A handover eventually occurs to Neighbor PCI 737.
* Serving Cell (PCI 919): Mechanical Downtilt 4°, Digital Tilt 8°, Azimuth 100°, Beam Scenario SCENARIO_1, gNodeB 0000258.
* Top Neighbor Cell (PCI 737): gNodeB 0000258.
* Throughput immediately before handover: 13.23 Mbps. Throughput immediately after handover: 946.52 Mbps.
* Vehicle Speed: 34 km/h. RBs: 160-186.

**Expected Assistant Output:**
Task 1: Data analysis
Throughput falls to 13.23 Mbps on PCI 919 and immediately rises to 946.52 Mbps after the handover to PCI 737. Speed is 34 km/h and RB allocation remains 160-186.

Task 2: Root cause analysis
| Candidate | Required check | Observed evidence/calculation | Verdict |
|---|---|---|---|
| C1 | Total downtilt versus beamwidth and coverage behavior | 4° + 8° = 12° versus 6° beamwidth, but the decisive event is immediate throughput recovery on PCI 737 | Present but secondary |
| C2 | Serving distance greater than 1 km | Coordinate differences place the UE below 100 m from PCI 919 | Ruled out |
| C3 | Below 600 Mbps before handover and above 600 Mbps immediately after | 13.23 Mbps on PCI 919, then 946.52 Mbps on PCI 737 | Supported |
| C4 | Strong persistent non-colocated neighbor during the drop | PCI 919 and PCI 737 share gNodeB 0000258 | Ruled out |
| C5 | At least two back-and-forth serving-PCI changes | One handover, 919 -> 737 | Ruled out |
| C6 | Equal serving and neighbor modulo-30 values | 919 % 30 = 19; 737 % 30 = 17 | Ruled out |
| C7 | Speed above 40 km/h during degradation | 34 km/h | Ruled out |
| C8 | Multiple degradation rows below 150 RBs | RBs remain 160-186 | Ruled out |

Task 3: Root cause identification
C3 is the strongest explanation because the former neighbor becomes serving and throughput immediately recovers from 13.23 Mbps to 946.52 Mbps.

Summary
The neighboring cell provides higher throughput, as demonstrated by the immediate post-handover recovery. \boxed{C3}"""

ELIMINATION_SYSTEM_PROMPT = f"""You are a senior 5G radio-network RCA engineer.
Systematically evaluate every candidate root cause against the drive-test and
engineering-parameter evidence once. Record each candidate's required check,
observed calculation, and verdict in the required Task 2 table, then select the
strongest supported explanation. Do not narrate backtracking or reconsider a
completed row.

{DOMAIN_KNOWLEDGE_AND_HEURISTICS}

{ELIMINATION_FEW_SHOT_EXAMPLE}

{COMMON_RULES}"""

CONTRADICTION_SYSTEM_PROMPT = f"""You are a senior 5G radio-network RCA engineer.
For each candidate root cause, state its required observation and compare that
requirement with the actual evidence exactly once. Record the comparison and
verdict in the required Task 2 table. After all eight rows are complete, select
the strongest supported candidate without revisiting earlier rows.

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

Task 1 must contain only a concise description of the degradation window and
key measurements. Task 2 must be a Markdown table with exactly eight candidate
rows and these columns:
| Candidate | Required check | Observed evidence/calculation | Verdict |

Each randomized R1-R8 candidate must appear exactly once. Use only `Supported`,
`Ruled out`, or `Present but secondary` as verdicts, and mark exactly the
validated candidate as `Supported`. Do not put pipe characters inside cell
text. Task 3 must be one short paragraph selecting the strongest supported
candidate. Do not backtrack, repeat calculations, or use reconsideration
language.

The Summary must end with the validated answer in \\boxed{R#}. Do not add
another boxed expression anywhere else or write anything after it."""


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
