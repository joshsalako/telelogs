"""Prompt templates for diverse reasoning and final trace synthesis."""

from __future__ import annotations

from .models import Strategy


COMMON_RULES = """
Use only the evidence in the question. The candidate identifiers have been
randomized, so preserve their R1-R8 labels exactly. Perform the technical RCA
carefully and end with exactly one final prediction written as \\boxed{R#}.
""".strip()

DOMAIN_KNOWLEDGE_AND_HEURISTICS = """### DOMAIN KNOWLEDGE & EVALUATION HEURISTICS: You must explicitly verify the following rules step-by-step using strict pattern matching before reaching a conclusion:

1. **Evaluating C1 (Excessive Downtilt):** Calculate the Total Downtilt by adding "Mechanical Downtilt" and "Digital Tilt" (Note: a Digital Tilt of 255 equals 6 degrees). Compare this total to the cell's vertical beamwidth (DEFAULT/SCENARIO_1-5 = 6°, SCENARIO_6-11 = 12°, SCENARIO_12+ = 25°). If the total downtilt is high but the beamwidth is narrow, far-end coverage will be weak.
2. **Evaluating C2 (Coverage distance > 1km):** Do not hallucinate complex distance math. Compare the Longitude/Latitude in the User Plane data against the Longitude/Latitude of the Serving Cell in the Engineering Parameters. *Rule of thumb:* 0.01 degrees is roughly 1 km. If BOTH the latitude difference AND the longitude difference are less than 0.009, the distance is guaranteed to be under 1km, and you MUST explicitly rule out C2. If EITHER difference is > 0.01, the distance exceeds 1km, and C2 is the root cause.
3. **Evaluating C3 (Neighbor Cell Higher Throughput):** Look for a handover event (the `5G KPI PCell RF Serving PCI` changes). Check the `5G KPI PCell Layer2 MAC DL Throughput [Mbps]`. If the throughput drops below the target 600 Mbps threshold (e.g., < 350 Mbps) BEFORE the handover, but immediately recovers to a high value (e.g., > 600 Mbps) AFTER the handover, the neighbor provides higher throughput.
4. **Evaluating C4 (Non-colocated Overlapping Coverage):** Focus specifically on the rows where the throughput drop occurs. Identify the `5G KPI PCell RF Serving PCI` and the top neighbor PCIs (e.g., `Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI` or Top 2 PCI). Look up these exact PCIs in the "PCI" column of the Engineering parameters table. If ANY Top 1 or Top 2 neighbor PCI has a DIFFERENT `gNodeB ID` than the serving cell during the throughput degradation, non-colocated overlapping coverage (C4) is present. If they always share the EXACT SAME `gNodeB ID`, they are colocated, and you MUST explicitly rule out C4.
5. **Evaluating C5 (Frequent Handovers):** Scan the `5G KPI PCell RF Serving PCI` column from top to bottom. Count every time the value changes from one row to the next. If it changes 2 or more times back-and-forth (e.g., PCI A -> PCI B -> PCI A), frequent handovers are occurring, and C5 is the root cause. If it changes 0 or 1 time, it is NOT frequent, and you MUST explicitly rule out C5.
6. **Evaluating C6 (PCI Mod 30 Conflict):** Do not calculate modulo math. Instead, find the absolute difference between the `5G KPI PCell RF Serving PCI` and the `Measurement PCell Neighbor Cell Top Set(Cell Level) Top 1 PCI`. Check if this difference is EXACTLY one of the following numbers: 0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 510, 540, 570, 600, 630, 660, 690, 720, 750, 780, 810, 840, 870, 900, 930, 960, 990. If the difference is in this list, there is a conflict. If it is NOT in this list, you MUST explicitly rule out C6.
7. **Evaluating C7 (Test Vehicle Speed > 40km/h):** Scan the `GPS Speed (km/h)` column. If you see ANY individual value strictly greater than 40, C7 is the cause. If all values are 40 or below, you MUST explicitly rule out C7.
8. **Evaluating C8 (Average RBs < 160):** Do not calculate an average. Scan the `5G KPI PCell Layer1 DL RB Num (Including 0)` column. Look for severe drops. If you see multiple rows where the value drops near or below 110 (e.g., 102, 103, 104), the average is dragged down and C8 is the cause. If the values are consistently high (e.g., staying above 160), you MUST explicitly rule out C8.

Ensure that your Step-by-Step Root Cause Analysis explicitly mentions the evaluation of all 8 of these points.
""".strip()

ELIMINATION_FEW_SHOT_EXAMPLE = r"""### EXAMPLE OF A PERFECT REASONING TRAJECTORY:

**User Prompt Context (Simulated):**
Throughput drops below 600 Mbps on Serving PCI 919. A handover eventually occurs to Neighbor PCI 737.
* Serving Cell (PCI 919): Mechanical Downtilt 4°, Digital Tilt 8°, Azimuth 100°, Beam Scenario SCENARIO_1, gNodeB 0000258.
* Top Neighbor Cell (PCI 737): gNodeB 0000258.
* Vehicle Speed: 34 km/h. RBs: 160-186.

**Expected Assistant Output:**
Okay, let's tackle this problem step by step. First, I need to understand the data provided and find out why the user throughput dropped.

Task 1: Data Analysis
*   Throughput Drop: The throughput drops to 0.14 Mbps and 13.23 Mbps when the serving cell is PCI 919. After switching to PCI 737, it recovers to 346.52 Mbps and above.
*   Neighbor Cells: PCI 737 (a top neighbor of PCI 919) becomes the serving cell at 10:25:37, coinciding with throughput recovery. PCI 737's throughput is consistently high when it is the serving cell.
*   PCI Modulo 30 Check: PCI 919 mod 30 = 19; PCI 737 mod 30 = 17. No conflict (C6 unlikely).
*   Downtilt/Azimuth: PCI 919's mechanical downtilt is 4°, digital tilt totals 8°, and azimuth is 100°, potentially limiting coverage in some directions.
*   Coverage Distance: Serving cell distance is minimal < 100m. (C2 unlikely).
*   Overlapping Coverage: PCI 737 and PCI 919 belong to the same gNodeB (0000258), so non-colocated interference (C4) is less likely.
*   Frequent Handover: Only one handover to PCI 737. C5 unlikely.
*   RBs: RB values (e.g., 160–186) are high. C8 unlikely.
*   Test Speed: Low speeds (≤ 34 km/h). C7 unlikely.

Task 2: Root Cause Analysis
1) C1 (Excessive Downtilt): PCI 919's total downtilt is 12° (mechanical 4° + digital 8°), but its Beam Scenario (SCENARIO_1) has a narrow vertical beamwidth (6°). This may cause weak coverage at the far end, but the test points are very close (~ 5−20m).
2) C2 (Over-Shooting Coverage): Serving distance is < 100m (within 1 km), so this is invalid.
3) C3 (Higher Neighbor Throughput): PCI 737 provides much higher throughput. The test device should have been served by PCI 737 earlier to avoid the drop.
4) C4 (Overlapping Coverage): PCI 737 and PCI 919 are colocated (same gNodeB), so non-colocated interference is ruled out.
5) C5 (Frequent Handovers): Only one handover occurred.
6) C6 (PCI Mod 30 Conflict): No PCI conflict detected.
7) C7 (High Test Speed): Speed is low.
8) C8 (Low RBs): RBs are sufficient.

Conclusion: The throughput drop occurs because the test device remains in the weak coverage area of PCI 919 until a handover to PCI 737 (which has 10x higher throughput) occurs. PCI 737 should have been the serving cell earlier to avoid the drop.

Task 3: Most Likely Root Cause
The root cause is C3: The neighbor cell (PCI 737) provides significantly higher throughput. The serving cell (PCI 919) should have triggered a handover to PCI 737 earlier to prevent the throughput degradation.

\boxed{C3}

Summary: The root cause of the throughput degradation is identified as C3: the test device experiences a significant drop in throughput while connected to the serving cell PCI 919, which has lower throughput capacity compared to the neighboring cell PCI 737. The throughput recovers once the device is handed over to PCI 737, which provides approximately 10 times higher throughput. This indicates that an earlier handover to PCI 737 would have prevented the throughput drop."""

ELIMINATION_SYSTEM_PROMPT = f"""You are a senior 5G radio-network RCA engineer.
Systematically evaluate every candidate root cause against the drive-test and
engineering-parameter evidence. Explicitly rule out implausible candidates,
then select the strongest remaining explanation. Incorporate every mandatory
calculation below into this elimination routine and use each result to retain
or eliminate the corresponding randomized candidate.

{DOMAIN_KNOWLEDGE_AND_HEURISTICS}

{ELIMINATION_FEW_SHOT_EXAMPLE}

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
