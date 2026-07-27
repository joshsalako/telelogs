from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from synthesis.client import MalformedModelOutput, VLLMClient
from synthesis.config import SETTINGS
from synthesis.models import Strategy, Trajectory
from synthesis.pipeline import _process_example
from synthesis.prompts import (
    CONTRADICTION_SYSTEM_PROMPT,
    DOMAIN_KNOWLEDGE_AND_HEURISTICS,
    ELIMINATION_FEW_SHOT_EXAMPLE,
    ELIMINATION_SYSTEM_PROMPT,
    reasoning_messages,
)
from synthesis.randomization import SourceFormatError, randomize_example
from synthesis.store import OutputStore
from synthesis.validation import (
    parse_boxed_answer,
    select_most_comprehensive,
    strict_majority,
    validate_formatted_output,
)


def sample_question() -> str:
    causes = "\n".join(f"C{i}: Cause number {i}." for i in range(1, 9))
    return f"""Analyze this network.
{causes}

User plane drive test data as follows：

Time|PCI|Throughput
t1|10|500
t2|20|700
t3|30|400

Engeneering parameters data as follows：

Cell|PCI|Tilt
a|10|6
b|20|8
c|30|10
"""


def table_rows(question: str, heading: str) -> list[str]:
    section = question.split(heading, maxsplit=1)[1]
    rows = [line for line in section.splitlines() if "|" in line]
    return rows[1:]


def formatted_response(answer: str = "R4") -> str:
    rows = "\n".join(
        f"| R{index} | Check {index} | Evidence {index} | "
        f"{'Supported' if f'R{index}' == answer else 'Ruled out'} |"
        for index in range(1, 9)
    )
    return f"""Task 1: Data analysis
The degradation window and key measurements were identified.
Task 2: Root cause analysis
| Candidate | Required check | Observed evidence/calculation | Verdict |
|---|---|---|---|
{rows}
Task 3: Root cause identification
{answer} is the strongest supported candidate.
Summary
The validated root cause is \\boxed{{{answer}}}"""


class RandomizationTests(unittest.TestCase):
    def test_randomization_is_deterministic_and_preserves_rows(self) -> None:
        question = sample_question()
        first = randomize_example(question, "C3", 7, 1234)
        second = randomize_example(question, "C3", 7, 1234)

        self.assertEqual(first, second)
        self.assertEqual(set(first.identifier_mapping), {f"C{i}" for i in range(1, 9)})
        self.assertEqual(
            set(first.identifier_mapping.values()), {f"R{i}" for i in range(1, 9)}
        )
        self.assertEqual(first.answer, first.identifier_mapping["C3"])
        self.assertNotIn("\nC1:", first.question)

        original_drive = table_rows(
            question, "User plane drive test data as follows："
        )[:3]
        randomized_drive = table_rows(
            first.question, "User plane drive test data as follows："
        )[:3]
        self.assertEqual(Counter(original_drive), Counter(randomized_drive))

        original_engineering = table_rows(
            question, "Engeneering parameters data as follows："
        )
        randomized_engineering = table_rows(
            first.question, "Engeneering parameters data as follows："
        )
        self.assertEqual(Counter(original_engineering), Counter(randomized_engineering))

    def test_invalid_answer_is_rejected(self) -> None:
        with self.assertRaises(SourceFormatError):
            randomize_example(sample_question(), "C9", 0, 1)

    def test_variants_are_reproducible_and_independently_randomized(self) -> None:
        variants = [
            randomize_example(sample_question(), "C3", 7, 1234, variant_index)
            for variant_index in range(3)
        ]

        self.assertEqual(
            variants,
            [
                randomize_example(sample_question(), "C3", 7, 1234, index)
                for index in range(3)
            ],
        )
        self.assertEqual([item.variant_index for item in variants], [0, 1, 2])
        self.assertEqual(len({item.question for item in variants}), 3)


class PromptTests(unittest.TestCase):
    def test_generation_defaults_use_two_balanced_lower_temperature_agents(
        self,
    ) -> None:
        self.assertEqual(SETTINGS.agents_per_item, 2)
        self.assertEqual(SETTINGS.reasoning_temperature, 0.4)
        SETTINGS.validate()

    def test_domain_heuristics_include_every_required_check(self) -> None:
        prompt = DOMAIN_KNOWLEDGE_AND_HEURISTICS
        for candidate in range(1, 9):
            self.assertIn(f"Evaluating C{candidate}", prompt)
        self.assertIn("does not by itself rule out C1", prompt)
        self.assertIn("immediately recovers above 600 Mbps", prompt)
        self.assertIn("not sufficient by itself to make C4 the primary cause", prompt)
        self.assertIn("Never select C6 after calculating unequal remainders", prompt)
        self.assertIn(
            "| Candidate | Required check | Observed evidence/calculation | Verdict |",
            prompt,
        )
        self.assertIn("Complete the table in one pass", prompt)

    def test_both_reasoning_strategies_enforce_shared_heuristics(self) -> None:
        elimination = reasoning_messages("question", Strategy.ELIMINATION)
        contradiction = reasoning_messages("question", Strategy.CONTRADICTION)

        self.assertEqual(elimination[1], {"role": "user", "content": "question"})
        self.assertEqual(contradiction[1], elimination[1])
        self.assertIn(DOMAIN_KNOWLEDGE_AND_HEURISTICS, elimination[0]["content"])
        self.assertIn(DOMAIN_KNOWLEDGE_AND_HEURISTICS, contradiction[0]["content"])
        self.assertEqual(elimination[0]["content"], ELIMINATION_SYSTEM_PROMPT)
        self.assertEqual(contradiction[0]["content"], CONTRADICTION_SYSTEM_PROMPT)
        self.assertIn("Do not narrate backtracking", elimination[0]["content"])
        self.assertIn("without revisiting earlier rows", contradiction[0]["content"])

    def test_elimination_prompt_contains_exact_c3_few_shot_example(self) -> None:
        self.assertIn(ELIMINATION_FEW_SHOT_EXAMPLE, ELIMINATION_SYSTEM_PROMPT)
        self.assertNotIn(ELIMINATION_FEW_SHOT_EXAMPLE, CONTRADICTION_SYSTEM_PROMPT)
        self.assertIn("919 % 30 = 19", ELIMINATION_FEW_SHOT_EXAMPLE)
        self.assertIn("4° + 8° = 12°", ELIMINATION_FEW_SHOT_EXAMPLE)
        self.assertIn("946.52 Mbps", ELIMINATION_FEW_SHOT_EXAMPLE)
        self.assertIn("| Candidate | Required check |", ELIMINATION_FEW_SHOT_EXAMPLE)
        self.assertIn(r"\boxed{C3}", ELIMINATION_FEW_SHOT_EXAMPLE)
        self.assertTrue(
            ELIMINATION_SYSTEM_PROMPT.endswith("as \\boxed{R#}, with no text after it.")
        )


class ValidationTests(unittest.TestCase):
    def test_box_parser_uses_last_valid_box(self) -> None:
        text = r"Discard \boxed{R2}; final answer is \boxed{ r4 }."
        self.assertEqual(parse_boxed_answer(text), "R4")

    def test_strict_majority_requires_more_than_half(self) -> None:
        trajectories = [
            Trajectory(Strategy.ELIMINATION, "a", answer)
            for answer in ("R1", "R1", "R2", "R3")
        ]
        self.assertEqual(
            strict_majority(trajectories, "R1", 4),
            (None, "no_strict_majority"),
        )

        trajectories[2] = Trajectory(Strategy.CONTRADICTION, "b", "R1")
        self.assertEqual(strict_majority(trajectories, "R1", 4), ("R1", "accepted"))
        self.assertEqual(
            strict_majority(trajectories, "R2", 4),
            (None, "incorrect_majority"),
        )

    def test_comprehensive_selection_prefers_candidate_coverage(self) -> None:
        shorter_but_broader = Trajectory(
            Strategy.ELIMINATION, "R1 R2 R3 R4 \\boxed{R1}", "R1"
        )
        longer_but_narrow = Trajectory(
            Strategy.CONTRADICTION, "R1 " * 100 + r"\boxed{R1}", "R1"
        )
        self.assertIs(
            select_most_comprehensive([longer_but_narrow, shorter_but_broader]),
            shorter_but_broader,
        )

    def test_formatted_output_requires_exact_structure_and_answer(self) -> None:
        valid = formatted_response("R4")
        validate_formatted_output(valid, "R4")
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(valid, "R3")

    def test_formatted_output_requires_all_eight_unique_table_rows(self) -> None:
        valid = formatted_response("R4")
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(valid.replace("| R8 |", "| R7 |"), "R4")
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(
                valid.replace("| R8 | Check 8 | Evidence 8 | Ruled out |\n", ""),
                "R4",
            )

    def test_formatted_output_rejects_prose_or_invalid_table_contract(self) -> None:
        valid = formatted_response("R4")
        prose = valid.replace(
            "| Candidate | Required check | Observed evidence/calculation | Verdict |\n"
            "|---|---|---|---|\n",
            "",
        )
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(prose, "R4")
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(
                valid.replace("| Ruled out |", "| Maybe |", 1), "R4"
            )
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(
                valid.replace("| Ruled out |", "| Supported |", 1), "R4"
            )

    def test_formatted_output_rejects_loops_and_nonterminal_box(self) -> None:
        valid = formatted_response("R4")
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(
                valid.replace("R4 is the strongest", "Wait, R4 is the strongest"),
                "R4",
            )
        with self.assertRaises(MalformedModelOutput):
            validate_formatted_output(valid + "\nTrailing text.", "R4")


class RetryClient(VLLMClient):
    def __init__(self) -> None:
        settings = replace(
            SETTINGS,
            max_retry_attempts=2,
            retry_backoff_min_seconds=0,
            retry_backoff_max_seconds=0,
        )
        super().__init__(settings)
        self.calls = 0

    async def _request_once(self, **_: object) -> str:
        self.calls += 1
        return "missing answer" if self.calls == 1 else r"Final: \boxed{R5}"


class RetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_malformed_output_is_retried(self) -> None:
        client = RetryClient()
        result = await client.chat(
            [],
            temperature=0,
            top_p=1,
            max_tokens=10,
            seed=1,
            validator=lambda text: parse_boxed_answer(text),
        )
        self.assertEqual(parse_boxed_answer(result), "R5")
        self.assertEqual(client.calls, 2)


class ScriptedPipelineClient:
    async def chat(self, messages: list[dict[str, str]], **_: object) -> str:
        if "reasoning editor" in messages[0]["content"]:
            return formatted_response("R1")
        await asyncio.sleep(0)
        return "R1 R2 R3 R4 R5 R6 R7 R8; final \\boxed{R1}"


class PipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_accepted_item_is_written_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(
                SETTINGS,
                output_path=root / "output.jsonl",
                state_path=root / "state.jsonl",
                model_name="test-model",
            )
            store = OutputStore(settings, "input-digest")
            store.open()
            example = randomize_example(sample_question(), "C1", 0, 99)
            # Scripted output is R1, so make R1 the validated target for this
            # orchestration test without coupling it to a particular permutation.
            example = replace(example, answer="R1")
            status = await _process_example(
                ScriptedPipelineClient(),
                store,
                settings,
                example,  # type: ignore[arg-type]
            )
            store.close()

            self.assertEqual(status, "accepted")
            record = json.loads(settings.output_path.read_text().strip())
            self.assertEqual(set(record), {"question", "response"})
            self.assertEqual(record["question"], example.question)

            resumed = OutputStore(settings, "input-digest")
            resumed.open()
            self.assertIn((0, 0), resumed.completed_variants)
            self.assertIn(example.question, resumed.output_questions)
            resumed.close()

    async def test_each_variant_has_an_independent_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(
                SETTINGS,
                output_path=root / "output.jsonl",
                state_path=root / "state.jsonl",
                model_name="test-model",
                augmentations_per_item=3,
            )
            store = OutputStore(settings, "input-digest")
            store.open()
            await store.append_discard(4, 0, "test_discard")
            await store.append_discard(4, 2, "test_discard")
            store.close()

            resumed = OutputStore(settings, "input-digest")
            resumed.open()
            self.assertEqual(resumed.completed_variants, {(4, 0), (4, 2)})
            self.assertNotIn((4, 1), resumed.completed_variants)
            resumed.close()


if __name__ == "__main__":
    unittest.main()
