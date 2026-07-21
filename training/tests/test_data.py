from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from rca_training.data import (
    DataValidationError,
    SFTRecord,
    format_sft_record,
    load_sft_records,
    load_training_records,
    load_validation_records,
    strip_validation_suffix,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        ids = [1]
        for message in messages:
            if message["role"] == "assistant":
                ids.extend([9, len(message["content"])])
            else:
                ids.extend([2, len(message["content"])])
        if add_generation_prompt:
            ids.append(9)
        return ids


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_jsonl(self, rows):
        path = self.root / "sft.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_sft_response_and_answer_alias(self):
        path = self.write_jsonl(
            [
                {"question": "q1", "response": "reason \\boxed{C1}"},
                {"question": "q2", "answer": "legacy \\boxed{C2}"},
            ]
        )
        records = load_sft_records(path)
        self.assertEqual(
            [record.response for record in records], ["reason \\boxed{C1}", "legacy \\boxed{C2}"]
        )

    def test_sft_rejects_malformed_duplicate_empty_and_overlength(self):
        cases = [
            [{"question": "q", "response": "r", "extra": 1}],
            [{"question": "q", "response": "r"}, {"question": "q", "response": "r2"}],
            [{"question": "", "response": "r"}],
            [{"question": "q", "response": "r", "answer": "r"}],
        ]
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(DataValidationError):
                load_sft_records(self.write_jsonl(rows))
        with self.assertRaisesRegex(DataValidationError, "maximum"):
            load_sft_records(
                self.write_jsonl([{"question": "abc", "response": "def"}]),
                token_counter=len,
                max_tokens=5,
            )

    def test_completion_only_format_masks_prompt(self):
        tokenized = format_sft_record(FakeTokenizer(), SFTRecord("question", "answer"), 100)
        self.assertEqual(tokenized["labels"][:-1], [-100] * (len(tokenized["labels"]) - 1))
        self.assertEqual(tokenized["labels"][-1:], tokenized["input_ids"][-1:])
        with self.assertRaises(DataValidationError):
            format_sft_record(FakeTokenizer(), SFTRecord("question", "answer"), 2)

    def test_training_loader_projects_only_original_fields(self):
        path = self.root / "train.json"
        path.write_text(json.dumps([{"question": "q", "answer": "C8"}]), encoding="utf-8")
        record = load_training_records(path)[0]
        self.assertEqual(record.question, "q")
        self.assertFalse(hasattr(record, "response"))
        path.write_text(
            json.dumps([{"question": "q", "answer": "C8", "response": "synthetic"}]),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(DataValidationError, "only raw question and answer"):
            load_training_records(path)

    def write_validation(self, questions, targets):
        question_path = self.root / "questions.csv"
        target_path = self.root / "targets.csv"
        with question_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "question"])
            writer.writerows(questions)
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["ID", "Target"])
            writer.writerows(targets)
        return question_path, target_path

    def test_validation_collapses_four_identical_targets(self):
        paths = self.write_validation(
            [("ID_A", "question")], [(f"ID_A_{index}", "C3") for index in range(1, 5)]
        )
        records = load_validation_records(*paths)
        self.assertEqual((records[0].identifier, records[0].answer), ("ID_A", "C3"))
        self.assertEqual(strip_validation_suffix("ID_A_4"), ("ID_A", 4))
        with self.assertRaises(DataValidationError):
            strip_validation_suffix("ID_A_5")

    def test_validation_rejects_inconsistency_missing_duplicate_and_invalid(self):
        cases = [
            ([("ID_A", "q")], [("ID_A_1", "C1"), ("ID_A_2", "C1"), ("ID_A_3", "C1")]),
            (
                [("ID_A", "q")],
                [("ID_A_1", "C1"), ("ID_A_2", "C1"), ("ID_A_3", "C1"), ("ID_A_4", "C2")],
            ),
            (
                [("ID_A", "q")],
                [("ID_B_1", "C1"), ("ID_B_2", "C1"), ("ID_B_3", "C1"), ("ID_B_4", "C1")],
            ),
            ([("ID_A", "q"), ("ID_A", "q2")], [(f"ID_A_{i}", "C1") for i in range(1, 5)]),
            (
                [("ID_A", "q")],
                [("ID_A_1", "C1"), ("ID_A_1", "C1"), ("ID_A_3", "C1"), ("ID_A_4", "C1")],
            ),
            ([("ID_A", "q")], [(f"ID_A_{i}", "C9") for i in range(1, 5)]),
        ]
        for questions, targets in cases:
            with (
                self.subTest(questions=questions, targets=targets),
                self.assertRaises(DataValidationError),
            ):
                load_validation_records(*self.write_validation(questions, targets))


if __name__ == "__main__":
    unittest.main()
