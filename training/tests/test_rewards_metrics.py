from __future__ import annotations

import unittest

from rca_training.metrics import compute_metrics, parsed_prediction, unique_plurality
from rca_training.grpo import trim_generated_tokens
from rca_training.rewards import extract_boxed_answer, exact_reward, group_advantages, group_rewards


class RewardTests(unittest.TestCase):
    def test_generated_padding_is_removed_after_first_eos(self):
        self.assertEqual(trim_generated_tokens([10, 2, 2], eos_token_id=2, pad_token_id=2), [10, 2])
        self.assertEqual(trim_generated_tokens([10, 0, 0], eos_token_id=2, pad_token_id=0), [10])

    def test_final_box_exact_matching(self):
        self.assertEqual(extract_boxed_answer("first \\boxed{C1}, final \\boxed{C7}"), "C7")
        self.assertIsNone(extract_boxed_answer("C7"))
        self.assertEqual(exact_reward("\\boxed{C7}", "C7"), 1.0)
        self.assertEqual(exact_reward("\\boxed{ C7 }", "C7"), 0.0)
        self.assertEqual(exact_reward("\\boxed{c7}", "C7"), 0.0)

    def test_group_alignment_and_advantages(self):
        completions = ["\\boxed{C1}"] * 4 + ["\\boxed{C2}"] * 4
        rewards = group_rewards(completions, "C1")
        self.assertEqual(rewards, [1.0] * 4 + [0.0] * 4)
        advantages = group_advantages(rewards)
        self.assertEqual(advantages, [1.0] * 4 + [-1.0] * 4)
        self.assertEqual(group_advantages([1.0] * 8), [0.0] * 8)
        with self.assertRaises(ValueError):
            group_rewards(completions[:7], "C1")


class MetricTests(unittest.TestCase):
    def test_parsing_and_unique_plurality(self):
        self.assertIsNone(parsed_prediction("\\boxed{ C1}"))
        self.assertIsNone(parsed_prediction("missing"))
        self.assertEqual(unique_plurality(["C1", "C1", "C2", "C3"]), "C1")
        self.assertIsNone(unique_plurality(["C1", "C1", "C2", "C2"]))
        self.assertEqual(unique_plurality(["C1", "C1", None, None]), "C1")

    def test_pass_and_majority_metrics(self):
        groups = [
            ("C1", ["\\boxed{C1}"] * 4),
            ("C2", ["\\boxed{C2}", "\\boxed{C2}", "\\boxed{C3}", "bad"]),
            ("C4", ["\\boxed{C4}", "\\boxed{C4}", "\\boxed{C5}", "\\boxed{C5}"]),
        ]
        metrics = compute_metrics(groups)
        self.assertEqual(metrics.correct_samples, 8)
        self.assertEqual(metrics.correct_majorities, 2)
        self.assertEqual(metrics.samples, 12)
        self.assertAlmostEqual(metrics.pass_at_1, 8 / 12)
        self.assertAlmostEqual(metrics.maj_at_4, 2 / 3)


if __name__ == "__main__":
    unittest.main()
