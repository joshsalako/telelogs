from __future__ import annotations

import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from qwen_vllm.config import DeploymentSettings, SettingsError, load_settings


class SettingsTests(unittest.TestCase):
    def test_defaults(self) -> None:
        settings = load_settings({}, dotenv_path=Path("/does/not/exist"))
        self.assertEqual(settings.model, "Qwen/Qwen3.6-27B")
        self.assertEqual(settings.served_model_name, "Qwen3.6-27B")
        self.assertEqual(settings.port, 8000)
        self.assertEqual(settings.max_model_len, 16_384)
        self.assertEqual(settings.gpu_memory_utilization, 0.90)
        self.assertTrue(settings.language_model_only)
        self.assertIsNone(settings.api_key)

    def test_environment_overrides_and_conversions(self) -> None:
        settings = load_settings(
            {
                "QWEN_VLLM_PORT": "9000",
                "QWEN_VLLM_MAX_MODEL_LEN": "8192",
                "QWEN_VLLM_GPU_MEMORY_UTILIZATION": "0.75",
                "QWEN_VLLM_MAX_NUM_SEQS": "3",
                "QWEN_VLLM_LANGUAGE_MODEL_ONLY": "no",
                "QWEN_VLLM_API_KEY": "secret",
            },
            dotenv_path=Path("/does/not/exist"),
        )
        self.assertEqual(settings.port, 9000)
        self.assertEqual(settings.max_model_len, 8192)
        self.assertEqual(settings.gpu_memory_utilization, 0.75)
        self.assertEqual(settings.max_num_seqs, 3)
        self.assertFalse(settings.language_model_only)
        self.assertEqual(settings.api_key, "secret")

    def test_process_environment_wins_over_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "QWEN_VLLM_PORT=8100\n"
                "QWEN_VLLM_API_KEY='from dotenv' # comment\n"
                "HF_TOKEN=hf_test\n"
            )
            settings = load_settings({"QWEN_VLLM_PORT": "8200"}, dotenv_path=path)
        self.assertEqual(settings.port, 8200)
        self.assertEqual(settings.api_key, "from dotenv")
        self.assertEqual(settings.hf_token, "hf_test")

    def test_settings_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            DeploymentSettings().port = 9000  # type: ignore[misc]

    def test_invalid_values_have_named_errors(self) -> None:
        cases = [
            ({"QWEN_VLLM_PORT": "0"}, "QWEN_VLLM_PORT"),
            ({"QWEN_VLLM_PORT": "abc"}, "QWEN_VLLM_PORT"),
            ({"QWEN_VLLM_MAX_MODEL_LEN": "0"}, "QWEN_VLLM_MAX_MODEL_LEN"),
            (
                {"QWEN_VLLM_GPU_MEMORY_UTILIZATION": "1.1"},
                "QWEN_VLLM_GPU_MEMORY_UTILIZATION",
            ),
            ({"QWEN_VLLM_MAX_NUM_SEQS": "0"}, "QWEN_VLLM_MAX_NUM_SEQS"),
            (
                {"QWEN_VLLM_LANGUAGE_MODEL_ONLY": "sometimes"},
                "QWEN_VLLM_LANGUAGE_MODEL_ONLY",
            ),
        ]
        for environment, expected in cases:
            with self.subTest(environment=environment):
                with self.assertRaisesRegex(SettingsError, expected):
                    load_settings(environment, dotenv_path=Path("/does/not/exist"))


if __name__ == "__main__":
    unittest.main()
