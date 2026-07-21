from __future__ import annotations

import unittest

from qwen_vllm.command import build_serve_command, build_serve_environment
from qwen_vllm.config import DeploymentSettings


class CommandTests(unittest.TestCase):
    def test_default_command_is_exact(self) -> None:
        self.assertEqual(
            build_serve_command(DeploymentSettings()),
            [
                "vllm",
                "serve",
                "Qwen/Qwen3.6-27B",
                "--served-model-name",
                "Qwen3.6-27B",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
                "--tensor-parallel-size",
                "1",
                "--max-model-len",
                "16384",
                "--gpu-memory-utilization",
                "0.9",
                "--max-num-seqs",
                "8",
                "--reasoning-parser",
                "qwen3",
                "--quantization",
                "bitsandbytes",
                "--language-model-only",
            ],
        )

    def test_api_key_is_optional(self) -> None:
        command = build_serve_command(DeploymentSettings(api_key="token"))
        self.assertEqual(command[-2:], ["--api-key", "token"])

    def test_text_only_flag_can_be_disabled(self) -> None:
        command = build_serve_command(DeploymentSettings(language_model_only=False))
        self.assertNotIn("--language-model-only", command)

    def test_hugging_face_settings_are_forwarded_to_server(self) -> None:
        settings = DeploymentSettings(
            hf_token="hf_secret",
            hf_home="/model-cache",
            huggingface_hub_cache="/model-cache/hub",
        )
        environment = build_serve_environment(settings, {"PATH": "/bin"})
        self.assertEqual(environment["PATH"], "/bin")
        self.assertEqual(environment["HF_TOKEN"], "hf_secret")
        self.assertEqual(environment["HF_HOME"], "/model-cache")
        self.assertEqual(environment["HUGGINGFACE_HUB_CACHE"], "/model-cache/hub")


if __name__ == "__main__":
    unittest.main()
