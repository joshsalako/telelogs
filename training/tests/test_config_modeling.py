from __future__ import annotations

import unittest

from rca_training.config import RunConfig
from rca_training.modeling import language_lora_module_names, set_active_adapter


class Linear:
    pass


class FakeModel:
    def __init__(self):
        self.parameters = {
            "base.layer.q_proj.lora_A.default.weight": FakeParameter(),
            "base.layer.q_proj.lora_A.reference.weight": FakeParameter(),
            "base.weight": FakeParameter(),
        }
        self.active = None

    def named_modules(self):
        return [
            ("model.layers.0.q_proj", Linear()),
            ("model.layers.0.down_proj", Linear()),
            ("visual.layers.0.q_proj", Linear()),
            ("model.layers.0.norm", object()),
        ]

    def named_parameters(self):
        return self.parameters.items()

    def set_adapter(self, name):
        self.active = name


class FakeParameter:
    def __init__(self):
        self.requires_grad = False


class ConfigModelingTests(unittest.TestCase):
    def test_resolved_defaults_and_effective_batches(self):
        config = RunConfig()
        self.assertEqual(config.model.model_name, "Qwen/Qwen3.5-4B")
        self.assertEqual(config.sft.effective_batch_size, 128)
        self.assertEqual(config.grpo.effective_response_batch_size, 8)
        self.assertEqual(config.paths.sft_output.name, "SFT_Model")
        self.assertEqual(config.paths.rl_output.name, "RL_Model")
        self.assertEqual(config.to_dict()["model"]["dtype"], "bfloat16")
        config.model.validate()
        config.sft.validate()
        config.evaluation.validate()
        config.grpo.validate()
        config.grpo.gradient_accumulation_steps = 7
        with self.assertRaises(ValueError):
            config.grpo.validate()

    def test_language_lora_excludes_visual_modules(self):
        names = language_lora_module_names(FakeModel(), ("q_proj", "down_proj"))
        self.assertEqual(names, ["model.layers.0.q_proj", "model.layers.0.down_proj"])

    def test_reference_adapter_stays_frozen(self):
        model = FakeModel()
        set_active_adapter(model, "reference", trainable=False)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters.values()))
        set_active_adapter(model, "default", trainable=True)
        self.assertTrue(model.parameters["base.layer.q_proj.lora_A.default.weight"].requires_grad)
        self.assertFalse(
            model.parameters["base.layer.q_proj.lora_A.reference.weight"].requires_grad
        )


if __name__ == "__main__":
    unittest.main()
