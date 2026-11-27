"""用一个小型 Safetensors 文件验收 GPT-2 权重加载。"""

import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch
from safetensors.torch import save_file

from gpt2.model import GPT2
from gpt2.weights import load_gpt2_weights


class GPT2WeightTests(unittest.TestCase):
    def setUp(self):
        temp_dir = TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        self.path = Path(temp_dir.name) / "model.safetensors"
        self.model = GPT2(11, 6, 4, num_heads=2, num_layers=2)

        # 先按公开文件的命名和形状造张量。各层数值不同，方阵也不对称。
        self.checkpoint = {
            "wte.weight": torch.arange(44, dtype=torch.float32).reshape(11, 4),
            "wpe.weight": torch.arange(24, dtype=torch.float32).reshape(6, 4),
            "ln_f.weight": torch.arange(4, dtype=torch.float32),
            "ln_f.bias": torch.arange(4, dtype=torch.float32) + 10,
        }
        layer_shapes = {
            "ln_1.weight": (4,),
            "ln_1.bias": (4,),
            "attn.c_attn.weight": (4, 12),
            "attn.c_attn.bias": (12,),
            "attn.c_proj.weight": (4, 4),
            "attn.c_proj.bias": (4,),
            "ln_2.weight": (4,),
            "ln_2.bias": (4,),
            "mlp.c_fc.weight": (4, 16),
            "mlp.c_fc.bias": (16,),
            "mlp.c_proj.weight": (16, 4),
            "mlp.c_proj.bias": (4,),
        }
        for layer in range(2):
            for number, (name, shape) in enumerate(layer_shapes.items()):
                values = torch.arange(math.prod(shape), dtype=torch.float32)
                offset = 100 * number + 1000 * layer
                self.checkpoint[f"h.{layer}.{name}"] = values.reshape(shape) + offset
            self.checkpoint[f"h.{layer}.attn.bias"] = torch.ones(1, 1, 6, 6).tril()

    def test_copies_embeddings_norms_and_all_projections(self):
        save_file(self.checkpoint, self.path)
        load_gpt2_weights(self.model, self.path)

        actual = self.model.state_dict()
        torch.testing.assert_close(actual["embeddings.wpe.weight"], self.checkpoint["wpe.weight"])
        torch.testing.assert_close(actual["ln_f.weight"], self.checkpoint["ln_f.weight"])
        torch.testing.assert_close(actual["ln_f.bias"], self.checkpoint["ln_f.bias"])
        for index in range(2):
            for name in ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"):
                with self.subTest(layer=index, projection=name):
                    target = f"blocks.{index}.{name}"
                    source = f"h.{index}.{name}"
                    torch.testing.assert_close(
                        actual[f"{target}.weight"], self.checkpoint[f"{source}.weight"].T
                    )
                    torch.testing.assert_close(
                        actual[f"{target}.bias"], self.checkpoint[f"{source}.bias"]
                    )
            for name in ("ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias"):
                torch.testing.assert_close(
                    actual[f"blocks.{index}.{name}"], self.checkpoint[f"h.{index}.{name}"]
                )

    def test_weight_sharing_survives_when_mask_buffers_are_omitted(self):
        tensors = {
            key: value for key, value in self.checkpoint.items() if not key.endswith(".attn.bias")
        }
        save_file(tensors, self.path)

        original_parameter = self.model.embeddings.wte.weight
        load_gpt2_weights(self.model, self.path)

        self.assertIs(self.model.embeddings.wte.weight, original_parameter)
        self.assertIs(self.model.lm_head.weight, original_parameter)
        torch.testing.assert_close(original_parameter, tensors["wte.weight"])

    def test_reports_missing_and_unexpected_checkpoint_keys(self):
        self.checkpoint.pop("h.1.ln_2.bias")
        self.checkpoint["h.2.ln_2.bias"] = torch.zeros(4)
        save_file(self.checkpoint, self.path)

        with self.assertRaisesRegex(
            ValueError, r"missing=.*h\.1\.ln_2\.bias.*unexpected=.*h\.2\.ln_2\.bias"
        ):
            load_gpt2_weights(self.model, self.path)

    def test_reports_wrong_checkpoint_shape_before_mutating_model(self):
        original_wte = self.model.embeddings.wte.weight.detach().clone()
        self.checkpoint["h.1.mlp.c_proj.weight"] = torch.zeros(4, 16)
        save_file(self.checkpoint, self.path)

        with self.assertRaisesRegex(ValueError, r"h\.1\.mlp\.c_proj\.weight: expected shape \(16, 4\)"):
            load_gpt2_weights(self.model, self.path)
        torch.testing.assert_close(self.model.embeddings.wte.weight, original_wte)
