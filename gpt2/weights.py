"""将公开的 GPT-2 Safetensors 权重加载到本项目的 GPT2。"""

from pathlib import Path

from safetensors import safe_open

from gpt2.model import GPT2


# 将文件中的名字和矩阵布局转换成 GPT2 的 state_dict，再复制到已有参数中。
def load_gpt2_weights(model: GPT2, checkpoint_path: str | Path) -> None:
    """加载 openai-community/gpt2 的 model.safetensors 到已初始化的模型。

    模型配置须与 checkpoint 对应。词表和位置表分别是 (V, C) 和 (P, C)。
    文件中的 Conv1D 权重为 (输入宽度, 输出宽度)，nn.Linear 使用相反的轴顺序。
    键和形状在写入前检查。复制参数保留 wte 与 lm_head 的权重共享。
    """
    source_names = {
        "embeddings.wte.weight": "wte.weight",
        "embeddings.wpe.weight": "wpe.weight",
        "ln_f.weight": "ln_f.weight",
        "ln_f.bias": "ln_f.bias",
    }
    mask_names = set()
    for index, block in enumerate(model.blocks):
        for suffix in block.state_dict():
            source_names[f"blocks.{index}.{suffix}"] = f"h.{index}.{suffix}"
        # 公开文件存了这个旧版因果遮罩，本项目的注意力会按输入长度生成。
        mask_names.add(f"h.{index}.attn.bias")

    conv_weights = (
        ".attn.c_attn.weight",
        ".attn.c_proj.weight",
        ".mlp.c_fc.weight",
        ".mlp.c_proj.weight",
    )
    target_state = model.state_dict()
    with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
        expected_names = set(source_names.values())
        file_names = set(checkpoint.keys())
        missing = expected_names - file_names
        unexpected = file_names - expected_names - mask_names
        if missing or unexpected:
            raise ValueError(
                f"Invalid GPT-2 checkpoint keys: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )

        mapped = {}
        for target_name, source_name in source_names.items():
            tensor = checkpoint.get_tensor(source_name)
            needs_transpose = source_name.endswith(conv_weights)
            expected_shape = target_state[target_name].shape
            if needs_transpose:
                expected_shape = expected_shape[::-1]
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{source_name}: expected shape {tuple(expected_shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            mapped[target_name] = tensor.T if needs_transpose else tensor

        # strict=True 需要两个键。默认的复制行为保留模型里的同一个 Parameter。
        mapped["lm_head.weight"] = mapped["embeddings.wte.weight"]
        model.load_state_dict(mapped, strict=True)
