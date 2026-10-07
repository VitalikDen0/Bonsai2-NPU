"""Experimental MLX runtime for folded dense qwen35 GGUF checkpoints."""

import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from codec import transcode
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

from mlx import nn


def fwht(x, block, signs, inverse=False):
    shape, dtype = x.shape, x.dtype
    if shape[-1] % block:
        raise ValueError("Hadamard block does not divide activation width")
    x = x.astype(mx.float32)
    if not inverse:
        x = x * signs
    x = mx.hadamard_transform(x.reshape(-1, block), scale=1 / math.sqrt(block)).reshape(
        shape
    )
    if inverse:
        x = x * signs
    return x.astype(dtype)


class Packed(nn.Module):
    def __init__(self, arrays, block=0, signs=None, embedding=False, dtype=mx.float16):
        super().__init__()
        self.weight, self.scales, self.biases = [mx.array(a) for a in arrays]
        self.block, self.signs, self.embedding, self.dtype = (
            block,
            signs,
            embedding,
            dtype,
        )

    def __call__(self, x):
        if self.embedding:
            shape = x.shape
            indices = x.reshape(-1)
            out = (
                mx.dequantize(
                    self.weight[indices],
                    self.scales[indices],
                    self.biases[indices],
                    group_size=128,
                    bits=2,
                )
                .reshape(*shape, -1)
                .astype(self.dtype)
            )
            return (
                fwht(out, self.block, self.signs, inverse=True) if self.block else out
            )
        if self.block:
            x = fwht(x, self.block, self.signs)
        return mx.quantized_matmul(
            x,
            self.weight,
            self.scales,
            self.biases,
            transpose=True,
            group_size=128,
            bits=2,
        )


def load(gguf_path, gguf_python, dtype=mx.float16):
    sys.path.insert(0, str(gguf_python))
    from gguf import GGUFReader

    reader = GGUFReader(str(gguf_path))
    fields = {k: v.contents() for k, v in reader.fields.items()}
    if fields.get("general.architecture") != "qwen35":
        raise ValueError("Only dense qwen35 is supported by this experimental loader")
    g = lambda k, default=None: fields.get("qwen35." + k, default)
    tensors = {t.name: t for t in reader.tensors}
    if "output.weight" not in tensors:
        raise ValueError(
            "Explicit output head required; tied-head policy must be resolved separately"
        )
    block = int(fields["prism.hadamard.block_size"])
    if block not in (512, 1024, 2048, 4096):
        raise ValueError("Unvalidated Hadamard block size")
    if fields.get("prism.hadamard.sign_mode") != "explicit":
        raise ValueError("Explicit signs required")
    widths = fields["prism.hadamard.sign_widths"]
    values = fields["prism.hadamard.sign_values"]
    signs, offset = {}, 0
    for width in widths:
        a = np.asarray(values[offset : offset + width], dtype=np.float32)
        if len(a) != width or not np.isin(a, [-1, 1]).all():
            raise ValueError("Invalid sign vector")
        signs[width] = mx.array(a)
        offset += width
    if offset != len(values):
        raise ValueError("Trailing sign values")
    folded = set(fields["prism.hadamard.weight_names"])
    inverse = set(fields.get("prism.hadamard.inverse_weight_names", []))
    if inverse != {"token_embd.weight"}:
        raise ValueError("Unexpected inverse-transform manifest")
    if folded & inverse:
        raise ValueError("Forward and inverse transform manifests overlap")
    if not (folded | inverse) <= tensors.keys():
        raise ValueError("Transform manifest references missing tensors")
    nv, nk = int(g("ssm.time_step_rank")), int(g("ssm.group_count"))
    if nv <= 0 or nk <= 0 or nv % nk or int(g("ssm.inner_size")) % nv:
        raise ValueError("Invalid GDN head dimensions")
    hd = int(g("ssm.inner_size")) // nv
    hk = int(g("ssm.state_size"))
    cfg = {
        "model_type": "qwen3_5_text",
        "hidden_size": int(g("embedding_length")),
        "intermediate_size": int(g("feed_forward_length")),
        "num_hidden_layers": int(g("block_count")),
        "num_attention_heads": int(g("attention.head_count")),
        "num_key_value_heads": int(g("attention.head_count_kv")),
        "head_dim": int(g("attention.key_length")),
        "rms_norm_eps": float(g("attention.layer_norm_rms_epsilon")),
        "vocab_size": int(tensors["token_embd.weight"].shape[1]),
        "max_position_embeddings": int(g("context_length")),
        "linear_num_value_heads": nv,
        "linear_num_key_heads": nk,
        "linear_value_head_dim": hd,
        "linear_key_head_dim": hk,
        "linear_conv_kernel_dim": int(g("ssm.conv_kernel")),
        "full_attention_interval": int(g("full_attention_interval")),
        "tie_word_embeddings": False,
        "rope_parameters": {
            "type": "default",
            "rope_theta": float(g("rope.freq_base")),
            "partial_rotary_factor": int(g("rope.dimension_count"))
            / int(g("attention.key_length")),
        },
    }
    model = TextModel(TextModelArgs.from_dict(cfg))
    expected = {name: value.shape for name, value in tree_flatten(model.parameters())}
    mapping = {
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
        "attn_gate.weight": "linear_attn.in_proj_z.weight",
        "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
        "ssm_beta.weight": "linear_attn.in_proj_b.weight",
        "ssm_out.weight": "linear_attn.out_proj.weight",
        "ssm_norm.weight": "linear_attn.norm.weight",
        "ssm_a": "linear_attn.A_log",
        "ssm_dt.bias": "linear_attn.dt_bias",
        "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    }
    global_map = {
        "output.weight": "lm_head.weight",
        "output_norm.weight": "model.norm.weight",
        "token_embd.weight": "model.embed_tokens.weight",
    }

    def vperm(unit):
        return (
            np.arange(nv * unit)
            .reshape(nv // nk, nk, unit)
            .transpose(1, 0, 2)
            .reshape(-1)
        )

    def reorder(a, stem):
        if nv == nk:
            return a
        if stem == "attn_qkv.weight":
            qk = 2 * nk * hk
            return np.concatenate([a[:qk], a[qk:][vperm(hd)]], axis=0)
        if stem == "attn_gate.weight":
            return a[vperm(hd)]
        if stem in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias"):
            return a[vperm(1)]
        if stem == "ssm_conv1d.weight":
            qk = 2 * nk * hk
            return np.concatenate([a[:qk], a[qk:][vperm(hd)]], axis=0)
        return a

    assigned = []
    for name, t in tensors.items():
        stem = name
        if name in global_map:
            target = global_map[name]
        elif name.startswith("blk."):
            _, layer, stem = name.split(".", 2)
            if stem not in mapping:
                raise ValueError("Unmapped tensor " + name)
            target = f"model.layers.{layer}." + mapping[stem]
        else:
            raise ValueError("Unmapped tensor " + name)
        if target not in expected or target in assigned:
            raise ValueError("Unexpected or duplicate target " + target)
        shape = tuple(int(n) for n in t.shape[::-1])
        if stem == "ssm_conv1d.weight":
            shape += (1,)
        if shape != expected[target]:
            raise ValueError(
                f"Shape mismatch for {name}: {shape} != {expected[target]}"
            )
        parent = model
        parts = target.split(".")
        for part in (
            parts[:-2] if t.tensor_type.name in ("PQ2_0", "PTQ1_0") else parts[:-1]
        ):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        if t.tensor_type.name in ("PQ2_0", "PTQ1_0"):
            if len(t.shape) != 2:
                raise ValueError("Quantized tensor is not a matrix")
            shape = tuple(int(n) for n in t.shape[::-1])
            arrays = transcode(t.data.tobytes(), shape, t.tensor_type.name)
            arrays = tuple(reorder(a, stem) for a in arrays)
            if (
                stem == "ssm_out.weight"
                and nv != nk
                and not fields.get("prism.hadamard.gdn_v_grouped", False)
            ):
                raise ValueError("Unimplemented ungrouped folded GDN output")
            active = name in folded or name in inverse
            module = Packed(
                arrays,
                block if active else 0,
                signs[shape[1]] if active else None,
                name in inverse,
                dtype,
            )
            setattr(parent, parts[-2], module)
            mx.eval(module.parameters())
        else:
            if t.tensor_type.name not in ("F32", "F16"):
                raise ValueError("Unsupported auxiliary type " + t.tensor_type.name)
            a = np.asarray(t.data).copy()
            a = reorder(a, stem)
            if stem == "ssm_a":
                if not (a < 0).all():
                    raise ValueError("Invalid stored SSM A")
                a = np.log(-a)
            if stem == "ssm_conv1d.weight":
                a = a[..., None]
            if name in folded or name in inverse:
                raise ValueError("Unimplemented transformed float matrix")
            setattr(parent, parts[-1], mx.array(a))
        assigned.append(target)
    missing = expected.keys() - set(assigned)
    if missing:
        raise ValueError(f"Missing model parameters: {sorted(missing)}")
    model.eval()
    mx.eval(model.parameters())
    return (
        model,
        {
            "config": cfg,
            "block": block,
            "source": str(Path(gguf_path).resolve()),
            "assigned_tensors": len(assigned),
            "folded": len(folded),
            "inverse": len(inverse),
        },
        reader,
    )
