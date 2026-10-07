"""Load a vision-capable Hadamard MLX pack.

The language model's projections are stored in a rotated basis and need the matching activation
transform, which the `Packed` modules in `runtime.py` apply. The vision tower is the stock Qwen
tower: unrotated, unquantized, plain FP16 passthrough. So this builds mlx-vlm's `qwen3_5` model and
installs the packed layers into its language model only.

    from vision_artifact import load_vl_model
    model, processor, config = load_vl_model("/path/to/pack")
"""

import json
from pathlib import Path

import mlx.core as mx


def load_vl_model(directory, load_processor=True):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "prism_hadamard_qwen35":
        raise ValueError("Unsupported packed model schema")
    if not config.get("components", {}).get("vision"):
        raise ValueError("This pack carries no vision tower; use artifact.load_model instead")
    if config.get("base_model_type") != "qwen3_5":
        raise ValueError("Unsupported base model type")

    from mlx_vlm.models.qwen3_5 import Model, ModelConfig
    from runtime import Packed  # the pack's own runtime dir, beside this file

    model = Model(ModelConfig.from_dict(config))
    weights = mx.load(str(directory / "model.safetensors"))

    lm = model.language_model
    seen = set()
    for record in config["modules"]:
        path = record["path"]
        if path in seen:
            raise ValueError("Duplicate packed module")
        seen.add(path)
        parts = path.split(".")
        parent = lm
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        key = "language_model." + path
        arrays = [weights[key + "." + s] for s in ("weight", "scales", "biases")]
        if record["dtype"] != "float16":
            raise ValueError("Unsupported activation dtype")
        block = record["block"]
        if block and block not in (512, 1024, 2048, 4096):
            raise ValueError("Unsupported block size")
        signs = weights.get(key + ".signs")
        if block and signs is None:
            raise ValueError("Missing sign vector")
        if signs is not None and not mx.all((signs == 1) | (signs == -1)).item():
            raise ValueError("Invalid sign values")
        setattr(parent, parts[-1], Packed(arrays, block, signs, record["embedding"], mx.float16))

    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    # mlx-vlm's generate() resets the tokenizer's stopping criteria to model.config.eos_token_id unless
    # the caller passes eos_tokens or stopping_criteria, so the sidecar's list has to live there too.
    eos = generation_eos_ids(directory)
    if eos:
        model.config.eos_token_id = eos

    processor = build_processor(directory) if load_processor else None
    return model, processor, config


def generation_eos_ids(directory):
    """Every end-of-generation id the pack's generation_config.json lists, as a list; [] if none."""
    path = Path(directory) / "generation_config.json"
    if not path.is_file():
        return []
    eos = json.loads(path.read_text()).get("eos_token_id")
    if eos is None:
        return []
    return list(eos) if isinstance(eos, list) else [eos]


def build_processor(directory):
    """Build the Qwen3VL processor without AutoProcessor.

    AutoProcessor resolves classes through the config's `model_type`, which is deliberately ours, and
    it picks the torchvision-backed image processor. Naming the PIL backend explicitly keeps the whole
    path on numpy and Pillow, so nothing here needs torch.
    """
    directory = Path(directory)
    from transformers import AutoTokenizer
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from mlx_vlm.models.qwen3_5 import Qwen3VLProcessor
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    image_processor = Qwen2VLImageProcessorPil.from_pretrained(str(directory))
    tokenizer = AutoTokenizer.from_pretrained(str(directory))
    processor = Qwen3VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=None,
        chat_template=(directory / "chat_template.jinja").read_text(),
    )
    processor.detokenizer = load_tokenizer(directory, return_tokenizer=False)(tokenizer)
    # The pack's generation_config.json lists every end-of-generation id (`<|im_end|>` and
    # `<|endoftext|>`); the HF tokenizer alone knows only its single eos_token. Prefer the file so the
    # runtime stops on both, matching what llama.cpp does with the same weights.
    eos = generation_eos_ids(directory) or (
        getattr(tokenizer, "eos_token_ids", None) or getattr(tokenizer, "eos_token_id", None)
    )
    criteria = StoppingCriteria(eos, tokenizer)
    processor.tokenizer.stopping_criteria = criteria
    processor.stopping_criteria = criteria
    return processor


def chat_config(config):
    """mlx-vlm's prompt helper keys off `model_type`; give it the base type so image tokens land."""
    return {**config, "model_type": config["base_model_type"]}
