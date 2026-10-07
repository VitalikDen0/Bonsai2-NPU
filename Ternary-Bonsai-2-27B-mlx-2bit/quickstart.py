#!/usr/bin/env python3
"""Generate text from this pack. Run: python quickstart.py "your prompt here"

This pack carries its own loader because the weights need transforms that stock
MLX loaders do not apply. That is why `mlx_lm.load()` reports an unsupported
model type; there is no patched mlx-lm to install.

Packs with a vision tower (config.json `components.vision`) load through
runtime/vision_artifact.py on mlx-vlm; text-only packs through runtime/artifact.py.
The sampler comes from generation_config.json and is passed explicitly, because
mlx-lm and mlx-vlm do not read that file.
"""

import json
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parent
sys.path.insert(0, str(PACK / "runtime"))

MAX_TOKENS = 256


def read_json(name, default):
    path = PACK / name
    return json.loads(path.read_text()) if path.is_file() else default


def sampler_settings():
    gen = read_json("generation_config.json", {})
    out = {"temperature": gen.get("temperature", 1.0), "top_p": gen.get("top_p", 0.95), "top_k": gen.get("top_k", 20)}
    if gen.get("do_sample") is False:
        out["temperature"] = 0.0
    if gen.get("min_p"):
        out["min_p"] = gen["min_p"]
    if gen.get("repetition_penalty") not in (None, 1.0):
        out["repetition_penalty"] = gen["repetition_penalty"]
    return out


def render(prompt):
    template_path = PACK / "chat_template.jinja"
    if not template_path.is_file():
        return prompt
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    return (
        ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        .from_string(template_path.read_text())
        .render(
            messages=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    )


def generate_vision(prompt, settings):
    from mlx_vlm import generate
    from vision_artifact import load_vl_model

    model, processor, _ = load_vl_model(PACK)
    result = generate(model, processor, render(prompt), max_tokens=MAX_TOKENS, **settings)
    return result if isinstance(result, str) else result.text


def generate_text(prompt, settings):
    import mlx.core as mx
    from artifact import load_model
    from mlx_lm.sample_utils import make_logits_processors, make_sampler
    from tokenizers import Tokenizer

    model, _ = load_model(PACK)
    tokenizer = Tokenizer.from_file(str(PACK / "tokenizer.json"))
    eos = read_json("generation_config.json", {}).get("eos_token_id")
    stop = set(eos if isinstance(eos, list) else [] if eos is None else [eos])
    stop |= {i for i in (tokenizer.token_to_id("<|im_end|>"), tokenizer.token_to_id("<|endoftext|>")) if i is not None}
    sample = make_sampler(
        temp=settings["temperature"], top_p=settings["top_p"], top_k=settings["top_k"],
        min_p=settings.get("min_p", 0.0),
    )
    processors = make_logits_processors(repetition_penalty=settings.get("repetition_penalty"))

    x = mx.array([tokenizer.encode(render(prompt), add_special_tokens=False).ids])
    cache = model.make_cache()
    generated = []
    for _ in range(MAX_TOKENS):
        logits = model.lm_head(model.model(x, cache=cache)[:, -1:, :])[:, -1, :]
        for processor in processors:
            logits = processor(mx.array(generated), logits)
        x = sample(logits)[:, None]
        mx.eval(x)
        token = int(x.item())
        if token in stop:
            break
        generated.append(token)
    return tokenizer.decode(generated)


def main():
    prompt = " ".join(sys.argv[1:]) or "What is 2 plus 2? Answer with the number only."
    settings = sampler_settings()
    if read_json("config.json", {}).get("components", {}).get("vision"):
        print(generate_vision(prompt, settings))
    else:
        print(generate_text(prompt, settings))


if __name__ == "__main__":
    main()
