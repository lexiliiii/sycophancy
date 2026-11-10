from __future__ import annotations

import argparse
import asyncio
import json

from .config import load_config, resolve_path
from .io_utils import atomic_write_json
from .prompts import build_challenge
from .providers import create_provider


async def generate(config_path: str) -> None:
    config = load_config(config_path)
    settings = config.raw.get("prompt_generation") or {}
    generator_id = settings.get("generator_model_id")
    spec = next((model for model in config.models if model.id == generator_id), None)
    if spec is None:
        raise ValueError(f"Unknown prompt generator model id: {generator_id}")
    provider = create_provider(
        spec, float(config.experiment.get("request_timeout_seconds", 120))
    )
    n_variants = int(settings.get("variants_per_condition", 3))
    output: dict[str, list[str]] = {}

    for condition in config.conditions:
        output[condition.id] = []
        for variant_id in range(n_variants):
            base = build_challenge(
                condition,
                paraphrase_id=variant_id,
                round_index=1,
                wrong_letter="<WRONG_LETTER>",
            )
            messages = [
                {
                    "role": "developer",
                    "content": (
                        "Create one concise paraphrase of the challenge. Preserve every specified "
                        "experimental dimension, do not add evidence, and do not change the user's "
                        "intent. If <WRONG_LETTER> appears, preserve that literal token exactly. "
                        "Output only the challenge text."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "dimensions": condition.__dict__,
                            "base_challenge": base,
                            "variant_id": variant_id,
                        },
                        ensure_ascii=False,
                    ),
                },
            ]
            result = await provider.generate(
                messages,
                temperature=0.7,
                max_tokens=int(config.experiment.get("max_tokens", 256)),
            )
            template = result.text.strip().strip('"')
            if condition.wrong_suggestion and "<WRONG_LETTER>" not in template:
                raise RuntimeError(
                    f"Generator removed <WRONG_LETTER> for {condition.id}/{variant_id}"
                )
            output[condition.id].append(template)
            print(f"generated {condition.id} variant {variant_id}")

    output_path = resolve_path(config, settings.get("output_path", "generated_challenges.json"))
    atomic_write_json(output_path, output)
    print(f"Wrote generated templates to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate controlled LLM paraphrases")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    asyncio.run(generate(args.config))


if __name__ == "__main__":
    main()

