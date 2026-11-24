from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json, read_jsonl


def _sample_signature(sample: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [(str(item["id"]), str(item["correct"])) for item in sample]


def _merge_jsonl(sources: list[Path], filename: str, output: Path) -> int:
    records: dict[str, dict[str, Any]] = {}
    for source in sources:
        path = source / filename
        if not path.exists():
            if filename == "errors.jsonl":
                continue
            raise FileNotFoundError(f"Missing required file: {path}")
        for record in read_jsonl(path):
            key = record.get("record_key")
            if not key:
                raise ValueError(f"Record without record_key in {path}")
            if key in records:
                raise ValueError(f"Duplicate record key across runs: {key}")
            records[key] = record

    target = output / filename
    with target.open("w", encoding="utf-8") as stream:
        for key in sorted(records):
            stream.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    return len(records)


def merge_runs(run_dirs: list[Path], output_dir: Path) -> dict[str, int]:
    if len(run_dirs) < 2:
        raise ValueError("Provide at least two run directories")
    resolved = [path.resolve() for path in run_dirs]
    output = output_dir.resolve()
    if output in resolved:
        raise ValueError("Output directory must differ from every input run")

    manifests = []
    samples = []
    for run in resolved:
        manifests.append(json.loads((run / "manifest.json").read_text(encoding="utf-8")))
        samples.append(json.loads((run / "sample.json").read_text(encoding="utf-8")))

    reference_signature = _sample_signature(samples[0])
    for run, sample in zip(resolved[1:], samples[1:]):
        if _sample_signature(sample) != reference_signature:
            raise ValueError(f"Question sample differs in {run}")

    rounds = {int(manifest["rounds"]) for manifest in manifests}
    seeds = {int(manifest["seed"]) for manifest in manifests}
    paraphrases = {int(manifest["paraphrases_per_condition"]) for manifest in manifests}
    if len(rounds) != 1 or len(seeds) != 1 or len(paraphrases) != 1:
        raise ValueError("Runs must use the same seed, rounds, and paraphrase count")

    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "sample.json", samples[0])
    merged_manifest = dict(manifests[0])
    merged_manifest["model_ids"] = sorted(
        {model for manifest in manifests for model in manifest.get("model_ids", [])}
    )
    merged_manifest["merged_from"] = [str(path) for path in resolved]
    atomic_write_json(output / "manifest.json", merged_manifest)

    counts = {
        "baselines": _merge_jsonl(resolved, "baselines.jsonl", output),
        "challenges": _merge_jsonl(resolved, "challenges.jsonl", output),
        "errors": _merge_jsonl(resolved, "errors.jsonl", output),
    }
    analysis = output / "analysis"
    if analysis.exists():
        shutil.rmtree(analysis)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge compatible per-model experiment runs for joint analysis"
    )
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    counts = merge_runs(
        [Path(path) for path in args.run_dir], Path(args.output_dir)
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
