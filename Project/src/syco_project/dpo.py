from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io_utils import read_jsonl


def preferred_response(correct_letter: str) -> str:
    return (
        "The challenge does not provide enough valid evidence to overturn the original "
        f"answer. After checking the problem independently, the answer remains {correct_letter}.\n"
        f"FINAL: {correct_letter}"
    )


def export_pairs(run_dir: Path, output: Path, max_pairs: int | None) -> int:
    records = read_jsonl(run_dir / "challenges.jsonl")
    candidates = [
        record
        for record in records
        if record["condition_id"] != "valid_correction"
        and record["baseline_correct"]
        and not record["correct_after"]
        and record["mitigation_type"] == "none"
    ]
    # Keep one pair per exact question/condition/paraphrase/round/model prompt.
    seen: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in candidates:
            key = record["record_key"]
            if key in seen:
                continue
            seen.add(key)
            pair = {
                "prompt": record["messages"],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": preferred_response(record["correct_letter"]),
                    }
                ],
                "rejected": [
                    {"role": "assistant", "content": record["raw_response"]}
                ],
                "metadata": {
                    "model_id": record["model_id"],
                    "question_id": record["question_id"],
                    "condition_id": record["condition_id"],
                    "paraphrase_id": record["paraphrase_id"],
                    "round": record["round"],
                    "correct_letter": record["correct_letter"],
                    "chosen_source": "auditable_template",
                    "rejected_source_record": record["record_key"],
                },
            }
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
            count += 1
            if max_pairs and count >= max_pairs:
                break
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Export conversational DPO pairs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()
    count = export_pairs(
        Path(args.run_dir).resolve(), Path(args.output).resolve(), args.max_pairs
    )
    print(f"Wrote {count} DPO pairs to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

