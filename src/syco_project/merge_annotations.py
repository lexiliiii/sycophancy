from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_LABELS = {
    "held_firm",
    "agreement_without_error",
    "correct_to_incorrect",
    "still_incorrect_or_unparsed",
}


def cohen_kappa(first: pd.Series, second: pd.Series) -> float:
    labels = sorted(set(first) | set(second))
    if not labels:
        return float("nan")
    observed = float((first == second).mean())
    first_prob = first.value_counts(normalize=True)
    second_prob = second.value_counts(normalize=True)
    expected = sum(first_prob.get(label, 0.0) * second_prob.get(label, 0.0) for label in labels)
    if np.isclose(expected, 1.0):
        return 1.0 if np.isclose(observed, 1.0) else float("nan")
    return (observed - expected) / (1.0 - expected)


def merge_annotation_files(first_path: Path, second_path: Path, output: Path) -> dict:
    first = pd.read_csv(first_path)
    second = pd.read_csv(second_path)
    required_first = {"record_key", "annotator_1"}
    required_second = {"record_key", "annotator_2"}
    if not required_first.issubset(first.columns):
        raise ValueError(f"First file needs columns {sorted(required_first)}")
    if not required_second.issubset(second.columns):
        raise ValueError(f"Second file needs columns {sorted(required_second)}")
    if first["record_key"].duplicated().any() or second["record_key"].duplicated().any():
        raise ValueError("Annotation files contain duplicate record_key values")

    second_labels = second[["record_key", "annotator_2"]]
    merged = first.drop(columns=["annotator_2", "manual_outcome"], errors="ignore").merge(
        second_labels, on="record_key", how="inner", validate="one_to_one"
    )
    if len(merged) != len(first) or len(merged) != len(second):
        raise ValueError("Annotator files do not contain exactly the same record keys")
    for column in ["annotator_1", "annotator_2"]:
        merged[column] = merged[column].fillna("").astype(str).str.strip()
        invalid = set(merged[column]) - ALLOWED_LABELS
        if invalid:
            raise ValueError(f"Invalid or empty labels in {column}: {sorted(invalid)}")

    agreement = merged["annotator_1"] == merged["annotator_2"]
    merged["manual_outcome"] = np.where(agreement, merged["annotator_1"], "")
    merged.to_csv(output, index=False)
    return {
        "rows": len(merged),
        "raw_agreement": float(agreement.mean()),
        "cohen_kappa": float(cohen_kappa(merged["annotator_1"], merged["annotator_2"])),
        "needs_adjudication": int((~agreement).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge two annotation copies and calculate inter-annotator agreement"
    )
    parser.add_argument("--annotator-1", required=True)
    parser.add_argument("--annotator-2", required=True)
    parser.add_argument("--output", default="annotations_for_adjudication.csv")
    args = parser.parse_args()
    metrics = merge_annotation_files(
        Path(args.annotator_1), Path(args.annotator_2), Path(args.output)
    )
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
