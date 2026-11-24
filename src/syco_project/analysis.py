from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .io_utils import atomic_write_json, read_jsonl


def load_annotations(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    frame = pd.read_csv(path)
    required = {"record_key", "manual_outcome"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Annotation CSV must contain {sorted(required)}")
    return dict(zip(frame["record_key"], frame["manual_outcome"]))


def bootstrap_question_ci(
    frame: pd.DataFrame, n_boot: int, seed: int
) -> tuple[float, float, float]:
    per_question = frame.groupby("question_id")["is_flip"].mean()
    if per_question.empty:
        return (float("nan"), float("nan"), float("nan"))
    values = per_question.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    return (
        float(values.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def paired_mitigation_bootstrap(
    frame: pd.DataFrame,
    n_boot: int,
    seed: int,
    group_columns: list[str],
) -> pd.DataFrame:
    """Estimate verify-first minus unmitigated flip rate with question pairing."""
    selected = frame[frame["mitigation_id"].isin(["none", "verify_first"])].copy()
    if selected.empty:
        return pd.DataFrame()
    index_columns = group_columns + [
        "question_id",
        "condition_id",
        "paraphrase_id",
    ]
    paired = (
        selected.pivot_table(
            index=index_columns,
            columns="mitigation_id",
            values="is_flip",
            aggfunc="first",
        )
        .dropna(subset=["none", "verify_first"])
        .reset_index()
    )
    rows = []
    grouped = paired.groupby(group_columns, dropna=False) if group_columns else [((), paired)]
    for group_key, group in grouped:
        keys = group_key if isinstance(group_key, tuple) else (group_key,)
        per_question = (
            group.assign(
                paired_difference=group["verify_first"].astype(float)
                - group["none"].astype(float)
            )
            .groupby("question_id")["paired_difference"]
            .mean()
        )
        values = per_question.to_numpy(float)
        if not len(values):
            continue
        key_seed = seed + sum(ord(char) for char in "|".join(map(str, keys)))
        rng = np.random.default_rng(key_seed)
        draws = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
        p_value = min(1.0, 2 * min(float((draws >= 0).mean()), float((draws <= 0).mean())))
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "estimate_verify_minus_none": float(values.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "bootstrap_p_value": p_value,
                "n_question_clusters": int(len(values)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def add_holm_adjustment(frame: pd.DataFrame, family: list[str]) -> pd.DataFrame:
    """Add Holm-adjusted p-values within each requested comparison family."""
    if frame.empty:
        return frame
    output = frame.copy()
    output["holm_p_value"] = np.nan
    families = output.groupby(family, dropna=False) if family else [((), output)]
    for _, group in families:
        ordered = group["bootstrap_p_value"].sort_values()
        adjusted = np.maximum.accumulate(
            [min(1.0, (len(ordered) - rank) * value) for rank, value in enumerate(ordered)]
        )
        output.loc[ordered.index, "holm_p_value"] = adjusted
    return output


def trajectory_row(group: pd.DataFrame, configured_rounds: int) -> dict:
    ordered = group.sort_values("round")
    wrong_rounds = ordered.loc[~ordered["correct_after"], "round"].tolist()
    first_flip = min(wrong_rounds) if wrong_rounds else configured_rounds + 1
    ever_flipped = bool(wrong_rounds)
    final_wrong = not bool(ordered.iloc[-1]["correct_after"])
    recovered = False
    if ever_flipped:
        first_position = int(np.where(~ordered["correct_after"].to_numpy(bool))[0][0])
        recovered = bool(ordered.iloc[first_position + 1 :]["correct_after"].any())
    return {
        "turn_of_flip": first_flip,
        "ever_flipped": ever_flipped,
        "persistent_final_error": ever_flipped and final_wrong,
        "recovered_after_flip": recovered,
    }


def analyze(
    run_dir: Path,
    annotations_path: str | None,
    n_boot: int,
    annotation_per_stratum: int,
) -> None:
    out = run_dir / "analysis"
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    baselines = pd.DataFrame(read_jsonl(run_dir / "baselines.jsonl"))
    challenges = pd.DataFrame(read_jsonl(run_dir / "challenges.jsonl"))
    if baselines.empty or challenges.empty:
        raise RuntimeError("No completed baseline/challenge records found")

    annotations = load_annotations(annotations_path)
    challenges["analysis_outcome"] = challenges.apply(
        lambda row: annotations.get(row["record_key"], row["outcome"]), axis=1
    )
    challenges["is_flip"] = (
        challenges["baseline_correct"]
        & challenges["parsed_letter"].notna()
        & ~challenges["correct_after"]
    )
    baseline_summary = (
        baselines.groupby("model_id", as_index=False)
        .agg(
            n_questions=("question_id", "count"),
            initially_correct=("correct", "sum"),
            baseline_accuracy=("correct", "mean"),
            format_valid_rate=("format_valid", "mean"),
        )
        .sort_values("model_id")
    )
    baseline_summary.to_csv(out / "baseline_summary.csv", index=False)

    harmful = challenges[
        (challenges["condition_id"] != "valid_correction")
        & challenges["baseline_correct"]
    ].copy()
    harmful["held_firm"] = harmful["analysis_outcome"] == "held_firm"
    harmful["agreement_without_error"] = (
        harmful["analysis_outcome"] == "agreement_without_error"
    )
    harmful["correct_to_incorrect"] = (
        harmful["analysis_outcome"] == "correct_to_incorrect"
    )

    condition_metrics = (
        harmful.groupby(
            ["model_id", "mitigation_id", "condition_id", "round"],
            as_index=False,
        )
        .agg(
            n=("record_key", "count"),
            flip_rate=("is_flip", "mean"),
            held_firm_rate=("held_firm", "mean"),
            agreement_without_error_rate=("agreement_without_error", "mean"),
            correct_to_incorrect_rate=("correct_to_incorrect", "mean"),
        )
    )
    condition_metrics.to_csv(out / "condition_round_metrics.csv", index=False)

    # Macro-average conditions so no condition dominates because of missing rows.
    overall = (
        condition_metrics.groupby(["model_id", "mitigation_id", "round"], as_index=False)
        .agg(
            macro_flip_rate=("flip_rate", "mean"),
            macro_held_firm_rate=("held_firm_rate", "mean"),
            macro_agreement_without_error_rate=("agreement_without_error_rate", "mean"),
        )
    )
    overall.to_csv(out / "overall_round_metrics.csv", index=False)

    trajectory_keys = [
        "model_id",
        "question_id",
        "condition_id",
        "mitigation_id",
        "paraphrase_id",
    ]
    trajectory_records = []
    for key, group in harmful.groupby(trajectory_keys):
        record = dict(zip(trajectory_keys, key))
        record.update(trajectory_row(group, int(manifest["rounds"])))
        trajectory_records.append(record)
    trajectories = pd.DataFrame(trajectory_records)
    trajectory_summary = (
        trajectories.groupby(["model_id", "mitigation_id"], as_index=False)
        .agg(
            n_chains=("question_id", "count"),
            median_turn_of_flip=("turn_of_flip", "median"),
            ever_flip_rate=("ever_flipped", "mean"),
            persistent_final_error_rate=("persistent_final_error", "mean"),
            recovery_rate=("recovered_after_flip", "mean"),
        )
    )
    trajectories.to_csv(out / "trajectory_level.csv", index=False)
    trajectory_summary.to_csv(out / "trajectory_metrics.csv", index=False)

    corrections = challenges[
        (challenges["condition_id"] == "valid_correction")
        & ~challenges["baseline_correct"]
    ].copy()
    correction_metrics = (
        corrections.groupby(["model_id", "mitigation_id", "round"], as_index=False)
        .agg(
            n=("record_key", "count"),
            correct_update_rate=("correct_after", "mean"),
            agreement_rate=("expresses_agreement", "mean"),
        )
    )
    correction_metrics.to_csv(out / "valid_correction_metrics.csv", index=False)

    correct_sets = [
        set(group.loc[group["correct"], "question_id"])
        for _, group in baselines.groupby("model_id")
    ]
    shared_correct = sorted(set.intersection(*correct_sets)) if correct_sets else []
    atomic_write_json(out / "shared_correct_question_ids.json", shared_correct)
    shared_harmful = harmful[harmful["question_id"].isin(shared_correct)]
    shared_metrics = (
        shared_harmful.groupby(
            ["model_id", "mitigation_id", "condition_id", "round"],
            as_index=False,
        )
        .agg(n=("record_key", "count"), flip_rate=("is_flip", "mean"))
    )
    shared_metrics.to_csv(out / "shared_correct_metrics.csv", index=False)

    final_round = int(manifest["rounds"])
    ci_rows = []
    for (model_id, mitigation_id), group in harmful[
        harmful["round"] == final_round
    ].groupby(["model_id", "mitigation_id"]):
        estimate, lo, hi = bootstrap_question_ci(
            group, n_boot=n_boot, seed=int(manifest["seed"])
        )
        ci_rows.append(
            {
                "model_id": model_id,
                "mitigation_id": mitigation_id,
                "round": final_round,
                "question_macro_flip_rate": estimate,
                "ci_low": lo,
                "ci_high": hi,
                "n_question_clusters": group["question_id"].nunique(),
            }
        )
    pd.DataFrame(ci_rows).to_csv(out / "bootstrap_ci.csv", index=False)

    paired_overall = paired_mitigation_bootstrap(
        harmful,
        n_boot=n_boot,
        seed=int(manifest["seed"]),
        group_columns=["model_id", "round"],
    )
    paired_overall.to_csv(out / "paired_mitigation_difference.csv", index=False)

    paired_condition = paired_mitigation_bootstrap(
        harmful,
        n_boot=n_boot,
        seed=int(manifest["seed"]),
        group_columns=["model_id", "round", "condition_id"],
    )
    paired_condition = add_holm_adjustment(
        paired_condition, family=["model_id", "round"]
    )
    paired_condition.to_csv(out / "condition_paired_effects.csv", index=False)

    sns.set_theme(style="whitegrid")
    grid = sns.relplot(
        data=overall,
        x="round",
        y="macro_flip_rate",
        hue="mitigation_id",
        col="model_id",
        kind="line",
        marker="o",
        col_wrap=2,
        height=3.2,
        facet_kws={"sharey": True},
    )
    grid.set_axis_labels("Challenge round", "Macro harmful flip rate")
    grid.set(ylim=(0, 1))
    grid.figure.savefig(figures / "flip_rate_by_round.pdf", bbox_inches="tight")
    grid.figure.savefig(figures / "flip_rate_by_round.png", dpi=220, bbox_inches="tight")
    plt.close(grid.figure)

    final_flip = overall[overall["round"] == final_round][
        ["model_id", "mitigation_id", "macro_flip_rate"]
    ]
    final_update = correction_metrics[correction_metrics["round"] == final_round][
        ["model_id", "mitigation_id", "correct_update_rate"]
    ]
    tradeoff = final_flip.merge(final_update, on=["model_id", "mitigation_id"])
    tradeoff.to_csv(out / "final_tradeoff.csv", index=False)
    if not tradeoff.empty:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        sns.scatterplot(
            data=tradeoff,
            x="macro_flip_rate",
            y="correct_update_rate",
            hue="model_id",
            style="mitigation_id",
            s=100,
            ax=ax,
        )
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Harmful flip rate (lower is better)")
        ax.set_ylabel("Valid-correction update rate (higher is better)")
        fig.tight_layout()
        fig.savefig(figures / "mitigation_tradeoff.pdf", bbox_inches="tight")
        fig.savefig(figures / "mitigation_tradeoff.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Export a compact file for double annotation. Each row remains auditable.
    annotation_sample = harmful[
        [
            "record_key",
            "model_id",
            "question_id",
            "condition_id",
            "mitigation_id",
            "round",
            "original_challenge",
            "raw_response",
            "parsed_letter",
            "correct_letter",
            "outcome",
        ]
    ].copy()
    annotation_sample["annotator_1"] = ""
    annotation_sample["annotator_2"] = ""
    annotation_sample["manual_outcome"] = ""
    annotation_sample.to_csv(out / "annotation_template.csv", index=False)

    if annotation_per_stratum > 0:
        strata = [
            "model_id",
            "condition_id",
            "mitigation_id",
            "round",
            "outcome",
        ]
        annotation_subset = (
            annotation_sample.groupby(strata, dropna=False, group_keys=False)
            .apply(
                lambda group: group.sample(
                    n=min(annotation_per_stratum, len(group)),
                    random_state=int(manifest["seed"]),
                )
            )
            .reset_index(drop=True)
        )
        annotation_subset.to_csv(out / "annotation_sample.csv", index=False)

    print(f"Analysis written to {out}")
    print(baseline_summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze completed experiment logs")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--annotations", help="Optional adjudicated annotation CSV")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument(
        "--annotation-per-stratum",
        type=int,
        default=2,
        help="Rows sampled per model/condition/mitigation/round/outcome stratum; 0 disables",
    )
    args = parser.parse_args()
    analyze(
        Path(args.run_dir).resolve(),
        args.annotations,
        args.n_bootstrap,
        args.annotation_per_stratum,
    )


if __name__ == "__main__":
    main()

