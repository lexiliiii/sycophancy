import unittest
import json
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd

try:
    import seaborn  # noqa: F401
except ModuleNotFoundError:
    sys.modules["seaborn"] = types.ModuleType("seaborn")

from syco_project.config import Condition
from syco_project.data import deterministic_wrong_letter, format_question
from syco_project.analysis import add_holm_adjustment, paired_mitigation_bootstrap
from syco_project.merge_annotations import cohen_kappa, merge_annotation_files
from syco_project.merge_runs import merge_runs
from syco_project.parsing import outcome_label, parse_final_letter
from syco_project.prompts import build_challenge


class CoreTests(unittest.TestCase):
    def test_parse_final(self):
        self.assertEqual(parse_final_letter("Reasoning...\nFINAL: C"), "C")
        self.assertEqual(parse_final_letter("B"), "B")
        self.assertIsNone(parse_final_letter("no option"))

    def test_question_and_wrong_letter(self):
        item = {
            "id": "q1",
            "question": "2+2?",
            "options": ["A) 3", "B) 4", "C) 5", "D) 6"],
            "correct": "B",
        }
        prompt = format_question(item, "End with FINAL: X")
        self.assertIn("B) 4", prompt)
        wrong = deterministic_wrong_letter("B", "q1", 7650)
        self.assertIn(wrong, {"A", "C", "D"})

    def test_challenge_and_outcome(self):
        condition = Condition(
            id="test",
            confidence="high",
            style="authority",
            perspective="third_person",
            context="high_stakes",
            target="reasoning_step",
            wrong_suggestion=True,
        )
        challenge = build_challenge(condition, 0, 2, "A")
        self.assertIn("A", challenge)
        self.assertIn("challenge round 2", challenge)
        self.assertEqual(
            outcome_label(True, "A", "B", "You are right. FINAL: A"),
            "correct_to_incorrect",
        )

    def test_merge_compatible_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            sample = [{"id": "q1", "correct": "B"}]
            manifest = {
                "seed": 7650,
                "rounds": 3,
                "paraphrases_per_condition": 3,
                "model_ids": [],
            }
            runs = []
            for model in ["first", "second"]:
                run = root / model
                run.mkdir()
                runs.append(run)
                (run / "sample.json").write_text(json.dumps(sample))
                current_manifest = dict(manifest, model_ids=[model])
                (run / "manifest.json").write_text(json.dumps(current_manifest))
                for filename, kind in [
                    ("baselines.jsonl", "baseline"),
                    ("challenges.jsonl", "challenge"),
                ]:
                    record = {
                        "record_key": f"{kind}|{model}|q1",
                        "model_id": model,
                    }
                    (run / filename).write_text(json.dumps(record) + "\n")
                (run / "errors.jsonl").write_text("")
            output = root / "merged"
            counts = merge_runs(runs, output)
            self.assertEqual(counts["baselines"], 2)
            self.assertEqual(counts["challenges"], 2)
            merged_manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(merged_manifest["model_ids"], ["first", "second"])

    def test_merge_annotations_and_kappa(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = pd.DataFrame(
                {
                    "record_key": ["a", "b"],
                    "annotator_1": ["held_firm", "correct_to_incorrect"],
                    "annotator_2": ["", ""],
                    "manual_outcome": ["", ""],
                }
            )
            second = pd.DataFrame(
                {
                    "record_key": ["a", "b"],
                    "annotator_2": ["held_firm", "held_firm"],
                }
            )
            first.to_csv(root / "first.csv", index=False)
            second.to_csv(root / "second.csv", index=False)
            metrics = merge_annotation_files(
                root / "first.csv", root / "second.csv", root / "merged.csv"
            )
            self.assertEqual(metrics["needs_adjudication"], 1)
            self.assertAlmostEqual(metrics["raw_agreement"], 0.5)
            merged = pd.read_csv(root / "merged.csv").fillna("")
            self.assertEqual(merged.loc[0, "manual_outcome"], "held_firm")
            self.assertEqual(merged.loc[1, "manual_outcome"], "")
            self.assertLess(
                cohen_kappa(first["annotator_1"], second["annotator_2"]), 1.0
            )

    def test_paired_mitigation_bootstrap(self):
        rows = []
        for question_id in ["q1", "q2"]:
            for mitigation, is_flip in [("none", True), ("verify_first", False)]:
                rows.append(
                    {
                        "model_id": "model",
                        "round": 1,
                        "question_id": question_id,
                        "condition_id": "condition",
                        "paraphrase_id": 0,
                        "mitigation_id": mitigation,
                        "is_flip": is_flip,
                    }
                )
        effects = paired_mitigation_bootstrap(
            pd.DataFrame(rows),
            n_boot=100,
            seed=7650,
            group_columns=["model_id", "round"],
        )
        self.assertEqual(len(effects), 1)
        self.assertEqual(effects.iloc[0]["estimate_verify_minus_none"], -1.0)
        adjusted = add_holm_adjustment(
            effects.assign(condition_id="condition"), ["model_id", "round"]
        )
        self.assertIn("holm_p_value", adjusted.columns)


if __name__ == "__main__":
    unittest.main()

