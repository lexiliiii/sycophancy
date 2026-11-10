from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, Mitigation, ModelSpec, load_config, resolve_path
from .data import (
    deterministic_wrong_letter,
    format_question,
    load_questions,
)
from .io_utils import append_jsonl, atomic_write_json, read_json, read_jsonl
from .parsing import expresses_agreement, outcome_label, parse_final_letter
from .prompts import (
    build_challenge,
    build_correct_correction,
    fresh_assessment_request,
    mitigation_instruction,
    neutral_rewrite_request,
)
from .providers import GenerationResult, Provider, create_provider


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def baseline_key(model_id: str, question_id: str) -> str:
    return f"baseline|{model_id}|{question_id}"


def chain_key(
    model_id: str,
    question_id: str,
    condition_id: str,
    mitigation_id: str,
    paraphrase_id: int,
    round_index: int,
) -> str:
    return "|".join(
        map(
            str,
            [
                "challenge",
                model_id,
                question_id,
                condition_id,
                mitigation_id,
                paraphrase_id,
                round_index,
            ],
        )
    )


class ExperimentRunner:
    def __init__(self, config: Config, selected_models: set[str] | None = None):
        self.config = config
        exp = config.experiment
        self.questions = load_questions(config)
        self.models = [
            spec for spec in config.models if not selected_models or spec.id in selected_models
        ]
        if not self.models:
            raise ValueError("No models selected")
        self.output_dir = resolve_path(config, exp["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_path = self.output_dir / "baselines.jsonl"
        self.challenge_path = self.output_dir / "challenges.jsonl"
        self.error_path = self.output_dir / "errors.jsonl"
        self.temperature = exp.get("temperature", 0.0)
        self.max_tokens = int(exp.get("max_tokens", 256))
        self.rounds = int(exp.get("rounds", 5))
        self.n_paraphrases = int(exp.get("paraphrases_per_condition", 3))
        self.response_instruction = exp["response_instruction"]
        self.timeout = float(exp.get("request_timeout_seconds", 120))
        self.concurrency = int(exp.get("concurrency_per_model", 8))
        self.resume = bool(exp.get("resume", True))
        generated_path = exp.get("generated_templates_path")
        self.generated_templates = (
            read_json(resolve_path(config, generated_path)) if generated_path else {}
        )
        self.log_lock = asyncio.Lock()
        self.baselines_by_key = {
            record["record_key"]: record for record in read_jsonl(self.baseline_path)
        }
        self.challenges_by_key = {
            record["record_key"]: record for record in read_jsonl(self.challenge_path)
        }
        self.completed_challenge_keys = set(self.challenges_by_key)

    async def append_record(self, path: Path, record: dict[str, Any]) -> None:
        async with self.log_lock:
            append_jsonl(path, [record])

    async def call(
        self,
        provider: Provider,
        semaphore: asyncio.Semaphore,
        messages: list[dict[str, str]],
    ) -> GenerationResult:
        async with semaphore:
            return await provider.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

    async def run_baseline(
        self,
        spec: ModelSpec,
        provider: Provider,
        semaphore: asyncio.Semaphore,
        question: dict[str, Any],
    ) -> dict[str, Any]:
        key = baseline_key(spec.id, question["id"])
        if self.resume and key in self.baselines_by_key:
            return self.baselines_by_key[key]
        prompt = format_question(question, self.response_instruction)
        messages = [{"role": "user", "content": prompt}]
        try:
            result = await self.call(provider, semaphore, messages)
            parsed = parse_final_letter(result.text)
            record = {
                "kind": "baseline",
                "record_key": key,
                "created_at": utc_now(),
                "model_id": spec.id,
                "provider": result.provider,
                "requested_model": spec.model,
                "returned_model": result.model,
                "question_id": question["id"],
                "domain": question["domain"],
                "difficulty": question["difficulty"],
                "correct_letter": question["correct"],
                "question_prompt": prompt,
                "messages": messages,
                "raw_response": result.text,
                "parsed_letter": parsed,
                "correct": parsed == question["correct"],
                "format_valid": parsed is not None,
                "usage": result.usage,
                "response_id": result.response_id,
            }
            await self.append_record(self.baseline_path, record)
            self.baselines_by_key[key] = record
            print(
                f"baseline {spec.id} {question['id']} parsed={parsed} "
                f"gold={question['correct']}",
                flush=True,
            )
            return record
        except Exception as exc:
            error = {
                "kind": "baseline_error",
                "record_key": key,
                "created_at": utc_now(),
                "model_id": spec.id,
                "question_id": question["id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            await self.append_record(self.error_path, error)
            raise

    async def prepare_turn(
        self,
        provider: Provider,
        semaphore: asyncio.Semaphore,
        mitigation: Mitigation,
        question_prompt: str,
        baseline_response: str,
        challenge: str,
    ) -> tuple[str, str | None, str | None]:
        """Return challenge sent to the dialogue plus optional rewrite/assessment."""
        if mitigation.type == "neutral_rewrite":
            rewrite = await self.call(
                provider, semaphore, neutral_rewrite_request(challenge)
            )
            return rewrite.text.strip(), rewrite.text.strip(), None
        if mitigation.type == "neutral_rewrite_fresh_assessment":
            rewrite = await self.call(
                provider, semaphore, neutral_rewrite_request(challenge)
            )
            neutral = rewrite.text.strip()
            assessment = await self.call(
                provider,
                semaphore,
                fresh_assessment_request(
                    question_prompt, baseline_response, neutral
                ),
            )
            return neutral, neutral, assessment.text.strip()
        if mitigation.type == "fresh_assessment":
            assessment = await self.call(
                provider,
                semaphore,
                fresh_assessment_request(
                    question_prompt, baseline_response, challenge
                ),
            )
            return challenge, None, assessment.text.strip()
        return challenge, None, None

    async def run_chain(
        self,
        spec: ModelSpec,
        provider: Provider,
        semaphore: asyncio.Semaphore,
        question: dict[str, Any],
        baseline: dict[str, Any],
        condition_id: str,
        mitigation: Mitigation,
        paraphrase_id: int,
    ) -> None:
        condition = next(
            (item for item in self.config.conditions if item.id == condition_id), None
        )
        is_valid_correction = condition_id == "valid_correction"
        conversation = [
            {"role": "user", "content": baseline["question_prompt"]},
            {"role": "assistant", "content": baseline["raw_response"]},
        ]
        if mitigation.type == "system_instruction" and mitigation.instruction:
            conversation.insert(
                0, {"role": "developer", "content": mitigation.instruction}
            )

        for round_index in range(1, self.rounds + 1):
            key = chain_key(
                spec.id,
                question["id"],
                condition_id,
                mitigation.id,
                paraphrase_id,
                round_index,
            )
            if self.resume and key in self.completed_challenge_keys:
                # Recover the existing assistant reply so later rounds retain history.
                existing = self.challenges_by_key[key]
                conversation.append(
                    {"role": "user", "content": existing["challenge_sent"]}
                )
                conversation.append(
                    {"role": "assistant", "content": existing["raw_response"]}
                )
                continue

            if is_valid_correction:
                original_challenge = build_correct_correction(
                    question["correct"], paraphrase_id, round_index
                )
            else:
                assert condition is not None
                wrong = deterministic_wrong_letter(
                    question["correct"], question["id"], self.config.seed
                )
                original_challenge = build_challenge(
                    condition, paraphrase_id, round_index, wrong
                )
                templates = self.generated_templates.get(condition.id, [])
                # IDs 0..2 remain the controlled manual paraphrases. IDs 3+
                # use cached LLM-generated variants, so a six-variant run uses both.
                if templates and paraphrase_id >= 3:
                    generated_index = (paraphrase_id - 3) % len(templates)
                    original_challenge = templates[generated_index].replace(
                        "<WRONG_LETTER>", wrong
                    )
                    if round_index > 1:
                        original_challenge += (
                            f" This is challenge round {round_index}; reconsider the original "
                            "problem again."
                        )

            try:
                challenge_sent, neutral_rewrite, assessment = await self.prepare_turn(
                    provider,
                    semaphore,
                    mitigation,
                    baseline["question_prompt"],
                    baseline["raw_response"],
                    original_challenge,
                )
                response_instruction = self.config.experiment.get(
                    "response_instruction", ""
                ).strip()
                if response_instruction:
                    challenge_sent = (
                        f"{challenge_sent}\n\n{response_instruction}"
                    )
                turn_messages = deepcopy(conversation)
                extra_instruction = mitigation_instruction(
                    mitigation.type, assessment=assessment
                )
                if extra_instruction:
                    turn_messages.insert(
                        0, {"role": "developer", "content": extra_instruction}
                    )
                turn_messages.append({"role": "user", "content": challenge_sent})
                result = await self.call(provider, semaphore, turn_messages)
                parsed = parse_final_letter(result.text)
                correct_after = parsed == question["correct"]
                record = {
                    "kind": "challenge",
                    "record_key": key,
                    "created_at": utc_now(),
                    "model_id": spec.id,
                    "provider": result.provider,
                    "requested_model": spec.model,
                    "returned_model": result.model,
                    "question_id": question["id"],
                    "domain": question["domain"],
                    "difficulty": question["difficulty"],
                    "correct_letter": question["correct"],
                    "baseline_key": baseline["record_key"],
                    "baseline_letter": baseline["parsed_letter"],
                    "baseline_correct": baseline["correct"],
                    "condition_id": condition_id,
                    "condition": condition.__dict__ if condition else None,
                    "mitigation_id": mitigation.id,
                    "mitigation_type": mitigation.type,
                    "paraphrase_id": paraphrase_id,
                    "round": round_index,
                    "original_challenge": original_challenge,
                    "challenge_sent": challenge_sent,
                    "neutral_rewrite": neutral_rewrite,
                    "fresh_assessment": assessment,
                    "messages": turn_messages,
                    "raw_response": result.text,
                    "parsed_letter": parsed,
                    "correct_after": correct_after,
                    "format_valid": parsed is not None,
                    "changed_from_baseline": (
                        parsed is not None
                        and parsed != baseline["parsed_letter"]
                    ),
                    "expresses_agreement": expresses_agreement(result.text),
                    "outcome": outcome_label(
                        baseline["correct"],
                        parsed,
                        question["correct"],
                        result.text,
                    ),
                    "usage": result.usage,
                    "response_id": result.response_id,
                }
                await self.append_record(self.challenge_path, record)
                self.completed_challenge_keys.add(key)
                self.challenges_by_key[key] = record
                conversation.append({"role": "user", "content": challenge_sent})
                conversation.append({"role": "assistant", "content": result.text})
            except Exception as exc:
                await self.append_record(
                    self.error_path,
                    {
                        "kind": "challenge_error",
                        "record_key": key,
                        "created_at": utc_now(),
                        "model_id": spec.id,
                        "question_id": question["id"],
                        "condition_id": condition_id,
                        "mitigation_id": mitigation.id,
                        "paraphrase_id": paraphrase_id,
                        "round": round_index,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                return

    async def run_model(self, spec: ModelSpec) -> None:
        provider = create_provider(spec, self.timeout)
        semaphore = asyncio.Semaphore(self.concurrency)
        baselines = await asyncio.gather(
            *[
                self.run_baseline(spec, provider, semaphore, question)
                for question in self.questions
            ]
        )
        tasks = []
        by_id = {question["id"]: question for question in self.questions}
        for baseline in baselines:
            question = by_id[baseline["question_id"]]
            condition_ids = (
                [condition.id for condition in self.config.conditions]
                if baseline["correct"]
                else ["valid_correction"]
            )
            for condition_id in condition_ids:
                for mitigation in self.config.mitigations:
                    paraphrase_count = (
                        min(self.n_paraphrases, 3)
                        if condition_id == "valid_correction"
                        else self.n_paraphrases
                    )
                    for paraphrase_id in range(paraphrase_count):
                        tasks.append(
                            self.run_chain(
                                spec,
                                provider,
                                semaphore,
                                question,
                                baseline,
                                condition_id,
                                mitigation,
                                paraphrase_id,
                            )
                        )
        await asyncio.gather(*tasks)

    async def run(self) -> None:
        manifest = {
            "created_at": utc_now(),
            "config_path": str(self.config.path),
            "seed": self.config.seed,
            "question_count": len(self.questions),
            "model_ids": [model.id for model in self.models],
            "condition_ids": [condition.id for condition in self.config.conditions],
            "mitigation_ids": [m.id for m in self.config.mitigations],
            "rounds": self.rounds,
            "paraphrases_per_condition": self.n_paraphrases,
            "config": self.config.raw,
        }
        atomic_write_json(self.output_dir / "manifest.json", manifest)
        atomic_write_json(self.output_dir / "sample.json", self.questions)
        for model in self.models:
            print(f"=== Running {model.id} ({model.provider}:{model.model}) ===")
            await self.run_model(model)


def estimate_calls(config: Config, question_count: int) -> dict[str, int]:
    exp = config.experiment
    rounds = int(exp.get("rounds", 5))
    paraphrases = int(exp.get("paraphrases_per_condition", 3))
    mitigations = config.mitigations
    auxiliary_per_round = sum(
        (
            2
            if mitigation.type == "neutral_rewrite_fresh_assessment"
            else 1
            if mitigation.type in {"neutral_rewrite", "fresh_assessment"}
            else 0
        )
        for mitigation in mitigations
    )
    # Worst case assumes every baseline is correct and all harmful conditions run.
    main_challenge_calls = (
        question_count
        * len(config.conditions)
        * len(mitigations)
        * paraphrases
        * rounds
    )
    auxiliary_calls = (
        question_count
        * len(config.conditions)
        * paraphrases
        * rounds
        * auxiliary_per_round
    )
    return {
        "baseline_calls": question_count * len(config.models),
        "max_main_challenge_calls_per_model": main_challenge_calls,
        "max_auxiliary_calls_per_model": auxiliary_calls,
        "max_total_calls_all_models": question_count * len(config.models)
        + len(config.models) * (main_challenge_calls + auxiliary_calls),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full CS 7650 experiment")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--models", nargs="*", help="Optional subset of model ids")
    parser.add_argument("--dry-run", action="store_true", help="Validate and estimate calls")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.dry_run:
        questions = load_questions(config)
        print(json.dumps(estimate_calls(config, len(questions)), indent=2))
        return
    try:
        asyncio.run(ExperimentRunner(config, set(args.models or []) or None).run())
    except KeyboardInterrupt:
        print("Interrupted. Completed JSONL records are safe; rerun to resume.", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()

