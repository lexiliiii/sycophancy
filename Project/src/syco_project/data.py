from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .config import Config, resolve_path
from .io_utils import read_json


def option_map(item: dict[str, Any]) -> dict[str, str]:
    options: dict[str, str] = {}
    raw = item["options"]
    if isinstance(raw, dict):
        return {str(k).upper(): str(v) for k, v in raw.items()}
    for option in raw:
        letter, value = str(option).split(")", 1)
        options[letter.strip().upper()] = value.strip()
    return options


def load_questions(config: Config) -> list[dict[str, Any]]:
    path = resolve_path(config, config.dataset["questions_path"])
    questions = read_json(path)
    domains = set(config.dataset.get("domains") or [])
    difficulties = set(config.dataset.get("difficulties") or [])
    if domains:
        questions = [q for q in questions if q["domain"] in domains]
    if difficulties:
        questions = [q for q in questions if q["difficulty"] in difficulties]

    sample_size = config.dataset.get("sample_size")
    if not sample_size or int(sample_size) >= len(questions):
        return sorted(questions, key=lambda q: q["id"])

    # Proportional allocation by domain with deterministic remainder handling.
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        by_domain[q["domain"]].append(q)
    rng = random.Random(config.seed)
    target = int(sample_size)
    selected: list[dict[str, Any]] = []
    domains_sorted = sorted(by_domain)
    base = target // len(domains_sorted)
    remainder = target % len(domains_sorted)
    for index, domain in enumerate(domains_sorted):
        n = base + int(index < remainder)
        pool = sorted(by_domain[domain], key=lambda q: q["id"])
        selected.extend(rng.sample(pool, min(n, len(pool))))
    return sorted(selected, key=lambda q: (q["domain"], q["id"]))


def format_question(item: dict[str, Any], response_instruction: str) -> str:
    options = option_map(item)
    lines = [item["question"].strip(), ""]
    for letter in "ABCD":
        lines.append(f"{letter}) {options[letter]}")
    lines.extend(["", response_instruction.strip()])
    return "\n".join(lines)


def deterministic_wrong_letter(correct: str, question_id: str, seed: int) -> str:
    wrong = [letter for letter in "ABCD" if letter != correct]
    rng = random.Random(f"{seed}:{question_id}")
    return rng.choice(wrong)

