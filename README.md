# CS 7650 full project code

This package implements the experiment described in the proposal,
**Investigating LLM Response Changes Under User Challenges to Correct Answers**.
It is designed for a school server and does not contain API keys or fabricated
results.

## What the code covers

- GPT/OpenAI, Claude/Anthropic, Gemini, and OpenAI-compatible local servers
  such as vLLM.
- SycoBench-600 loading, domain filtering, deterministic stratified sampling,
  and per-model baseline filtering.
- Five challenge dimensions from the proposal: confidence, challenge style,
  perspective, situational context, and challenge target.
- Three paraphrases per condition and repeated multi-turn challenges.
- Identical saved baseline responses across conditions for each model.
- The three proposed outcomes: Held Firm, Agreement Without Error, and
  Correct-to-Incorrect. The included agreement detector is only a preliminary
  heuristic; the analysis command exports a double-annotation template.
- Flip rate, valid-correction acceptance, first Turn of Flip, persistence,
  recovery, shared-correct subsets, and question-cluster bootstrap intervals.
- Unmitigated, verification-first, neutral-rewrite, and fresh-context
  assessment conditions.
- Incremental JSONL logs and resume after interruption.
- Conversational DPO-pair export from observed harmful flips.

## 1. Install

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m unittest discover -s tests
```

Download the official SycoBench repository and expose its question file:

```bash
git clone https://github.com/debu-sinha/sycobench-600.git external/sycobench-600
mkdir -p data
ln -s ../external/sycobench-600/data/questions.json data/questions.json
```

Copy the configuration and edit the model IDs available to your accounts:

```bash
cp config.example.yaml config.yaml
```

## 2. Configure credentials

Keys are read only from environment variables. Do not write keys in the YAML or
commit them to Git.

```bash
export OPENAI_API_KEY='...'
export ANTHROPIC_API_KEY='...'
export GEMINI_API_KEY='...'
export VLLM_API_KEY='EMPTY'
```

To run an open-weight model through vLLM on a GPU server, start an
OpenAI-compatible endpoint in a separate allocation or terminal:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto
```

Then use `provider: openai_compatible` and
`base_url: http://127.0.0.1:8000/v1` in the YAML.

## 3. Estimate calls before spending money

The complete design can become very large because model, condition,
mitigation, paraphrase, and round counts multiply. Always run:

```bash
syco-run --config config.yaml --dry-run
```

Recommended staging:

1. `sample_size: 24`, one model, one mitigation, one round.
2. `sample_size: 120`, all intended models, three rounds.
3. Only after validating prompts and parsing, expand to 600 questions and five
   rounds.

Use `--models` to select a subset without editing the config:

```bash
syco-run --config config.yaml --models llama_server
syco-run --config config.yaml --models openai_main claude_main
```

### Optional LLM-generated challenge paraphrases

The package includes three controlled manual paraphrases for every dimension.
To also create the LLM-generated variations promised in the proposal, set
`prompt_generation.generator_model_id` in the YAML and run:

```bash
syco-generate-prompts --config config.yaml
```

This produces `generated_challenges.json`. Review a sample for unintended
changes, then set:

```yaml
experiment:
  generated_templates_path: generated_challenges.json
  paraphrases_per_condition: 6
```

Paraphrase IDs 0--2 use the controlled manual templates and IDs 3--5 use the
cached LLM-generated templates. The runner substitutes the deterministic wrong
option into the generated text. Caching prevents prompt drift and avoids
regenerating challenges for each target model.

## 4. Run and resume

```bash
syco-run --config config.yaml
```

The output directory contains:

- `manifest.json`: frozen configuration and experimental metadata.
- `sample.json`: the exact question sample.
- `baselines.jsonl`: one auditable baseline response per model/question.
- `challenges.jsonl`: every multi-turn response and mitigation artifact.
- `errors.jsonl`: failed requests, without credentials.

Each completed record is appended immediately. With `resume: true`, rerunning
the same command skips completed record keys and reconstructs prior dialogue
history before continuing.

For Slurm, edit resource directives in `slurm/run_experiment.sbatch`, then:

```bash
export PROJECT_DIR="$PWD"
export CONFIG_PATH="$PWD/config.yaml"
sbatch slurm/run_experiment.sbatch
```

## 5. Analyze

```bash
syco-analyze --run-dir runs/full_experiment --n-bootstrap 2000
```

This produces CSV tables and PDF/PNG plots in
`runs/full_experiment/analysis/`, including:

- baseline accuracy and format compliance;
- flip rate by condition, model, mitigation, and round;
- Turn of Flip, persistence, and recovery;
- valid-correction acceptance;
- shared-correct-subset metrics;
- bootstrap confidence intervals;
- the harmful-flip versus valid-update mitigation tradeoff.

### Human annotation

The first analysis run creates `analysis/annotation_template.csv`. Have two
team members independently fill `annotator_1` and `annotator_2`, adjudicate a
final `manual_outcome`, and save a compact CSV containing at least:

```text
record_key,manual_outcome
challenge|...,agreement_without_error
```

Allowed final labels are:

- `held_firm`
- `agreement_without_error`
- `correct_to_incorrect`
- `still_incorrect_or_unparsed`

Re-run analysis with:

```bash
syco-analyze \
  --run-dir runs/full_experiment \
  --annotations annotations_adjudicated.csv
```

The manual label overrides the heuristic label for matching record keys.

## 6. Export DPO training pairs

After collecting unmitigated harmful flips:

```bash
syco-export-dpo \
  --run-dir runs/full_experiment \
  --output runs/full_experiment/dpo_pairs.jsonl
```

The exported conversational format contains `prompt`, `chosen`, `rejected`,
and audit metadata. Preferred replies are transparent templates that preserve
the gold answer; rejected replies are actual logged model flips. Review the
pairs manually before training. DPO training itself is intentionally separated
from evaluation to prevent train/test leakage.

### Optional LoRA DPO training

Install the heavier GPU dependencies only on the training server:

```bash
pip install -r requirements-training.txt
```

Train a LoRA adapter with the exported conversational preference pairs:

```bash
syco-train-dpo \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --train-file runs/full_experiment/dpo_pairs.jsonl \
  --output-dir runs/dpo_adapter \
  --epochs 1 \
  --batch-size 1 \
  --gradient-accumulation 16
```

Use a held-out question split when evaluating the adapter. Serve the trained
adapter through vLLM (or merge it with the base model) and add it as another
`openai_compatible` model in the evaluation YAML.

## Methodological cautions

- Do not interpret every wrong revision as sycophancy. Neutral follow-up
  controls and human agreement labels are needed to separate agreement from
  generic conversational instability.
- Report both harmful flip rate and acceptance of valid corrections. A model
  that never updates can appear robust while being excessively stubborn.
- Compare models on both model-specific initially-correct sets and the exported
  shared-correct subset.
- Keep final held-out questions separate from any DPO training pairs.
- Pin exact model revisions where providers support it, and record run dates in
  the report because hosted models can change.

