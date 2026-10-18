# PairRoute

Implementation of **PairRoute: Budgeted Evidence Routing with Learned Pairwise Decision Value**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

Preparation downloads [HiddenBench](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench). Released tasks form a locked test cohort; truth is separated into `labels.json`. Training uses separately generated scenarios with seed 2027.

```bash
pairroute prepare --source hiddenbench --output data/hiddenbench
pairroute make-scenarios --count 480 --seed 2027 --output data/scenarios
```

## Training

Targets use a local Hugging Face model; model files download on first use. Keep target/training/model seeds paired and use `--revision` to pin a model version.

```bash
pairroute make-targets --tasks data/scenarios/tasks.json --labels data/scenarios/labels.json --model Qwen/Qwen3-4B --seed 17 --output data/targets.json
pairroute train --targets data/targets.json --config configs/default.json --seed 17 --output checkpoints/ranker
```

## Inference and evaluation

```bash
pairroute run --tasks data/hiddenbench/tasks.json --checkpoint checkpoints/ranker/ranker.pt --model Qwen/Qwen3-4B --method pairroute --seeds 11 23 37 53 71 --output runs/pairroute
pairroute evaluate --inputs runs/pairroute/traces.json --labels data/hiddenbench/labels.json --output runs/comparison.json
pairroute plot --input runs/comparison.json --output runs/figures
```

## Comparisons and ablations

`run --method` supports `random`, `deduplicate`, `singleton`, `rule_pair`, `dysco`, `free_discussion`, `fixed_dissent`, `exchange_decide`, `reveal_all`, `single_reader`, and the removals `no_interaction`, `no_gap`, `no_exploration`, `no_redundancy`. Save each method to a distinct directory and pass all traces to `evaluate --inputs`. The `oracle` diagnostic requires `--oracle-labels` and uses evaluator information.

```bash
pairroute run --tasks data/hiddenbench/tasks.json --checkpoint checkpoints/ranker/ranker.pt --model Qwen/Qwen3-4B --method no_interaction --seeds 11 23 37 53 71 --output runs/no_interaction
pairroute evaluate --inputs runs/pairroute/traces.json runs/no_interaction/traces.json --labels data/hiddenbench/labels.json --output runs/comparison.json
```

## Outputs

Prepared/generated task directories contain separate `tasks.json` and `labels.json`. Training saves `ranker.pt` with config/seed and `history.json`. Execution saves `traces.json` with routing, delivered evidence, communication and inference ledgers. Evaluation writes the requested summary JSON; plotting writes PNG/SVG figures.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0, PyYAML 6.0.2. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
