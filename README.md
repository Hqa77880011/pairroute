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

The download/conversion was actually checked: 65 tasks, 58 with four owners and seven with three. The release's nominal `train` split remains test-only here. Options and original owner allocation are retained; runtime owner order is permuted by the paired episode seed without changing ownership. `prepare --source manifest --input pack.json --output data/custom` supports source-traceable custom spans with public text, options, owners and a declared split. Any `truth` and `evaluation` fields are removed into the separate label file.

`atomic_spans` splits owner material at sentence punctuation followed by an uppercase letter, or at a newline, preserving exact text and character offsets; decimal points are not boundaries. This conservative sentence-span rule is an implementation convention, not a claim to recover the paper's unreleased annotation decisions. Review context-dependent qualifiers before assigning semantic critical-group labels. A manifest can supply reviewed atomic spans directly.

The generator creates 400 training and 80 validation tasks by default. Training uses negative-conjunction and maximum-utility template families; validation uses positive-eligibility and minimum-cost families, rather than changing only entity names. `semantic_template_family` is recorded and target construction rejects overlapping train/validation families. These newly generated tasks are not claimed to be the exact original experiment corpus. Two-/three-fact, redundant, independent and wrong-minority diagnostics use a separate test pack, for example:

```bash
pairroute make-scenarios --kind pair --count 600 --seed 2028 --output data/two_fact
pairroute make-scenarios --kind triple --count 600 --seed 2029 --output data/three_fact
```

Critical fact/group annotations on official tasks need evaluation-only source review; the released answer labels alone do not supply those groups. Do not expose group labels or truth to deployable routers.

## Training

Targets use a local Hugging Face model; model files download on first use. Keep target/training/model seeds paired and use `--revision` to pin a model version.

```bash
pairroute make-targets --tasks data/scenarios/tasks.json --labels data/scenarios/labels.json --model Qwen/Qwen3-4B --seed 17 --output data/targets.json
pairroute train --targets data/targets.json --config configs/default.json --seed 17 --output checkpoints/ranker
```

Offline targets use mean owner decision loss from canonical option-ID likelihoods, singleton improvements and the discrete pair second difference. They use original training facts; runtime descriptors do not contain the targets. Only train/validation tasks are accepted, and HiddenBench/CollabSim sources are explicitly excluded. Signed unigram/bigram hashing produces two 2,048-dimensional vectors for descriptor and paid history. The 128/64 MLP, symmetric bilinear interaction and six symmetric relation features are trained with unit-weight singleton and interaction Huber losses. Training validates on disjoint scenario IDs; checkpoints retain config/training seed and target teacher metadata. No preprocessing is fitted on test content.

## Inference and evaluation

```bash
pairroute run --tasks data/hiddenbench/tasks.json --checkpoint checkpoints/ranker/ranker.pt --model Qwen/Qwen3-4B --method pairroute --seeds 11 23 37 53 71 --output runs/pairroute
pairroute evaluate --inputs runs/pairroute/traces.json --labels data/hiddenbench/labels.json --output runs/comparison.json
pairroute plot --input runs/comparison.json --output runs/figures
```

`configs/default.json` sets three active candidates per owner, 32 tokens per compact serialized descriptor, initial metadata at most B/4, exploration 0.1, redundancy weight 0.05, and a single B/8 gap exchange. Descriptors are admitted round-robin in seeded owner order. Every envelope and recipient copy is charged. Requests plus the complete owner upload and relay must fit before selecting a group; facts are not truncated. Initial and terminal votes are charged and terminal inference calls reserved before optional communication. Generation is capped at 4,096 tokens and prompt-plus-generation at 65,536 per episode. Invalid nonverbatim uploads retain their inference cost and supply no evidence. Unknown/invalid votes, including JSON booleans, cannot form a majority.

Success is a correct strict majority (>n/2); a missing majority fails. `single_reader` uses one correct answer instead. The evaluator averages paired seeds within original tasks, then tasks. It exposes task and episode counts and per-task success rather than treating 325 episodes as independent tasks. Critical-fact recall is recovered critical facts / critical facts; complete-group delivery is complete critical groups / annotated groups. Failure counts prioritize candidate omission, then missing delivery (zero pair signal / premature stop when independently audited), then integration failure. These labels are evaluation-only. Communication, prompt/generated/inference tokens, serial/router time and per-stage costs/times are reported separately.

For semantic delivery, redundant fraction and premature-stop diagnostics, use the independent, evaluator-only audit. Its teacher calls are outside deployable inference accounting and remain in the audit file:

```bash
pairroute audit-delivery --tasks data/hiddenbench/tasks.json --labels data/hiddenbench/labels.json --traces runs/pairroute/traces.json --model Qwen/Qwen3-8B --output runs/delivery_audit.json
```

Match the tasks/labels to the actual trace cohort and add reviewed `critical_groups` (lists of source IDs) to evaluation labels before auditing delivery. Pass the resulting file as `evaluate --delivery-audit runs/delivery_audit.json`. For discussion baselines, complete team delivery needs semantic auditing; absence of an audit leaves these metrics unavailable. Descriptor receipt by the coordinator alone is not complete team delivery.

## Comparisons and ablations

`run --method` supports `random`, `deduplicate`, `singleton`, `rule_pair`, `dysco`, `free_discussion`, `fixed_dissent`, `exchange_decide`, `reveal_all`, `single_reader`, and the removals `no_interaction`, `no_gap`, `no_exploration`, `no_redundancy`. `no_interaction` preserves the original singleton-only removal (both pair actions and interaction scores removed). `zero_interaction` retains pair actions with their interaction score set to zero. Save each method to a distinct directory and pass all traces to `evaluate --inputs`. The `oracle` diagnostic requires `--oracle-labels` and uses evaluator information.

```bash
pairroute run --tasks data/hiddenbench/tasks.json --checkpoint checkpoints/ranker/ranker.pt --model Qwen/Qwen3-4B --method no_interaction --seeds 11 23 37 53 71 --output runs/no_interaction
pairroute evaluate --inputs runs/pairroute/traces.json runs/no_interaction/traces.json --labels data/hiddenbench/labels.json --output runs/comparison.json
```

For supplemental E01, use `configs/fixed_pool.json` for all four selectors: `singleton`, `zero_interaction`, `rule_pair`, `pairroute`. The configuration disables exploration and gap queries, so the same seeded charged candidate pool is used. All four share learned singleton scores; the rule variant replaces the learned interaction with a unit entity/condition relation bonus (an implementation convention). For example:

```bash
pairroute run --tasks data/hiddenbench/tasks.json --checkpoint checkpoints/ranker/ranker.pt --config configs/fixed_pool.json --method zero_interaction --budgets 1024 --seeds 11 23 37 53 71 --output runs/fixed_zero
```

For E02, repeat `pairroute` and `singleton` with `configs/cap1.json` and `configs/cap6.json`. Only the candidate limit changes; metadata remains B/4 (256 at B=1024), and exploration/gap return to the default settings.

```bash
pairroute run --tasks data/hiddenbench/tasks.json --checkpoint checkpoints/ranker/ranker.pt --config configs/cap6.json --method pairroute --budgets 1024 --seeds 11 23 37 53 71 --output runs/cap6_full
```

Keep delivery audits separate for each candidate setting. Summary/plot labels retain cap and fixed-pool settings, and duplicate task/method/budget/seed/setting observations are rejected. The DySCo entry is a local adaptation with charged summaries, critiques, visible-evidence trust updates and the declared edge-score weights; it is not a packaged upstream reproduction.

An adapter for an externally supplied CollabSim checkout is available:

```bash
pairroute run-collabsim --source-root data/raw/CollabSim --checkpoint checkpoints/ranker/ranker.pt --native-config configs/study_conditions/hidden_profile/baseline.yml --method pairroute --budget 1024 --seeds 0 1 2 --output runs/collabsim
```

It expects the upstream `src.controller`, agent interface and hidden-profile configuration API used by `collabsim.py`. The native individual-vote accuracy is separate from HiddenBench majority success. This external adapter has not been run against a real checkout in this update.

## Outputs

Prepared/generated task directories contain separate `tasks.json` and `labels.json`. Training saves `ranker.pt` with config/seed and `history.json`. Execution saves `traces.json` with routing, delivered evidence, communication and inference ledgers. Evaluation writes the requested summary JSON; plotting writes PNG/SVG figures.

| Paper module or metric | Code entry | Configuration/command |
| --- | --- | --- |
| Locked test data / separate scenario training | `cli.prepare`, `cli.synthetic` | `prepare`, `make-scenarios` |
| Teacher improvements and pair second differences | `cli.collect_targets`, `Owners.loss` | `make-targets --model --revision --seed` |
| Signed features and symmetric value network | `method.Ranker`, `cli.train` | `train`, Huber loss in `configs/default.json` |
| Paid descriptors, group selection, gap and votes | `Owners`, `Ledger`, `cli.route`, `llm.py` | `run --method`, budget/inference settings |
| Frozen-pool and charged candidate-cap controls | `cli.route` | `fixed_pool.json`, `cap1.json`, `cap6.json` |
| Majority success, delivery/failure and token/time metrics | `cli.evaluate`, `audit.py` | `evaluate`, `audit-delivery`, `plot` |
| Native individual-vote transfer | `collabsim.py` | `run-collabsim --source-root` |

## Verification

Rechecked on 2026-11-01 using the existing `python tests/smoke.py` with one CPU thread. Optional TensorFlow/Flax backends were disabled (`USE_TF=0`, `USE_FLAX=0`); local test-only dependencies were isolated from the shared Python environment.

The existing implementation was retained. This update repairs charged candidate admission/gap bounds/vote handling and adds the supplemental controls, disjoint scenario templates, grouped evaluation and functional verification. It does not establish that this repository produced the paper's recorded results.

```bash
python tests/smoke.py
```

Actually passed on Windows, Python 3.13.5, torch 2.7.1+cu118 (CPU), NumPy 2.1.3, SciPy 1.15.3, scikit-learn 1.6.1, transformers 4.57.6 and tokenizers 0.22.1. The check creates random local GPT-2 weights and runs actual forward likelihoods, generation and inference caps. Explicitly scripted owner responses then test routing contracts, not language-model reasoning: teacher second differences averaged over owner views, two optimizer steps with finite gradients and every Ranker parameter updated, checkpoint equivalence and pair symmetry, complete-span feasibility, charged recipient fan-out, invalid uploads, B/8 gap limits, frozen controls, cap-one/six metadata bounds, and strict-majority/recall/group-delivery hand examples. It also runs data conversion and plotting. Small artifacts stay in ignored `outputs/smoke/`.

HiddenBench download/conversion was additionally executed on the real release. Qwen3 target production, real-model owner dialogue/auditing, full training, the benchmark/diagnostic sweeps, and external CollabSim execution remain unverified. Scripted protocol checks and random tiny-model outputs are not benchmark results, and no performance table is filled from them.
