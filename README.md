# PairRoute

Implementation of **PairRoute: Budgeted Evidence Routing with Learned Pairwise Decision Value**.

PairRoute scores paid fact descriptors using public history, singleton decision value and symmetric pair interactions. It chooses affordable disclosures, verifies owner replies against their source spans, and meters every request, upload, relay and terminal vote.

## Installation and data

Use Python 3.11 and install PyTorch for your hardware, then install this package in an activated virtual environment:

```bash
python -m pip install -e .
pairroute prepare --source hiddenbench --output data/hiddenbench
```

The importer downloads [HiddenBench](https://huggingface.co/datasets/YuxuanLi1225/HiddenBench) or reads a downloaded `benchmark.json` with `--input`. It preserves public facts, options and one owner per original private bundle. All released tasks, including the upstream `train` split, are assigned to the locked test cohort. Ground-truth options are written to a separate `labels.json` and never supplied to deployable routing.

Custom task packs use `prepare --source manifest --input tasks.json --output data/custom`. Each task has `id`, `public`, `options`, `owners` (one list of `{id,text}` source spans per owner), `split` and `source`. Optional `truth` and `evaluation` fields are removed into the sealed label file. Critical-group diagnostics use `critical_groups` as lists of source IDs. Keep a span's original `source_id` stable when changing owner allocation.

## Offline targets and ranker training

Generate independent constraint scenarios rather than using public benchmark answers for training:

```bash
pairroute make-scenarios --count 480 --output data/scenarios
pairroute make-targets --tasks data/scenarios/tasks.json --labels data/scenarios/labels.json --model Qwen/Qwen3-4B --output data/targets.json
pairroute train --targets data/targets.json --output checkpoints/ranker
```

Scenario types include complementary pairs, three-condition interactions, redundant facts, independent useful facts and an unverified minority claim. `make-targets` accepts only training/development scenarios and rejects HiddenBench/CollabSim. The frozen reader computes normalized likelihoods of canonical option identifiers for each owner's local view. Each public history has a common baseline loss; singleton utility is its reduction after one disclosure, and pair interaction subtracts both singleton reductions from the joint reduction. Negative targets remain signed. Two histories and up to eight cross-owner pairs are sampled per scenario.

The ranker hashes paid descriptor and public-history unigrams/bigrams into 2,048 signed buckets each. A 4,096 → 128 → 64 projection supplies singleton scores. A symmetric bilinear matrix and six symmetric relation features supply pair interactions. `configs/default.json` specifies 30 AdamW epochs, learning rate 0.001, weight decay 0.01, batch size 64 and Huber loss. `ranker.pt` is selected by held-out Huber error; `history.json` records the training curve. This checkpoint criterion is an implementation convention differing from terminal decision-utility selection.

## Communication-budget evaluation

```bash
pairroute run --tasks data/hiddenbench/tasks.json --checkpoint checkpoints/ranker/ranker.pt --model Qwen/Qwen3-4B --method pairroute --output runs/pairroute
pairroute evaluate --inputs runs/pairroute/traces.json --labels data/hiddenbench/labels.json --output runs/comparison.json
pairroute plot --input runs/comparison.json --output runs/figures
```

Default delivered-token caps are 512, 1,024, 2,048 and 4,096 with paired seeds 11, 23, 37, 53 and 71. `--budgets` and `--seeds` override those grids. Repeat with `--model Qwen/Qwen3-8B`; `--revision` pins a model revision. Qwen3 runs with thinking disabled and temperature 0.2. No paid model API is used.

An informative descriptor is paid owner-to-coordinator disclosure. At most three descriptors per owner and 32 serialized tokens per descriptor are admitted under a B/4 allowance. Selection uses only those descriptors and already disclosed history. A group reserves its full request/upload/relay cost before an owner call. Verbatim replies are checked locally; invalid replies consume generation tokens without evidence credit. The unused B/8 gap allowance returns to selection after one paid inquiry. Final vote delivery is reserved first.

`traces.json` contains every serialized message, recipient set, charged phase, fact/source ID, owner vote, candidate rejection and inference call. Communication is tokenizer-length × recipient-count; relay copies are paid without regenerating the payload. Prompt and generated tokens are reported separately under 65,536 total and 4,096 generation caps. Exhausted inference produces an explicit episode failure. Router arithmetic uses no language-model tokens.

Metrics include strict-majority success, individual vote accuracy, critical-fact recall, complete-group delivery, communication by phase, prompt/generated totals and serial time. Paired seeds are grouped by original task. No majority fails. Critical labels are evaluation-only. An incomplete critical group and an incorrect decision after sufficient delivery are separate failure categories.

## Baselines, ablations and transfer

`run --method` accepts random routing, `deduplicate`, `singleton`, `rule_pair`, `dysco`, `free_discussion`, `fixed_dissent`, `exchange_decide`, `reveal_all` and `single_reader`. All use the same owner interface and token ledger. Direct broadcast charges n−1 copies; hub routing charges one upload plus n−1 relays. Single-reader success uses its own terminal answer. `oracle --oracle-labels data/hiddenbench/labels.json` is a privileged truth-aware reference.

Remove components with `no_interaction`, `no_gap`, `no_exploration` or `no_redundancy`, reusing the trained ranker. Controlled test scenarios can be generated separately:

```bash
pairroute make-scenarios --kind pair --count 120 --seed 9001 --output data/pair_diagnostic
pairroute run --tasks data/pair_diagnostic/tasks.json --checkpoint checkpoints/ranker/ranker.pt --method pairroute --budgets 1024 --output runs/pair_diagnostic
pairroute evaluate --inputs runs/pair_diagnostic/traces.json --labels data/pair_diagnostic/labels.json --output runs/pair_diagnostic/metrics.json
```

The CollabSim adapter below runs its native initial-vote/discussion/final-vote controller. The standalone `run` command remains the delivered-token protocol for HiddenBench and controlled task packs.

The [DySCo](https://arxiv.org/abs/2606.01828) adaptation retains paid answer/reason/confidence states, trust/confidence/divergence/dependency/cost edge terms, TopK neighbors, compressed critiques, trust updates and entropy stopping. Its receiver reads only local evidence and actually delivered messages. Both the native trust-weighted answer and the common strict-majority outcome are stored. Direct-discussion and full-disclosure controls retain their own message interfaces and do not pay for unused candidate descriptors.

Training scenarios are generated constraint tasks, and PairRoute selection is greedy with second-order interactions. These conventions should accompany comparisons made with this repository.

## Native CollabSim and mechanism metrics

```bash
git clone https://github.com/neuhai/CollabSim data/raw/CollabSim
git -C data/raw/CollabSim checkout 08ed0ed1cefb2edf1ea389b20ba8355815a7c8d0
pairroute run-collabsim --source-root data/raw/CollabSim --checkpoint checkpoints/ranker/ranker.pt --model Qwen/Qwen3-4B --method pairroute --budget 1024 --output runs/collabsim
pairroute evaluate --inputs runs/collabsim/traces.json --labels runs/collabsim/labels.json --output runs/collabsim/metrics.json
```

The adapter uses CollabSim's 30-step baseline phase schedule and 25 seeds. Initial votes, delivered messages, final votes and probe events remain in the native event log. Routing is computed at the first discussion opportunity; only messages actually delivered by the native controller enter final decision prompts. Initial votes and all local probe calls count toward the communication/inference ledgers. Native probes remain outside agent memory.

The static `oracle` evaluates all affordable subsets of the fixed advertised pool with sealed answer labels. Its target-construction ledger is separate from deployable inference. It is an oracle over that pool, not over undisclosed candidates or future gap responses.

To audit semantic fact delivery, including paraphrases from discussion baselines:

```bash
pairroute audit-delivery --tasks data/pair_diagnostic/tasks.json --labels data/pair_diagnostic/labels.json --traces runs/pair_diagnostic/traces.json --model Qwen/Qwen3-8B --output runs/pair_diagnostic/delivery.json
pairroute evaluate --inputs runs/pair_diagnostic/traces.json --labels data/pair_diagnostic/labels.json --delivery-audit runs/pair_diagnostic/delivery.json --output runs/pair_diagnostic/metrics.json
```

The evaluator requires the full critical group to be known by each recipient, including its own original facts. Audits also compute remaining positive affordable utility, zero pair signal and repeated delivery tokens. Failure classification prioritizes candidate omission, absent pair signal, premature stopping, then decision integration. The audit model and its calls are saved separately from routing cost.

Generate 120 scenarios for each of `pair`, `triple`, `redundant`, `independent` and `wrong_minority` with separate output directories. Compare all methods at the same budgets and seeds. `no_interaction`, `no_gap`, `no_exploration` and `no_redundancy` use the full ranker checkpoint. Exchange–Decide uses two exchanges with one or two facts and a reason against the front-runner. `evaluate` reports success, individual-vote accuracy, critical-fact recall, complete-group delivery, failure categories and every communication phase. Plots are saved as PNG and SVG.
