import argparse
import json
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .method import (
    METHODS,
    Ledger,
    Owners,
    Ranker,
    affordable_groups,
    compact,
    group_value,
    overlap,
    payload,
    relations,
    vector,
)
from .utils import load, plot_results, save, seed_all


def atomic_spans(text, owner):
    """Retain verbatim sentence/line spans with original character offsets."""
    boundaries = [
        0,
        *[m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text)],
        len(text),
    ]
    facts = []
    for left, right in zip(boundaries, boundaries[1:]):
        segment = text[left:right]
        start = left + len(segment) - len(segment.lstrip())
        end = right - len(segment) + len(segment.rstrip())
        if start < end:
            identifier = str(len(facts))
            facts.append(
                {
                    "id": identifier,
                    "text": text[start:end],
                    "source_id": f"{owner}:{identifier}",
                    "source_start": start,
                    "source_end": end,
                }
            )
    return facts


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.source == "hiddenbench":
        if args.input:
            rows = load(args.input)
        else:
            url = "https://huggingface.co/datasets/YuxuanLi1225/HiddenBench/resolve/main/benchmark.json"
            with urllib.request.urlopen(url, timeout=60) as response:
                rows = json.load(response)
        tasks, labels = [], []
        for row in rows:
            if row["correct_answer"] not in row["possible_answers"]:
                raise ValueError("Correct answer outside option set")
            tasks.append(
                {
                    "id": str(row["id"]),
                    "public": row["description"]
                    + "\n"
                    + "\n".join(row["shared_information"]),
                    "options": row["possible_answers"],
                    "owners": [
                        atomic_spans(fact, owner)
                        for owner, fact in enumerate(row["hidden_information"])
                    ],
                    "split": "test",
                    "source": "hiddenbench",
                }
            )
            labels.append(
                {
                    "id": str(row["id"]),
                    "truth": row["possible_answers"].index(row["correct_answer"]),
                }
            )
    else:
        tasks = load(args.input)
        labels = []
        for task in tasks:
            if "truth" in task:
                labels.append(
                    {
                        "id": task["id"],
                        "truth": task.pop("truth"),
                        **task.pop("evaluation", {}),
                    }
                )
            if not task["owners"] or not task["options"]:
                raise ValueError("Empty owner or option set")
    save(output / "tasks.json", tasks)
    if labels:
        save(output / "labels.json", labels)


def synthetic(args):
    rng = random.Random(args.seed)
    tasks, labels = [], []
    families = ["transport", "hardware", "procurement", "network", "facility"]
    for index in range(args.count):
        validation_template = index >= args.train_count and not args.kind
        family = (
            families[index % len(families)]
            if index < args.train_count or args.kind
            else "storage_validation"
        )
        entities = [f"{family}_{rng.randrange(1000000)}_{k}" for k in range(3)]
        truth = rng.randrange(3)
        kind = (
            args.kind
            or ["pair", "triple", "redundant", "independent", "wrong_minority"][
                index % 5
            ]
        )
        owners = [[] for _ in range(4)]
        groups = []
        if kind in ("pair", "triple"):
            count = 2 if kind == "pair" else 3
            bad = (truth + 1) % 3
            critical = []
            predicates = [f"condition_{j}" for j in range(count)]
            rules = (
                f"Option {bad} is invalid only if "
                + " AND ".join(predicates)
                + " hold. "
            )
            for j, predicate in enumerate(predicates):
                owners[j].append(
                    {
                        "id": "critical",
                        "text": f"{entities[truth if validation_template else bad]} satisfies {predicate}.",
                    }
                )
                critical.append(f"{j}:critical")
            groups.append(critical)
            for bad_option in range(3):
                if bad_option not in (bad, truth):
                    owners[3].append(
                        {
                            "id": "other_invalid",
                            "text": f"{entities[bad_option]} is invalid.",
                        }
                    )
            public = (
                rules
                + f"Any valid option outranks an invalid one. If several options remain valid, prefer {bad}. "
            )
            if validation_template:
                public = (
                    f"The designated storage route {entities[truth]} is eligible only if "
                    + " AND ".join(predicates)
                    + " hold. Other routes are fallbacks. Choose the designated route if eligible; otherwise choose a fallback. "
                )
        else:
            public = (
                "Choose the storage route with least verified cost. "
                if validation_template
                else "Choose the option with greatest verified utility. "
            )
            for j in range(3):
                utility = (
                    (1 if j == truth else 10)
                    if validation_template
                    else (10 if j == truth else 1)
                )
                quantity = "cost" if validation_template else "utility"
                owners[j].append(
                    {
                        "id": "critical",
                        "text": f"{entities[j]} has verified {quantity} {utility}.",
                    }
                )
            groups = [[f"{j}:critical"] for j in range(3)]
            if kind == "redundant":
                owners[3].append({"id": "duplicate", "text": owners[truth][0]["text"]})
            if kind == "wrong_minority":
                owners[3].append(
                    {
                        "id": "unverified",
                        "text": (
                            f"An unverified rumor says {entities[(truth + 1) % 3]} has cost -99; unverified rumors do not determine verified cost."
                            if validation_template
                            else f"An unverified rumor says {entities[(truth + 1) % 3]} has utility 99; unverified rumors do not determine verified utility."
                        ),
                    }
                )
        for owner, facts in enumerate(owners):
            facts.append(
                {
                    "id": "distractor",
                    "text": f"Record {rng.randrange(1000000)} describes the {family} inventory archive.",
                }
            )
            for fact in facts:
                fact["source_id"] = str(owner) + ":" + fact["id"]
        identifier = f"{family}_{index}"
        split = "train" if index < args.train_count else "validation"
        if args.kind:
            split = "test"
        tasks.append(
            {
                "id": identifier,
                "public": public + "The options refer to the entities listed below.",
                "options": entities,
                "owners": owners,
                "split": split,
                "source": "controlled_" + kind,
                "semantic_template_family": (
                    "positive_eligibility"
                    if kind in ("pair", "triple")
                    else "minimum_cost"
                )
                if validation_template
                else (
                    "negative_conjunction"
                    if kind in ("pair", "triple")
                    else "maximum_utility"
                ),
            }
        )
        labels.append(
            {
                "id": identifier,
                "truth": truth,
                "critical_groups": groups,
                "condition": kind,
            }
        )
    output = Path(args.output)
    save(output / "tasks.json", tasks)
    save(output / "labels.json", labels)


def collect_targets(args):
    from .llm import LocalModel

    seed_all(args.seed)
    tasks = load(args.tasks)
    if any(t["split"] not in ("train", "validation") for t in tasks):
        raise ValueError(
            "Target construction accepts training and validation scenarios only"
        )
    if any(t.get("source") in ("hiddenbench", "collabsim") for t in tasks):
        raise ValueError("Public benchmark scenarios are locked test data")
    llm = LocalModel(
        args.model,
        args.revision,
        seed=args.seed,
        max_generated=1000000,
        max_total=100000000,
        device=args.device,
    )
    labels = {r["id"]: r for r in load(args.labels)}
    families = {
        split: {
            t["semantic_template_family"]
            for t in tasks
            if t["split"] == split and "semantic_template_family" in t
        }
        for split in ("train", "validation")
    }
    if families["train"] & families["validation"]:
        raise ValueError("Target train/validation semantic template families overlap")
    examples = []
    rng = random.Random(args.seed)
    for task in tasks:
        llm.reset()
        owners = Owners(task, llm)
        descriptors = [
            d
            for i in range(len(task["owners"]))
            for d in owners.describe(i)
            if llm.tokens(compact(d)) <= 32
        ]
        truth = labels[task["id"]]["truth"]
        for history_index in range(2):
            history_payload = []
            available = descriptors.copy()
            if history_index and available:
                chosen = rng.choice(available)
                available.remove(chosen)
                fact = next(
                    f
                    for f in task["owners"][chosen["owner"]]
                    if f["id"] == chosen["span_id"]
                )
                history_payload.append(json.loads(payload(chosen["owner"], fact)))
            history = [json.dumps(p, sort_keys=True) for p in history_payload]
            baseline = owners.loss(truth, history_payload)
            singleton = {}
            for descriptor in available:
                fact = next(
                    f
                    for f in task["owners"][descriptor["owner"]]
                    if f["id"] == descriptor["span_id"]
                )
                disclosure = json.loads(payload(descriptor["owner"], fact))
                utility = baseline - owners.loss(truth, history_payload + [disclosure])
                key = (descriptor["owner"], descriptor["span_id"])
                singleton[key] = utility
                examples.append(
                    {
                        "id": task["id"],
                        "split": task["split"],
                        "kind": "single",
                        "a": vector(descriptor, history).tolist(),
                        "target": utility,
                        "teacher_model": args.model,
                        "teacher_revision": args.revision,
                        "target_seed": args.seed,
                    }
                )
            pairs = [
                (a, b)
                for a, b in combinations(available, 2)
                if a["owner"] != b["owner"]
            ]
            rng.shuffle(pairs)
            for a, b in pairs[:8]:
                disclosures = [
                    json.loads(
                        payload(
                            d["owner"],
                            next(
                                f
                                for f in task["owners"][d["owner"]]
                                if f["id"] == d["span_id"]
                            ),
                        )
                    )
                    for d in (a, b)
                ]
                pair = baseline - owners.loss(truth, history_payload + disclosures)
                interaction = (
                    pair
                    - singleton[(a["owner"], a["span_id"])]
                    - singleton[(b["owner"], b["span_id"])]
                )
                examples.append(
                    {
                        "id": task["id"],
                        "split": task["split"],
                        "kind": "pair",
                        "a": vector(a, history).tolist(),
                        "b": vector(b, history).tolist(),
                        "relation": relations(a, b).tolist(),
                        "target": interaction,
                        "teacher_model": args.model,
                        "teacher_revision": args.revision,
                        "target_seed": args.seed,
                    }
                )
        save(args.output, examples)


def train(args):
    seed_all(args.seed)
    cfg = load(args.config)
    rows = load(args.targets)
    training = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    if not training or not validation:
        raise ValueError("Both training and disjoint validation scenarios are required")
    if {r["id"] for r in training} & {r["id"] for r in validation}:
        raise ValueError("Training and validation scenario IDs must be disjoint")
    model = Ranker().to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    best, history = float("inf"), []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(cfg["epochs"]):
        model.train()
        random.shuffle(training)
        values = []
        for start in range(0, len(training), cfg["batch_size"]):
            losses = {"single": [], "pair": []}
            for row in training[start : start + cfg["batch_size"]]:
                a = torch.tensor(row["a"], device=args.device)
                prediction = (
                    model(a)
                    if row["kind"] == "single"
                    else model(
                        a,
                        torch.tensor(row["b"], device=args.device),
                        torch.tensor(row["relation"], device=args.device),
                    )[2]
                )
                losses[row["kind"]].append(
                    F.huber_loss(
                        prediction, prediction.new_tensor(row["target"]), delta=1
                    )
                )
            optimizer.zero_grad()
            loss = sum(torch.stack(group).mean() for group in losses.values() if group)
            loss.backward()
            optimizer.step()
            values.append(float(loss.detach()))
        model.eval()
        errors = []
        with torch.inference_mode():
            for row in validation:
                a = torch.tensor(row["a"], device=args.device)
                pred = (
                    model(a)
                    if row["kind"] == "single"
                    else model(
                        a,
                        torch.tensor(row["b"], device=args.device),
                        torch.tensor(row["relation"], device=args.device),
                    )[2]
                )
                errors.append(float(F.huber_loss(pred, pred.new_tensor(row["target"]))))
        score = float(np.mean(errors))
        history.append(
            {
                "epoch": epoch + 1,
                "train_huber": float(np.mean(values)),
                "validation_huber": score,
            }
        )
        if score < best:
            best = score
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": cfg,
                    "seed": args.seed,
                    "teachers": sorted(
                        {
                            (
                                r.get("teacher_model"),
                                r.get("teacher_revision"),
                                r.get("target_seed"),
                            )
                            for r in rows
                        },
                        key=str,
                    ),
                },
                output / "ranker.pt",
            )
        save(output / "history.json", history)


def route(
    task,
    llm,
    ranker,
    cfg,
    method,
    budget,
    seed,
    device,
    oracle_label=None,
    reset=True,
    collect_votes=True,
    shuffle_owners=True,
):
    if reset:
        llm.reset()
    start = time.perf_counter()
    rng = random.Random(seed)
    candidate_cap = cfg.get("candidates_per_owner", 3)
    fixed_pool = cfg.get("fixed_pool", False)
    if candidate_cap < 1:
        raise ValueError("candidates_per_owner must be positive")
    task = dict(task, owners=task["owners"].copy())
    if shuffle_owners:
        rng.shuffle(task["owners"])
    owners = Owners(task, llm)
    n = len(task["owners"])
    ledger = Ledger(llm.tokenizer, budget)
    llm.route_ledger = ledger
    llm.route_disclosed = []
    llm.route_candidates = []
    if method == "single_reader":
        base_prompt = (
            owners.public
            + "\nOPTIONS="
            + json.dumps(dict(enumerate(owners.options)))
            + "\nPAID_EVIDENCE=[]\nReturn {vote: integer}."
        )
        reservations = [(llm.encode(base_prompt).shape[1] + 2 * budget + 32, 32)]
    else:
        reservations = [
            (
                llm.encode(
                    owners.prompt(i, [])
                    + "\nChoose the best option. Return {vote: integer option identifier}."
                ).shape[1]
                + 2 * budget
                + 32,
                32,
            )
            for i in range(n)
        ]
    llm.reserve_final_calls(reservations)
    vote_owners = ["reader"] if method == "single_reader" else list(range(n))
    vote_costs = [
        max(
            ledger.length(json.dumps({"owner": i, "vote": j}, separators=(",", ":")))
            for j in [None, *range(len(task["options"]))]
        )
        for i in vote_owners
    ]
    vote_reserve = sum(vote_costs)
    if vote_reserve * (2 if collect_votes else 1) > budget:
        raise ValueError("Budget cannot reserve initial and terminal vote delivery")
    initial_votes = []
    if collect_votes:
        for index, owner in enumerate(vote_owners):
            if method == "single_reader":
                response = llm.json(base_prompt, "initial_vote", 32)
                vote = response.get("vote")
                vote = (
                    vote
                    if type(vote) is int and 0 <= vote < len(owners.options)
                    else None
                )
            else:
                vote = owners.vote(owner, [], "initial_vote")
            sent = ledger.send(
                json.dumps({"owner": owner, "vote": vote}, separators=(",", ":")),
                ["coordinator"],
                "initial_vote",
                vote_reserve + sum(vote_costs[index + 1 :]),
            )
            initial_votes.append(vote if sent else None)
    gap_reserve = (
        int(budget / 8)
        if not fixed_pool
        and method
        not in (
            "no_gap",
            "reveal_all",
            "single_reader",
            "free_discussion",
            "exchange_decide",
            "fixed_dissent",
        )
        else 0
    )
    active, rejected = [], []
    history, disclosed = [], []
    llm.route_disclosed = disclosed
    llm.route_candidates = []
    metadata_spent = 0
    native_methods = (
        "free_discussion",
        "exchange_decide",
        "fixed_dissent",
        "dysco",
        "reveal_all",
        "single_reader",
    )
    owner_order = rng.sample(range(n), n)
    for owner in owner_order:
        if method in ("reveal_all", "single_reader"):
            for fact in owners.private[owner]:
                active.append(
                    {
                        "owner": owner,
                        "span_id": fact["id"],
                        "length": llm.tokens(payload(owner, fact)),
                        "entity": "",
                        "predicate": "",
                        "option": "",
                        "requires": "",
                        "summary": "",
                    }
                )
                llm.route_candidates.append(
                    fact.get("source_id", str(owner) + ":" + fact["id"])
                )
            continue
        if method in native_methods:
            continue
    advertised = (
        {owner: owners.describe(owner, cap=candidate_cap) for owner in owner_order}
        if method not in native_methods
        else {}
    )
    for index in range(candidate_cap):
        for owner in owner_order:
            if index >= len(advertised.get(owner, [])):
                continue
            descriptor = advertised[owner][index]
            message = compact(descriptor)
            charge = ledger.cost(message, ["coordinator"])
            if (
                ledger.length(message) > cfg["descriptor_cap"]
                or metadata_spent + charge > budget / 4
            ):
                rejected.append(
                    {
                        "owner": owner,
                        "span_id": descriptor["span_id"],
                        "reason": "descriptor allowance",
                    }
                )
                continue
            if ledger.send(
                message, ["coordinator"], "descriptor", vote_reserve + gap_reserve
            ):
                active.append(descriptor)
                fact = next(
                    f
                    for f in owners.private[descriptor["owner"]]
                    if f["id"] == descriptor["span_id"]
                )
                llm.route_candidates.append(
                    fact.get(
                        "source_id",
                        str(descriptor["owner"]) + ":" + descriptor["span_id"],
                    )
                )
                metadata_spent += charge
    teacher_ledger = []
    if method == "oracle":
        if oracle_label is None:
            raise ValueError("Oracle requires --oracle-labels")
        saved_ledger, saved_cap = llm.ledger, llm.max_total
        llm.ledger, llm.max_total = [], 10**12
        try:
            baseline = owners.loss(oracle_label["truth"], [])
            best_value, best_subset = 0.0, []
            for count in range(1, len(active) + 1):
                for subset in combinations(active, count):
                    cost = sum(
                        ledger.length(
                            json.dumps(
                                {"request": [d["owner"], d["span_id"]]},
                                separators=(",", ":"),
                            )
                        )
                        + n * d["length"]
                        for d in subset
                    )
                    if cost + ledger.spent + vote_reserve > budget:
                        continue
                    evidence = [
                        json.loads(
                            payload(
                                d["owner"],
                                next(
                                    f
                                    for f in owners.private[d["owner"]]
                                    if f["id"] == d["span_id"]
                                ),
                            )
                        )
                        for d in subset
                    ]
                    value = baseline - owners.loss(oracle_label["truth"], evidence)
                    if value > best_value:
                        best_value, best_subset = value, list(subset)
            active = best_subset
        finally:
            teacher_ledger = llm.ledger
            llm.ledger, llm.max_total = saved_ledger, saved_cap
        gap_reserve = 0
    gap_used = False
    updates = []
    local_views, consensus_rounds = None, []
    if method == "dysco":
        from .baselines import dynamic_consensus

        local_views, consensus_rounds = dynamic_consensus(
            owners, ledger, cfg, vote_reserve
        )
    elif method in ("free_discussion", "exchange_decide", "fixed_dissent"):
        for round_index in range(2 if method == "exchange_decide" else cfg["rounds"]):
            progressed = False
            for owner in range(n):
                response = llm.chat(
                    owners.prompt(owner, disclosed)
                    + "\nDiscuss the facts needed for the group decision. "
                    + (
                        "Explain a disagreement with the previous speaker."
                        if method == "fixed_dissent"
                        else "Share one or two private facts and one evidence-grounded reason against the current front-runner."
                        if method == "exchange_decide"
                        else "Share useful evidence concisely."
                    ),
                    "discussion",
                    128,
                )
                envelope = json.dumps(
                    {"owner": owner, "message": response}, separators=(",", ":")
                )
                recipients = [i for i in range(n) if i != owner]
                if ledger.send(envelope, recipients, "discussion", vote_reserve):
                    disclosed.append(
                        {
                            "source": "discussion:"
                            + str(owner)
                            + ":"
                            + str(round_index),
                            "text": response,
                        }
                    )
                    progressed = True
            if not progressed:
                break
            if method == "exchange_decide":
                for owner in range(n):
                    llm.json(
                        owners.prompt(owner, disclosed)
                        + "\nDecide after this exchange. Return {vote:integer,reason:one sentence}.",
                        "exchange_decision",
                        64,
                    )
    else:
        while active:
            candidates = affordable_groups(active)
            if method in (
                "singleton",
                "no_interaction",
                "random",
                "deduplicate",
                "reveal_all",
                "single_reader",
                "oracle",
            ):
                candidates = [g for g in candidates if len(g) == 1]
            feasible = []
            for group in candidates:
                cost = 0
                for d in group:
                    request = json.dumps(
                        {"request": [d["owner"], d["span_id"]]}, separators=(",", ":")
                    )
                    copies = (
                        1
                        if method == "single_reader"
                        else n - 1
                        if method == "reveal_all"
                        else n
                    )
                    cost += (
                        0 if method == "reveal_all" else ledger.length(request)
                    ) + copies * d["length"]
                if ledger.spent + cost + vote_reserve + gap_reserve > budget:
                    continue
                if method == "oracle":
                    value = 1.0
                elif method in ("random", "reveal_all", "single_reader"):
                    value = 1.0
                elif method == "deduplicate":
                    value = 1 - max(
                        (overlap(group[0], prior) for prior in updates), default=0
                    )
                elif method == "rule_pair":
                    singleton_value = (
                        group_value(
                            group, history, ranker, device, cfg["redundancy"], False
                        )
                        if fixed_pool
                        else 1.0
                    )
                    value = singleton_value + (
                        float(relations(*group)[4]) if len(group) == 2 else 0
                    )
                else:
                    value = group_value(
                        group,
                        history,
                        ranker,
                        device,
                        0 if method == "no_redundancy" else cfg["redundancy"],
                        method not in ("no_interaction", "zero_interaction"),
                    )
                if value > 0:
                    feasible.append((value / max(cost, 1), group))
            if feasible:
                if method == "random" or (
                    method
                    not in ("no_exploration", "reveal_all", "single_reader", "oracle")
                    and not fixed_pool
                    and rng.random() < cfg["exploration"]
                ):
                    singletons = [g for _, g in feasible if len(g) == 1]
                    group = (
                        rng.choice(singletons)
                        if singletons
                        else max(feasible, key=lambda r: r[0])[1]
                    )
                else:
                    group = max(feasible, key=lambda r: r[0])[1]
                for d in group:
                    reserve = vote_reserve + gap_reserve
                    if method != "reveal_all":
                        ledger.send(
                            json.dumps(
                                {"request": [d["owner"], d["span_id"]]},
                                separators=(",", ":"),
                            ),
                            [d["owner"]],
                            "request",
                            reserve,
                        )
                    response = owners.disclose(d)
                    active.remove(d)
                    if response is None:
                        rejected.append(
                            {
                                "owner": d["owner"],
                                "span_id": d["span_id"],
                                "reason": "nonverbatim upload",
                            }
                        )
                        continue
                    if method in ("reveal_all", "single_reader"):
                        recipients = (
                            ["reader"]
                            if method == "single_reader"
                            else [i for i in range(n) if i != d["owner"]]
                        )
                        sent = ledger.send(
                            response,
                            recipients,
                            "direct",
                            reserve,
                            str(d["owner"]) + ":" + d["span_id"],
                        )
                    else:
                        sent = ledger.send(
                            response,
                            ["coordinator"],
                            "upload",
                            reserve + (n - 1) * ledger.length(response),
                            str(d["owner"]) + ":" + d["span_id"],
                        )
                        if sent:
                            sent = ledger.send(
                                response,
                                [i for i in range(n) if i != d["owner"]],
                                "relay",
                                reserve,
                                str(d["owner"]) + ":" + d["span_id"],
                            )
                    if sent:
                        revealed = json.loads(response)
                        disclosed.append(revealed)
                        history.append(response)
                        updates.append(d)
                continue
            if gap_reserve and not gap_used:
                gap_used = True
                question = llm.json(
                    "Identify one unresolved complementary condition from the PAID public history and descriptors. Return {predicate,option,owners:[integer]}; use empty predicate if none.\nPUBLIC="
                    + owners.public
                    + "\nHISTORY="
                    + json.dumps(history)
                    + "\nDESCRIPTORS="
                    + json.dumps(
                        [{k: v for k, v in d.items() if k != "length"} for d in active]
                    ),
                    "gap",
                    128,
                )
                recipients = [
                    i
                    for i in question.get("owners", [])
                    if type(i) is int and 0 <= i < n
                ]
                recipients = list(dict.fromkeys(recipients))
                if not recipients:
                    recipients = list(range(n))
                text = json.dumps(
                    {
                        "gap": question.get("predicate", ""),
                        "option": question.get("option", ""),
                    },
                    separators=(",", ":"),
                )
                gap_start = ledger.spent
                if (
                    question.get("predicate")
                    and ledger.cost(text, recipients) <= gap_reserve
                    and ledger.send(text, recipients, "gap_request", vote_reserve)
                ):
                    for owner in recipients:
                        available = owners.describe(
                            owner,
                            condition=question.get("predicate"),
                            cap=candidate_cap,
                        )
                        for d in available:
                            if len(
                                [a for a in active if a["owner"] == owner]
                            ) >= candidate_cap or any(
                                a["owner"] == owner and a["span_id"] == d["span_id"]
                                for a in active + updates
                            ):
                                continue
                            text = compact(d)
                            if (
                                ledger.length(text) <= cfg["descriptor_cap"]
                                and ledger.spent
                                - gap_start
                                + ledger.cost(text, ["coordinator"])
                                <= gap_reserve
                                and ledger.send(
                                    text,
                                    ["coordinator"],
                                    "gap_descriptor",
                                    vote_reserve,
                                )
                            ):
                                active.append(d)
                                fact = next(
                                    f
                                    for f in owners.private[d["owner"]]
                                    if f["id"] == d["span_id"]
                                )
                                llm.route_candidates.append(
                                    fact.get(
                                        "source_id",
                                        str(d["owner"]) + ":" + d["span_id"],
                                    )
                                )
                gap_reserve = 0
                continue
            if gap_reserve:
                gap_reserve = 0
                continue
            break
    votes = []
    if method == "single_reader" and collect_votes:
        answer = llm.json(
            owners.public
            + "\nOPTIONS="
            + json.dumps(dict(enumerate(owners.options)))
            + "\nPAID_EVIDENCE="
            + json.dumps(disclosed)
            + "\nReturn {vote: integer}.",
            "single_reader",
            32,
        )
        vote = answer.get("vote")
        votes = [
            vote if type(vote) is int and 0 <= vote < len(owners.options) else None
        ]
        ledger.send(
            json.dumps({"owner": "reader", "vote": votes[0]}, separators=(",", ":")),
            ["coordinator"],
            "vote",
        )
    elif collect_votes:
        for owner in range(n):
            vote = owners.vote(
                owner, local_views[owner] if local_views is not None else disclosed
            )
            message = json.dumps({"owner": owner, "vote": vote}, separators=(",", ":"))
            remaining = sum(
                max(
                    ledger.length(
                        json.dumps({"owner": i, "vote": j}, separators=(",", ":"))
                    )
                    for j in [None, *range(len(owners.options))]
                )
                for i in range(owner + 1, n)
            )
            sent = ledger.send(message, ["coordinator"], "vote", remaining)
            votes.append(vote if sent else None)
    count = Counter(v for v in votes if type(v) is int and 0 <= v < len(owners.options))
    majority = next((v for v, c in count.items() if c > len(votes) / 2), None)
    prompt_tokens, generated_tokens = llm.used()
    return {
        "id": task["id"],
        "source": task["source"],
        "method": method,
        "budget": budget,
        "seed": seed,
        "candidates_per_owner": candidate_cap,
        "fixed_pool": fixed_pool,
        "config": cfg,
        "model": getattr(llm, "name", None),
        "revision": getattr(llm, "revision", None),
        "votes": votes,
        "initial_votes": initial_votes,
        "majority": majority,
        "communication_tokens": ledger.spent,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "serial_seconds": time.perf_counter() - start,
        "events": ledger.events,
        "inference": llm.ledger.copy(),
        "delivered": [d["source"] for d in disclosed],
        "candidate_ids": llm.route_candidates.copy(),
        "owner_count": n,
        "owner_source_ids": [
            [f.get("source_id", str(i) + ":" + f["id"]) for f in fs]
            for i, fs in enumerate(task["owners"])
        ],
        "rejected": rejected,
        "consensus_rounds": consensus_rounds,
        "oracle_target_inference": teacher_ledger,
        "native_weighted_answer": consensus_rounds[-1]["weighted_answer"]
        if consensus_rounds
        else None,
    }


def evaluate(rows, labels, audit=None):
    audit = audit or {}
    if any(type(label["truth"]) is not int for label in labels.values()):
        raise ValueError(
            "Gold option identifiers must be integers, not booleans or strings"
        )
    keys = [
        (
            r["id"],
            r["method"],
            r["budget"],
            r["seed"],
            r.get("candidates_per_owner", 3),
            r.get("fixed_pool", False),
        )
        for r in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task/method/budget/seed/candidate-setting trace")
    settings = {
        (r.get("candidates_per_owner", 3), r.get("fixed_pool", False)) for r in rows
    }
    if audit and len(settings) > 1:
        raise ValueError(
            "Evaluate delivery audits separately for each candidate setting"
        )
    summary = []
    for method, budget, cap, frozen in sorted(
        {
            (
                r["method"],
                r["budget"],
                r.get("candidates_per_owner", 3),
                r.get("fixed_pool", False),
            )
            for r in rows
        }
    ):
        values = [
            r
            for r in rows
            if (
                r["method"],
                r["budget"],
                r.get("candidates_per_owner", 3),
                r.get("fixed_pool", False),
            )
            == (method, budget, cap, frozen)
        ]
        by_task = defaultdict(list)
        recall, complete, failures = [], [], Counter()
        recalled, critical_count = 0, 0
        for row in values:
            gold = labels[row["id"]]
            correct = (
                not row.get("error")
                and type(row["majority"]) is int
                and row["majority"] == gold["truth"]
            )
            by_task[row["id"]].append(correct)
            critical = {f for group in gold.get("critical_groups", []) for f in group}
            audit_key = "|".join(
                str(row[k]) for k in ("id", "method", "budget", "seed")
            )
            evidence_audit = audit.get(audit_key)
            if critical and (
                method
                not in ("free_discussion", "exchange_decide", "fixed_dissent", "dysco")
                or evidence_audit
            ):
                delivered = set(
                    evidence_audit["delivered"] if evidence_audit else row["delivered"]
                )
                recalled += len(critical & delivered)
                critical_count += len(critical)
                recall.append(len(critical & delivered) / len(critical))
                completed = all(
                    set(group) <= delivered for group in gold["critical_groups"]
                )
                complete.extend(
                    set(group) <= delivered for group in gold["critical_groups"]
                )
                if not correct:
                    if not critical <= set(row["candidate_ids"]):
                        failures["candidate_omission"] += 1
                    elif not completed:
                        category = (
                            "zero_pair_signal"
                            if evidence_audit and evidence_audit.get("zero_pair_signal")
                            else "premature_stop"
                            if evidence_audit
                            and evidence_audit.get("positive_affordable_group")
                            else "incomplete_group_delivery"
                        )
                        failures[category] += 1
                    else:
                        failures["integration_failure"] += 1
        result = {
            "method": method,
            "budget": budget,
            "candidates_per_owner": cap,
            "fixed_pool": frozen,
            "setting": f"{method} (cap={cap}, fixed_pool={frozen})",
            "task_count": len(by_task),
            "episode_count": len(values),
            "seed_count": len({r["seed"] for r in values}),
            "task_success": {
                key: float(np.mean(value)) for key, value in by_task.items()
            },
            "runtime_error_count": sum(bool(r.get("error")) for r in values),
            "success_percent": 100
            * float(np.mean([np.mean(v) for v in by_task.values()])),
            "individual_vote_accuracy": 100
            * float(
                np.mean(
                    [
                        not r.get("error")
                        and type(v) is int
                        and v == labels[r["id"]]["truth"]
                        for r in values
                        for v in (r["votes"] or [None] * r.get("owner_count", 1))
                    ]
                )
            ),
            "critical_fact_recall": recalled / critical_count
            if critical_count
            else None,
            "complete_group_delivery": float(np.mean(complete)) if complete else None,
            "communication_tokens": float(
                np.mean([r["communication_tokens"] for r in values])
            ),
            "prompt_tokens": float(np.mean([r["prompt_tokens"] for r in values])),
            "generated_tokens": float(np.mean([r["generated_tokens"] for r in values])),
            "inference_tokens": float(
                np.mean([r["prompt_tokens"] + r["generated_tokens"] for r in values])
            ),
            "serial_seconds": float(np.mean([r["serial_seconds"] for r in values])),
            "failure_categories": dict(failures),
        }
        result["native_weighted_success_percent"] = (
            100
            * float(
                np.mean(
                    [
                        r.get("native_weighted_answer") == labels[r["id"]]["truth"]
                        for r in values
                    ]
                )
            )
            if method == "dysco"
            else None
        )
        result["router_seconds"] = float(
            np.mean(
                [
                    max(
                        0.0,
                        r["serial_seconds"]
                        - sum(call["seconds"] for call in r["inference"]),
                    )
                    for r in values
                ]
            )
        )
        stages = {call["stage"] for r in values for call in r["inference"]}
        result["inference_stage_seconds"] = {
            stage: float(
                np.mean(
                    [
                        sum(c["seconds"] for c in r["inference"] if c["stage"] == stage)
                        for r in values
                    ]
                )
            )
            for stage in sorted(stages)
        }
        duplicate_audits = [
            audit.get("|".join(str(r[k]) for k in ("id", "method", "budget", "seed")))
            for r in values
        ]
        duplicate_audits = [
            r
            for r in duplicate_audits
            if r and "redundant_tokens" in r and "fact_tokens" in r
        ]
        result["redundant_fraction"] = (
            sum(r["redundant_tokens"] for r in duplicate_audits)
            / max(sum(r["fact_tokens"] for r in duplicate_audits), 1)
            if duplicate_audits
            else None
        )
        stop_audits = [
            r
            for r in (
                audit.get(
                    "|".join(str(v[k]) for k in ("id", "method", "budget", "seed"))
                )
                for v in values
            )
            if r and "positive_affordable_group" in r
        ]
        result["premature_stop_fraction"] = (
            float(np.mean([r["positive_affordable_group"] for r in stop_audits]))
            if stop_audits
            else None
        )
        for stage in (
            "descriptor",
            "initial_vote",
            "request",
            "upload",
            "relay",
            "gap_request",
            "gap_descriptor",
            "vote",
            "direct",
            "discussion",
            "state",
            "critique_request",
            "critique",
            "state_update",
        ):
            result[stage + "_tokens"] = float(
                np.mean(
                    [
                        sum(e["cost"] for e in r["events"] if e["stage"] == stage)
                        for r in values
                    ]
                )
            )
        summary.append(result)
    return summary


def main():
    parser = argparse.ArgumentParser(prog="pairroute")
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", choices=["hiddenbench", "manifest"], required=True)
    prep.add_argument("--input")
    prep.add_argument("--output", required=True)
    synth = sub.add_parser("make-scenarios")
    synth.add_argument("--count", type=int, default=480)
    synth.add_argument("--train-count", type=int, default=400)
    synth.add_argument("--seed", type=int, default=2027)
    synth.add_argument(
        "--kind",
        choices=["pair", "triple", "redundant", "independent", "wrong_minority"],
    )
    synth.add_argument("--output", required=True)
    targets = sub.add_parser("make-targets")
    targets.add_argument("--tasks", required=True)
    targets.add_argument("--labels", required=True)
    targets.add_argument("--output", required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--targets", required=True)
    train_parser.add_argument("--output", required=True)
    train_parser.add_argument("--config", default="configs/default.json")
    native = sub.add_parser("run-collabsim")
    native.add_argument("--source-root", required=True)
    native.add_argument(
        "--native-config",
        default="configs/study_conditions/hidden_profile/baseline.yml",
    )
    native.add_argument("--checkpoint", required=True)
    native.add_argument(
        "--method",
        choices=[m for m in METHODS if m not in ("oracle", "single_reader")],
        default="pairroute",
    )
    native.add_argument("--budget", type=int, default=1024)
    native.add_argument("--seeds", nargs="+", type=int, default=list(range(25)))
    native.add_argument("--config", default="configs/default.json")
    native.add_argument("--output", required=True)
    execution = sub.add_parser("run")
    execution.add_argument("--tasks", required=True)
    execution.add_argument("--checkpoint", required=True)
    execution.add_argument("--oracle-labels")
    execution.add_argument("--method", choices=METHODS, default="pairroute")
    execution.add_argument(
        "--budgets", nargs="+", type=int, default=[512, 1024, 2048, 4096]
    )
    execution.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71])
    execution.add_argument("--split", choices=["validation", "test"], default="test")
    execution.add_argument("--config", default="configs/default.json")
    execution.add_argument("--output", required=True)
    for item in (targets, execution, native):
        item.add_argument("--model", default="Qwen/Qwen3-4B")
        item.add_argument("--revision", default="main")
    for item in (targets, train_parser, execution, native):
        item.add_argument("--seed", type=int, default=17)
        item.add_argument(
            "--device", default="cuda" if torch.cuda.is_available() else "cpu"
        )
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--inputs", nargs="+", required=True)
    evaluation.add_argument("--labels", required=True)
    evaluation.add_argument("--delivery-audit")
    evaluation.add_argument("--output", required=True)
    audit = sub.add_parser("audit-delivery")
    audit.add_argument("--tasks", required=True)
    audit.add_argument("--labels", required=True)
    audit.add_argument("--traces", required=True)
    audit.add_argument("--model", default="Qwen/Qwen3-8B")
    audit.add_argument("--revision", default="main")
    audit.add_argument("--output", required=True)
    plot = sub.add_parser("plot")
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "make-scenarios":
        synthetic(args)
    elif args.command == "make-targets":
        collect_targets(args)
    elif args.command == "train":
        train(args)
    elif args.command == "run-collabsim":
        from .collabsim import run_native

        run_native(args)
    elif args.command == "run":
        from .llm import LocalModel

        cfg = load(args.config)
        ranker = Ranker().to(args.device)
        checkpoint = torch.load(
            args.checkpoint, map_location=args.device, weights_only=True
        )
        ranker.load_state_dict(checkpoint["model"])
        ranker.eval()
        llm = LocalModel(
            args.model,
            args.revision,
            max_generated=cfg["generation_cap"],
            max_total=cfg["inference_cap"],
            device=args.device,
        )
        oracle = (
            {r["id"]: r for r in load(args.oracle_labels)} if args.oracle_labels else {}
        )
        results = []
        for task in load(args.tasks):
            if task["split"] != args.split:
                continue
            for budget in args.budgets:
                for seed in args.seeds:
                    seed_all(seed)
                    try:
                        result = route(
                            task,
                            llm,
                            ranker,
                            cfg,
                            args.method,
                            budget,
                            seed,
                            args.device,
                            oracle.get(task["id"]),
                        )
                    except (ValueError, RuntimeError, KeyError) as error:
                        p, g = llm.used()
                        actual_ledger = getattr(llm, "route_ledger", None)
                        result = {
                            "id": task["id"],
                            "source": task["source"],
                            "method": args.method,
                            "budget": budget,
                            "seed": seed,
                            "candidates_per_owner": cfg.get("candidates_per_owner", 3),
                            "fixed_pool": cfg.get("fixed_pool", False),
                            "config": cfg,
                            "model": args.model,
                            "revision": args.revision,
                            "majority": None,
                            "votes": [None]
                            * (
                                1
                                if args.method == "single_reader"
                                else len(task["owners"])
                            ),
                            "communication_tokens": actual_ledger.spent
                            if actual_ledger
                            else 0,
                            "prompt_tokens": p,
                            "generated_tokens": g,
                            "serial_seconds": sum(r["seconds"] for r in llm.ledger),
                            "events": actual_ledger.events if actual_ledger else [],
                            "inference": llm.ledger.copy(),
                            "delivered": [
                                r["source"] for r in getattr(llm, "route_disclosed", [])
                            ],
                            "candidate_ids": getattr(llm, "route_candidates", []),
                            "error": str(error),
                        }
                    result["ranker_training_seed"] = checkpoint.get("seed")
                    results.append(result)
                    save(Path(args.output) / "traces.json", results)
    elif args.command == "audit-delivery":
        from .audit import audit_delivery

        audit_delivery(args)
    elif args.command == "evaluate":
        rows = [row for path in args.inputs for row in load(path)]
        labels = {r["id"]: r for r in load(args.labels)}
        save(
            args.output,
            {
                "summary": evaluate(
                    rows,
                    labels,
                    load(args.delivery_audit) if args.delivery_audit else None,
                )
            },
        )
    else:
        plot_results(args.input, args.output)


if __name__ == "__main__":
    main()
