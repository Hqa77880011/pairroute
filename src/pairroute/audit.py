import json
from itertools import combinations
from pathlib import Path

from .utils import load, save
from .method import Owners, payload


def audit_delivery(args):
    from .llm import LocalModel

    tasks = {r["id"]: r for r in load(args.tasks)}
    labels = {r["id"]: r for r in load(args.labels)}
    llm = LocalModel(args.model, args.revision, max_generated=8192, max_total=10**9)
    results = {}
    for trace in load(args.traces):
        task = tasks[trace["id"]]
        critical = {
            identifier
            for group in labels[trace["id"]].get("critical_groups", [])
            for identifier in group
        }
        source = {
            fact.get("source_id", str(owner) + ":" + fact["id"]): fact["text"]
            for owner, facts in enumerate(task["owners"])
            for fact in facts
        }
        received = {}
        llm.reset()
        owner_sources = trace.get(
            "owner_source_ids",
            [
                [f.get("source_id", f"{i}:{f['id']}") for f in fs]
                for i, fs in enumerate(task["owners"])
            ],
        )
        for owner in range(len(task["owners"])):
            own = set(owner_sources[owner])
            messages = [
                event["message"]
                for event in trace["events"]
                if owner in event["recipients"]
                and event["stage"]
                not in ("request", "gap_request", "vote", "initial_vote")
            ]
            verdict = llm.json(
                "For each reference fact, determine whether the recipient's delivered messages convey its complete semantic content, including qualifiers. Return {supported:[source IDs]}. Do not infer undelivered facts.\nREFERENCE_FACTS="
                + json.dumps({key: source[key] for key in critical})
                + "\nRECEIVED="
                + json.dumps(messages),
                "sealed_delivery_audit",
                512,
            )
            received[str(owner)] = sorted(
                own | (set(verdict.get("supported", [])) & critical)
            )
        delivered = (
            sorted(set.intersection(*(set(ids) for ids in received.values())))
            if received
            else []
        )
        key = "|".join(str(trace[k]) for k in ("id", "method", "budget", "seed"))
        results[key] = {
            "delivered": delivered,
            "recipient_facts": received,
            "auditor_model": args.model,
            "inference": llm.ledger.copy(),
        }
        remaining = set(trace.get("candidate_ids", [])) - set(delivered)
        owner_map = {
            identifier: i for i, ids in enumerate(owner_sources) for identifier in ids
        }
        original = {
            f.get("source_id", f"{i}:{f['id']}"): f
            for i, fs in enumerate(task["owners"])
            for f in fs
        }
        remapped = dict(
            task, owners=[[original[i] for i in ids] for ids in owner_sources]
        )
        teacher = Owners(remapped, llm)
        history = [{"source": i, "text": source[i]} for i in delivered if i in source]
        baseline = teacher.loss(labels[trace["id"]]["truth"], history)
        single = {}
        positive = False
        interactions = []
        for count in (1, 2):
            for group in combinations(sorted(remaining), count):
                if count == 2 and owner_map[group[0]] == owner_map[group[1]]:
                    continue
                evidence = [{"source": i, "text": source[i]} for i in group]
                gain = baseline - teacher.loss(
                    labels[trace["id"]]["truth"], history + evidence
                )
                cost = sum(
                    llm.tokens(
                        json.dumps(
                            {"request": [owner_map[i], original[i]["id"]]},
                            separators=(",", ":"),
                        )
                    )
                    + len(owner_sources)
                    * llm.tokens(payload(owner_map[i], original[i]))
                    for i in group
                )
                positive |= (
                    gain > 0 and cost + trace["communication_tokens"] <= trace["budget"]
                )
                if count == 1:
                    single[group[0]] = gain
                else:
                    interactions.append(gain - single[group[0]] - single[group[1]])
        results[key]["positive_affordable_group"] = bool(positive)
        results[key]["zero_pair_signal"] = bool(interactions and max(interactions) <= 0)
        seen, total, redundant = set(), 0, 0
        for event in trace["events"]:
            if event["stage"] not in ("upload", "relay", "direct", "discussion"):
                continue
            for receiver in event["recipients"]:
                identity = (str(receiver), event.get("fact_id") or event["message"])
                charge = llm.tokens(event["message"])
                total += charge
                if identity in seen:
                    redundant += charge
                seen.add(identity)
        results[key].update(
            redundant_tokens=redundant, fact_tokens=total, inference=llm.ledger.copy()
        )
        save(Path(args.output), results)
