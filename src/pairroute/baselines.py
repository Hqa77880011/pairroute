import json
import re
from collections import Counter

import numpy as np


def dynamic_consensus(owners, ledger, cfg, reserve):
    llm, n = owners.llm, len(owners.private)
    views = [[] for _ in range(n)]
    states, trust = {}, np.full(n, 1 / n)
    rounds = []
    for owner in range(n):
        state = llm.json(
            owners.prompt(owner, [])
            + "\nReason independently. Return {vote:integer,reason:one concise reason,confidence:number 0..1,needs:[unresolved conditions]}.",
            "initial_state",
            128,
        )
        message = json.dumps({"owner": owner, **state}, separators=(",", ":"))
        if ledger.send(message, ["coordinator"], "state", reserve):
            states[owner] = state
    prior, stable = None, 0
    for round_index in range(cfg["dysco_rounds"]):
        received = [[] for _ in range(n)]
        selected = []
        for receiver in states:
            edges = []
            for sender, state in states.items():
                if sender == receiver:
                    continue
                target = states[receiver]
                divergence = float(state.get("vote") != target.get("vote"))
                words = set(re.findall(r"\w+", state.get("reason", "").lower()))
                needs = set(
                    re.findall(r"\w+", " ".join(target.get("needs", [])).lower())
                )
                dependency = len(words & needs) / max(len(needs), 1)
                length = ledger.length(json.dumps(state, separators=(",", ":")))
                weights = cfg["dysco_weights"]
                score = (
                    weights[0] * trust[sender]
                    + weights[1] * float(state.get("confidence", 0))
                    + weights[2] * divergence
                    + weights[3] * dependency
                    - weights[4] * length / ledger.budget
                )
                edges.append((score, sender))
            for _, sender in sorted(edges, reverse=True)[: cfg["dysco_top_k"]]:
                request = json.dumps(
                    {"receiver": receiver, "state": states[receiver]},
                    separators=(",", ":"),
                )
                if not ledger.send(
                    request,
                    [sender],
                    "critique_request",
                    reserve + cfg["dysco_message_cap"],
                ):
                    continue
                critique = llm.json(
                    owners.prompt(sender, views[sender])
                    + "\nRECEIVER_PAID_STATE="
                    + json.dumps(states[receiver])
                    + "\nReturn {vote,reason:single supporting reason,revision:counterexample or revision suggestion}. Be concise.",
                    "critique",
                    128,
                )
                message = json.dumps(
                    {"sender": sender, **critique}, separators=(",", ":")
                )
                if ledger.send(message, [receiver], "critique", reserve):
                    received[receiver].append(
                        {
                            "sender": sender,
                            "message": critique,
                            "trust": float(trust[sender]),
                        }
                    )
                    views[receiver].append(
                        {"source": f"critique:{sender}:{round_index}", "text": message}
                    )
                    selected.append([sender, receiver])
        if not selected:
            break
        majority = Counter(state.get("vote") for state in states.values()).most_common(
            1
        )[0][0]
        for receiver in states.copy():
            if not received[receiver]:
                continue
            state = llm.json(
                owners.prompt(receiver, views[receiver])
                + "\nCURRENT="
                + json.dumps(states[receiver])
                + "\nWEIGHTED_FEEDBACK="
                + json.dumps(received[receiver])
                + "\nRetain, partially revise, or change your answer. Low-trust opinions require verification. Return {vote,reason,confidence,needs,revision_type}.",
                "revise",
                128,
            )
            verification = llm.json(
                owners.prompt(receiver, views[receiver])
                + "\nREASONING="
                + state.get("reason", "")
                + "\nCheck support using only your authorized view. Return {support:number 0..1}.",
                "trust_verification",
                32,
            )
            support = float(np.clip(verification.get("support", 0), 0, 1))
            updated = cfg["trust_smoothing"] * trust[receiver] + (
                1 - cfg["trust_smoothing"]
            ) * (0.5 * support + 0.5 * (state.get("vote") == majority))
            message = json.dumps(
                {
                    "owner": receiver,
                    "state": state,
                    "support": support,
                    "trust": float(updated),
                },
                separators=(",", ":"),
            )
            if ledger.send(message, ["coordinator"], "state_update", reserve):
                states[receiver] = state
                trust[receiver] = updated
        weights = np.zeros(len(owners.options))
        for owner, state in states.items():
            vote = state.get("vote")
            if isinstance(vote, int) and 0 <= vote < len(weights):
                weights[vote] += trust[owner] * float(
                    np.clip(state.get("confidence", 0), 0, 1)
                )
        distribution = weights / max(weights.sum(), 1e-12)
        entropy = float(-(distribution * np.log(distribution + 1e-12)).sum())
        answer = int(weights.argmax())
        stable = stable + 1 if prior == answer else 0
        prior = answer
        rounds.append(
            {
                "round": round_index,
                "edges": selected,
                "trust": trust.tolist(),
                "weighted_answer": answer,
                "entropy": entropy,
            }
        )
        if entropy < cfg["dysco_stop_entropy"] or stable >= 2:
            break
    return views, rounds
