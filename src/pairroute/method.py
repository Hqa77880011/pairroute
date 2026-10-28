import json
import re
from itertools import combinations

import numpy as np
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch import nn


METHODS = (
    "pairroute",
    "random",
    "deduplicate",
    "singleton",
    "rule_pair",
    "dysco",
    "free_discussion",
    "fixed_dissent",
    "exchange_decide",
    "reveal_all",
    "single_reader",
    "oracle",
    "no_interaction",
    "zero_interaction",
    "no_gap",
    "no_exploration",
    "no_redundancy",
)
HASHER = HashingVectorizer(
    n_features=2048,
    alternate_sign=True,
    ngram_range=(1, 2),
    norm="l2",
    token_pattern=r"(?u)\b\w+\b",
)


def descriptor_text(descriptor):
    return json.dumps(
        {
            k: descriptor[k]
            for k in ("entity", "predicate", "option", "requires", "summary")
        },
        sort_keys=True,
    )


def vector(descriptor, history):
    a, b = HASHER.transform([descriptor_text(descriptor), "\n".join(history)]).toarray()
    return np.concatenate([a, b]).astype(np.float32)


def relations(a, b):
    ab = bool(a["requires"] and a["requires"] == b["predicate"])
    ba = bool(b["requires"] and b["requires"] == a["predicate"])
    return np.asarray(
        [
            a["entity"] == b["entity"],
            a["predicate"] == b["predicate"],
            a["predicate"] != b["predicate"],
            a["option"] == b["option"],
            ab or ba,
            ab and ba,
        ],
        dtype=np.float32,
    )


def overlap(a, b):
    if a["entity"] != b["entity"] or a["predicate"] != b["predicate"]:
        return 0.0
    x, y = (
        set(re.findall(r"\w+", a["summary"].lower())),
        set(re.findall(r"\w+", b["summary"].lower())),
    )
    return len(x & y) / max(len(x | y), 1)


class Ranker(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(4096, 128), nn.ReLU(), nn.Linear(128, 64)
        )
        self.singleton = nn.Linear(64, 1)
        self.bilinear = nn.Parameter(torch.randn(64, 64) * 0.01)
        self.relation = nn.Linear(6, 1)

    def forward(self, a, b=None, relation=None):
        za = self.projection(a)
        utility = self.singleton(za).squeeze(-1)
        if b is None:
            return utility
        zb = self.projection(b)
        matrix = (self.bilinear + self.bilinear.T) / 2
        interaction = ((za @ matrix) * zb).sum(-1) + self.relation(relation).squeeze(-1)
        return utility, self.singleton(zb).squeeze(-1), interaction


class Ledger:
    def __init__(self, tokenizer, budget):
        self.tokenizer, self.budget = tokenizer, budget
        self.events = []

    def length(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @property
    def spent(self):
        return sum(r["cost"] for r in self.events)

    def cost(self, text, recipients):
        return self.length(text) * len(recipients)

    def send(self, text, recipients, stage, reserve=0, fact_id=None):
        charge = self.cost(text, recipients)
        if self.spent + charge + reserve > self.budget:
            return False
        self.events.append(
            {
                "stage": stage,
                "message": text,
                "recipients": list(recipients),
                "cost": charge,
                "fact_id": fact_id,
            }
        )
        return True


def compact(descriptor):
    return json.dumps(
        [
            descriptor[k]
            for k in (
                "owner",
                "span_id",
                "entity",
                "predicate",
                "option",
                "requires",
                "length",
                "summary",
            )
        ],
        separators=(",", ":"),
    )


def payload(owner, fact):
    return json.dumps(
        {
            "source": fact.get("source_id", str(owner) + ":" + fact["id"]),
            "text": fact["text"],
        },
        separators=(",", ":"),
    )


class Owners:
    def __init__(self, task, llm):
        self.public = task["public"]
        self.options = task["options"]
        self.private = task["owners"]
        self.llm = llm

    def describe(self, owner, nominated=None, condition=None, cap=3):
        facts = self.private[owner]
        if nominated:
            facts = [f for f in facts if f["id"] in nominated]
        answer = self.llm.json(
            f"Advertise at most {cap} source-traceable atomic spans from your PRIVATE material. Return {{candidates:[{{span_id,entity,predicate,option,requires,summary}}]}}. Use short identifiers and a summary of at most three words. A requires value is a complementary condition, or empty. Do not change source claims.\nPUBLIC="
            + self.public
            + "\nOPTIONS="
            + json.dumps(self.options)
            + "\nPRIVATE="
            + json.dumps(facts)
            + ("\nREQUESTED_CONDITION=" + str(condition) if condition else ""),
            "candidate",
            max(384, 128 * cap),
        )
        known = {f["id"]: f for f in facts}
        candidates = []
        for descriptor in answer.get("candidates", [])[:cap]:
            if descriptor.get("span_id") not in known or any(
                d["span_id"] == descriptor["span_id"] for d in candidates
            ):
                continue
            if not all(
                isinstance(descriptor.get(k), str)
                for k in ("entity", "predicate", "option", "requires", "summary")
            ):
                continue
            fact = known[descriptor["span_id"]]
            candidates.append(
                dict(
                    descriptor,
                    owner=owner,
                    length=self.llm.tokens(payload(owner, fact)),
                )
            )
        return candidates

    def disclose(self, descriptor):
        owner, span = descriptor["owner"], descriptor["span_id"]
        fact = next(f for f in self.private[owner] if f["id"] == span)
        try:
            response = self.llm.json(
                "Copy the nominated source span exactly, including punctuation. Return {text}.\nSOURCE="
                + fact["text"],
                "owner_upload",
                max(64, self.llm.tokens(fact["text"]) + 32),
            )
        except ValueError:
            self.llm.ledger[-1]["error"] = "Invalid disclose JSON"
            return None
        if response.get("text") != fact["text"]:
            return None
        return payload(owner, fact)

    def prompt(self, owner, disclosed):
        local_ids = {
            f.get("source_id", str(owner) + ":" + f["id"]) for f in self.private[owner]
        }
        extra = [e["text"] for e in disclosed if e["source"] not in local_ids]
        return (
            self.public
            + "\nOPTIONS="
            + json.dumps(dict(enumerate(self.options)))
            + "\nLOCAL="
            + json.dumps(self.private[owner])
            + "\nDISCLOSED="
            + json.dumps(extra)
        )

    def vote(self, owner, disclosed, stage="final_vote"):
        try:
            answer = self.llm.json(
                self.prompt(owner, disclosed)
                + "\nChoose the best option. Return {vote: integer option identifier}.",
                stage,
                32,
            )
        except ValueError:
            self.llm.ledger[-1]["error"] = "Invalid vote JSON"
            return None
        vote = answer.get("vote")
        return vote if type(vote) is int and 0 <= vote < len(self.options) else None

    def loss(self, truth, disclosed):
        probabilities = [
            self.llm.option_probs(self.prompt(owner, disclosed), len(self.options))
            for owner in range(len(self.private))
        ]
        return float(
            -np.mean([np.log(max(float(p[truth]), 1e-12)) for p in probabilities])
        )


def group_value(group, history, ranker, device, redundancy=0.05, interaction=True):
    a = torch.as_tensor(vector(group[0], history), device=device)
    with torch.inference_mode():
        if len(group) == 1:
            return float(ranker(a).item())
        b = torch.as_tensor(vector(group[1], history), device=device)
        relation = torch.as_tensor(relations(*group), device=device)
        ua, ub, pair = ranker(a, b, relation)
    return float(ua + ub + (pair if interaction else 0)) - redundancy * overlap(*group)


def affordable_groups(descriptors):
    return [[d] for d in descriptors] + [
        [a, b] for a, b in combinations(descriptors, 2) if a["owner"] != b["owner"]
    ]
