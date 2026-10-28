"""Small neural checks and explicitly scripted routing-contract fixtures."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import re
from unittest.mock import patch

import numpy as np
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from pairroute.cli import (
    atomic_spans,
    collect_targets,
    evaluate,
    prepare,
    route,
    synthetic,
    train,
)
from pairroute.llm import LocalModel
from pairroute.method import (
    Ledger,
    Owners,
    Ranker,
    group_value,
    relations,
    vector,
)
from pairroute.utils import load, plot_results, save, seed_all


def tiny_model(output):
    """Build random local weights; no download and no benchmark score."""
    words = [
        "[PAD]",
        "[UNK]",
        "[EOS]",
        "[BOS]",
        "0",
        "1",
        "2",
        "p",
        "q",
        "r",
        "yes",
        "x",
    ]
    tokenizer = Tokenizer(
        WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        eos_token="[EOS]",
        bos_token="[BOS]",
    )
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(words),
            n_embd=16,
            n_layer=1,
            n_head=2,
            n_positions=1024,
            bos_token_id=3,
            eos_token_id=2,
            pad_token_id=0,
        )
    ).save_pretrained(output)
    fast.save_pretrained(output)


class ScriptedModel(LocalModel):
    """Deterministic owner replies for protocol tests, never an LLM experiment."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ledger = []
        self.final_reservations = []
        self.max_total = 10**8
        self.max_generated = 10**6
        self.invalid_upload = False
        self.vote_value = 0

    def encode(self, prompt):
        return torch.tensor([self.tokenizer.encode(prompt)])

    def json(self, prompt, stage, limit=512):
        if stage == "candidate":
            facts = json.loads(
                prompt.split("\nPRIVATE=")[1].split("\nREQUESTED_CONDITION=")[0]
            )
            cap = int(re.search(r"at most (\d+)", prompt).group(1))
            result = {
                "candidates": [
                    dict(
                        span_id=f["id"],
                        entity="x",
                        predicate=f["id"],
                        option="0",
                        requires="q"
                        if f["id"] == "p"
                        else "p"
                        if f["id"] == "q"
                        else "",
                        summary="yes",
                    )
                    for f in facts[:cap]
                ]
            }
        elif stage == "owner_upload":
            result = {
                "text": "wrong"
                if self.invalid_upload
                else prompt.split("\nSOURCE=", 1)[1]
            }
        elif stage == "gap":
            result = {"predicate": "q", "option": "0", "owners": [0, 1, 2]}
        else:
            result = {"vote": self.vote_value}
        self.ledger.append(
            dict(
                stage=stage,
                prompt_tokens=self.tokens(prompt),
                generated_tokens=self.tokens(json.dumps(result)),
                seconds=0.0,
            )
        )
        return result

    def option_probs(self, prompt, count, stage="teacher"):
        disclosed = json.loads(prompt.split("\nDISCLOSED=", 1)[1])
        local = json.loads(prompt.split("\nLOCAL=", 1)[1].split("\nDISCLOSED=", 1)[0])
        known = set(disclosed) | {fact["text"] for fact in local}
        probability = 0.9 if {"p", "q"} <= known else 0.5 if {"p", "q"} & known else 0.4
        self.ledger.append(
            dict(
                stage=stage,
                prompt_tokens=self.tokens(prompt),
                generated_tokens=0,
                seconds=0.0,
            )
        )
        return np.array([probability, 1 - probability])


def main():
    torch.set_num_threads(1)
    seed_all(17)
    out = Path("outputs/smoke").resolve()
    out.mkdir(parents=True, exist_ok=True)
    tiny_model(out / "tiny_causal")
    actual = LocalModel(
        str(out / "tiny_causal"), max_generated=8, max_total=512, device="cpu"
    )
    actual.chat("x", "neural_smoke", 2, temperature=0)
    np.testing.assert_allclose(actual.option_probs("x", 2).sum(), 1.0, atol=1e-6)
    assert actual.used()[0] > 0 and actual.used()[1] > 0
    actual.max_generated = actual.used()[1]
    try:
        actual.chat("x", "exhausted", 1)
        raise AssertionError("A spent generation budget must stop")
    except RuntimeError as error:
        assert "budget" in str(error).lower()

    fixture = ScriptedModel(actual.tokenizer)
    task = dict(
        id="training_a",
        source="synthetic_functional",
        split="train",
        public="x",
        options=["yes", "no"],
        owners=[
            [dict(id=p, text=p, source_id=f"{i}:{p}")]
            for i, p in enumerate(["p", "q", "r"])
        ],
    )
    tasks = [task, dict(task, id="validation_b", split="validation")]
    labels = [
        dict(id=t["id"], truth=0, critical_groups=[["0:p", "1:q"]]) for t in tasks
    ]
    save(out / "tasks.json", tasks)
    save(out / "labels.json", labels)
    with patch("pairroute.llm.LocalModel", return_value=fixture):
        collect_targets(
            argparse.Namespace(
                tasks=str(out / "tasks.json"),
                labels=str(out / "labels.json"),
                model="scripted_fixture",
                revision="main",
                device="cpu",
                seed=17,
                output=str(out / "targets.json"),
            )
        )
    targets = load(out / "targets.json")
    assert {r["split"] for r in targets} == {"train", "validation"}
    interaction = next(
        r["target"]
        for r in targets
        if r["kind"] == "pair"
        and r["a"] == vector(Owners(task, fixture).describe(0)[0], []).tolist()
        and r["b"] == vector(Owners(task, fixture).describe(1)[0], []).tolist()
    )
    # Only the third owner lacks both p and q initially; loss averages all owners.
    np.testing.assert_allclose(interaction, np.log(1.44) / 3, atol=1e-6)
    cfg = load("configs/default.json")
    count = sum(r["split"] == "train" for r in targets)
    save(out / "config.json", dict(cfg, epochs=1, batch_size=(count + 1) // 2))
    seed_all(17)
    initial = deepcopy(Ranker().state_dict())
    train(
        argparse.Namespace(
            config=str(out / "config.json"),
            targets=str(out / "targets.json"),
            seed=17,
            device="cpu",
            output=str(out / "ranker"),
        )
    )
    state = torch.load(out / "ranker/ranker.pt", weights_only=True)
    model = Ranker()
    model.load_state_dict(state["model"])
    model.eval()
    clone = Ranker()
    clone.load_state_dict(state["model"])
    clone.eval()
    for name, value in state["model"].items():
        assert torch.isfinite(value).all() and not torch.equal(initial[name], value), (
            name
        )
    a, b = [Owners(task, fixture).describe(i)[0] for i in range(2)]
    xa, xb, relation = (
        torch.from_numpy(vector(a, [])),
        torch.from_numpy(vector(b, [])),
        torch.from_numpy(relations(a, b)),
    )
    with torch.no_grad():
        output = model(xa, xb, relation)
        restored = clone(xa, xb, relation)
        reverse = model(xb, xa, torch.from_numpy(relations(b, a)))
        for x, y in zip(output, restored):
            torch.testing.assert_close(x, y)
        torch.testing.assert_close(output[2], reverse[2])
    model.train()
    model.zero_grad()
    ua, ub, pair = model(xa, xb, relation)
    loss = (ua - 1).square() + (ub - 1).square() + (pair - 1).square()
    loss.backward()
    assert torch.isfinite(loss) and all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    model.eval()

    ledger = Ledger(actual.tokenizer, 100)
    message = '{"fact":"p"}'
    assert ledger.send(message, [0, 1], "relay")
    assert ledger.spent == 2 * fixture.tokens(message)
    assert not ledger.send("p", [0], "upload", reserve=100)
    negative = Ranker().eval()
    with torch.no_grad():
        for parameter in negative.parameters():
            parameter.zero_()
        negative.singleton.bias.fill_(-0.2)
        negative.relation.weight[0, 5] = 2.0
    assert group_value([a, b], [], negative, "cpu") > 0
    assert group_value([a, b], [], negative, "cpu", interaction=False) < 0
    frozen = dict(cfg, fixed_pool=True)
    learned = route(
        task,
        fixture,
        negative,
        frozen,
        "pairroute",
        1024,
        11,
        "cpu",
        shuffle_owners=False,
    )
    zero = route(
        task,
        fixture,
        negative,
        frozen,
        "zero_interaction",
        1024,
        11,
        "cpu",
        shuffle_owners=False,
    )
    assert {"0:p", "1:q"} <= set(learned["delivered"]) and not zero["delivered"]
    assert learned["candidate_ids"] == zero["candidate_ids"]
    assert all(
        e["stage"] not in ("gap_request", "gap_descriptor") for e in learned["events"]
    )
    for result in [learned, zero]:
        assert (
            result["communication_tokens"]
            == sum(e["cost"] for e in result["events"])
            <= 1024
        )
        assert result["initial_votes"] == [0, 0, 0] and result["majority"] == 0
        for event in result["events"]:
            assert event["cost"] == fixture.tokens(event["message"]) * len(
                event["recipients"]
            )
    fixture.invalid_upload = True
    invalid = route(
        task,
        fixture,
        negative,
        frozen,
        "pairroute",
        1024,
        11,
        "cpu",
        shuffle_owners=False,
    )
    assert (
        not invalid["delivered"]
        and invalid["rejected"]
        and any(
            c["stage"] == "owner_upload" and c["generated_tokens"] > 0
            for c in invalid["inference"]
        )
    )
    fixture.invalid_upload = False
    fixture.vote_value = True
    no_vote = route(
        task, fixture, negative, frozen, "zero_interaction", 1024, 11, "cpu"
    )
    assert no_vote["majority"] is None and no_vote["votes"] == [None] * 3
    fixture.vote_value = 0
    gap = route(
        task,
        fixture,
        negative,
        cfg,
        "zero_interaction",
        512,
        11,
        "cpu",
        shuffle_owners=False,
    )
    assert (
        sum(e["cost"] for e in gap["events"] if e["stage"].startswith("gap_"))
        <= 512 // 8
    )
    huge = deepcopy(task)
    for facts in huge["owners"]:
        facts[0]["text"] = "p " * 300
    full_only = route(
        huge,
        fixture,
        negative,
        frozen,
        "pairroute",
        512,
        11,
        "cpu",
        shuffle_owners=False,
    )
    assert not full_only["delivered"] and full_only["communication_tokens"] <= 512
    for cap in [1, 6]:
        expanded = deepcopy(task)
        for facts in expanded["owners"]:
            facts.extend(
                dict(facts[0], id=str(i), source_id=facts[0]["source_id"] + str(i))
                for i in range(5)
            )
        capped = route(
            expanded,
            fixture,
            negative,
            dict(frozen, candidates_per_owner=cap),
            "zero_interaction",
            1024,
            11,
            "cpu",
            shuffle_owners=False,
        )
        assert len(capped["candidate_ids"]) <= 3 * cap
        assert (
            sum(e["cost"] for e in capped["events"] if e["stage"] == "descriptor")
            <= 256
        )

    # One task succeeds on one of two seeds, another on its only seed: task mean=.75.
    rows = [
        dict(learned, id="a", seed=11),
        dict(
            learned,
            id="a",
            seed=23,
            majority=None,
            votes=[0, None, None],
            delivered=["0:p"],
        ),
        dict(learned, id="b", seed=11),
    ]
    gold = {
        name: dict(truth=0, critical_groups=[["0:p", "1:q"]]) for name in ["a", "b"]
    }
    metrics = evaluate(rows, gold)[0]
    assert (
        metrics["success_percent"] == 75.0 and metrics["critical_fact_recall"] == 5 / 6
    )
    assert metrics["complete_group_delivery"] == 2 / 3 and metrics[
        "failure_categories"
    ] == {"incomplete_group_delivery": 1}
    save(out / "metrics.json", {"summary": [metrics]})
    plot_results(
        out / "metrics.json",
        out / "figures",
        ["success_percent", "critical_fact_recall"],
    )
    raw = [
        dict(
            id=1,
            description="x",
            shared_information=["p"],
            possible_answers=["yes", "no"],
            correct_answer="yes",
            hidden_information=["p", "q", "r"],
        )
    ]
    original = "A level of 1.5 is reported. The route is closed."
    spans = atomic_spans(original, 0)
    assert len(spans) == 2 and all(
        f["text"] == original[f["source_start"] : f["source_end"]] for f in spans
    )
    save(out / "raw.json", raw)
    prepare(
        argparse.Namespace(
            source="hiddenbench",
            input=str(out / "raw.json"),
            output=str(out / "prepared"),
        )
    )
    assert (
        load(out / "prepared/tasks.json")[0]["split"] == "test"
        and "truth" not in load(out / "prepared/tasks.json")[0]
    )
    synthetic(
        argparse.Namespace(
            count=20,
            train_count=10,
            seed=2027,
            kind=None,
            output=str(out / "scenarios"),
        )
    )
    generated = load(out / "scenarios/tasks.json")
    assert not (
        {t["semantic_template_family"] for t in generated if t["split"] == "train"}
        & {
            t["semantic_template_family"]
            for t in generated
            if t["split"] == "validation"
        }
    )
    print(
        "PASS: actual tiny causal forward/generation/token limits; scripted target second differences; 2-step Ranker fit/gradients/reload/symmetry; paid routing, frozen pair controls, cap1/6, full-span feasibility, B/8 gap, strict-majority and hand metrics"
    )


if __name__ == "__main__":
    main()
