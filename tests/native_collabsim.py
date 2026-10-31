"""Real upstream controller + scripted model contract check; no LLM score claim."""

import argparse
import json
from pathlib import Path
import re
import subprocess

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from pairroute.collabsim import run_native
from pairroute.method import Ranker
from pairroute.utils import load, save

UPSTREAM = "08ed0ed1cefb2edf1ea389b20ba8355815a7c8d0"


class FixtureTokenizer:
    """Actual local WordLevel tokenization, with no pretrained tokenizer download."""

    def __init__(self):
        self.inner = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
        self.inner.pre_tokenizer = Whitespace()

    def encode(self, text, add_special_tokens=False):
        return self.inner.encode(text).ids


class ScriptedModel:
    """Supply fixed functional responses, including genuine native probe IDs."""

    def __init__(self):
        self.tokenizer = FixtureTokenizer()
        self.ledger, self.final_reservations, self.prompts = [], [], []
        self.max_total, self.max_generated = 65536, 4096

    def reset(self):
        self.ledger, self.final_reservations, self.prompts = [], [], []

    def tokens(self, text):
        return len(self.tokenizer.encode(text))

    def used(self):
        return (
            sum(row["prompt_tokens"] for row in self.ledger),
            sum(row["generated_tokens"] for row in self.ledger),
        )

    def encode(self, prompt):
        return torch.tensor([self.tokenizer.encode(prompt)])

    def reserve_final_calls(self, reservations):
        self.final_reservations = list(reservations)

    def record(self, prompt, response, stage):
        assert "correct_answer" not in prompt
        self.prompts.append((stage, prompt))
        if stage == "final_vote" and self.final_reservations:
            self.final_reservations.pop(0)
        self.ledger.append(
            dict(
                stage=stage,
                prompt_tokens=self.tokens(prompt),
                generated_tokens=self.tokens(response),
                seconds=0.0,
            )
        )

    def json(self, prompt, stage, limit=512):
        if stage == "candidate":
            facts = json.loads(
                prompt.split("\nPRIVATE=")[1].split("\nREQUESTED_CONDITION=")[0]
            )
            result = {
                "candidates": [
                    dict(
                        span_id=fact["id"],
                        entity="C",
                        predicate="p",
                        option="2",
                        requires="",
                        summary="fact",
                    )
                    for fact in facts
                ]
            }
        elif stage == "owner_upload":
            result = {"text": prompt.split("\nSOURCE=", 1)[1]}
        else:
            result = {
                "vote": 0
            }  # A fixed vote, unrelated to the native evaluator answer.
        self.record(prompt, json.dumps(result), stage)
        return result

    def chat(self, prompt, stage, limit=512):
        identifiers = re.findall(r'probe_id="([^"]+)"', prompt)
        response = json.dumps(
            {
                "responses": [
                    dict(probe_id=identifier, answer="unknown")
                    for identifier in identifiers
                ]
            }
        )
        self.record(prompt, response, stage)
        return response


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", default="outputs/native-collabsim")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, output = Path(args.source_root).resolve(), Path(args.output).resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == UPSTREAM, (
        "Review changes before testing a different upstream revision"
    )
    native = yaml.safe_load(
        (root / "configs/study_conditions/hidden_profile/baseline.yml").read_text(
            encoding="utf-8"
        )
    )
    native["protocol"]["max_concurrent_requests"] = 1
    output.mkdir(parents=True, exist_ok=True)
    native_path = output / "bounded_native.yml"
    native_path.write_text(yaml.safe_dump(native), encoding="utf-8")
    ranker = Ranker()
    with torch.no_grad():
        for parameter in ranker.parameters():
            parameter.zero_()
        ranker.singleton.bias.fill_(1)
    checkpoint = output / "fixed_ranker.pt"
    torch.save({"model": ranker.state_dict()}, checkpoint)
    model = ScriptedModel()
    rows = run_native(
        argparse.Namespace(
            source_root=str(root),
            native_config=str(native_path),
            native_max_steps=8,
            checkpoint=str(checkpoint),
            config="configs/fixed_pool.json",
            model="scripted_functional_not_Qwen",
            revision="local",
            budget=2048,
            seeds=[0],
            method="pairroute",
            device="cpu",
            output=str(output),
            purpose="functional_native_script",
        ),
        model,
    )
    row = rows[0]
    assert row["native_complete"] and row["native_steps"] <= 8
    assert row["native_source_commit"] == UPSTREAM
    assert Path(row["native_controller_file"]) == root / "src/controller/controller.py"
    assert row["initial_votes"] == row["votes"] == [0, 0, 0]
    for phase in ("initial", "final"):
        votes = [
            event
            for event in row["native_events"]
            if event["event_type"] == "hidden_profile_" + phase + "_vote_submitted"
        ]
        assert {event["actor_id"] for event in votes} == {"A", "B", "C"}
        assert all(event["payload"]["choice"] == "Candidate A" for event in votes)
    assert (
        row["communication_tokens"]
        == sum(event["cost"] for event in row["events"])
        <= 2048
    )
    for event in row["events"]:
        assert event["cost"] == model.tokens(event["message"]) * len(
            event["recipients"]
        )
    rejected = [
        event
        for event in row["native_events"]
        if event["event_type"] == "action_rejected"
    ]
    assert not rejected, rejected
    delivered = [
        event
        for event in row["native_events"]
        if event["event_type"] == "message_delivered"
    ]
    assert delivered, "The actual native controller delivered no source evidence"
    assert row["candidate_ids"] and row["delivered"]
    paid = [event for event in row["events"] if event["stage"] == "relay"]
    assert paid and all(len(event["recipients"]) == 2 for event in paid)
    recipient_deliveries = sum(
        len(event["payload"]["recipients"]) for event in delivered
    )
    assert recipient_deliveries == sum(len(event["recipients"]) for event in paid)
    final_prompts = [prompt for stage, prompt in model.prompts if stage == "final_vote"]
    assert len(final_prompts) == 3
    assert all(
        json.loads(prompt.split("\nDISCLOSED=")[1].split("\nReturn")[0])
        for prompt in final_prompts
    )
    probe_calls = sum(stage == "native_probe" for stage, _ in model.prompts)
    assert probe_calls > 0
    result = dict(
        purpose="functional_native_script",
        native_source_commit=UPSTREAM,
        controller_file=row["native_controller_file"],
        native_steps=row["native_steps"],
        native_complete=True,
        participant_count=3,
        paid_relays=len(paid),
        native_deliveries=len(delivered),
        communication_tokens=row["communication_tokens"],
        native_recipient_deliveries=recipient_deliveries,
        prompt_tokens=row["prompt_tokens"],
        generated_tokens=row["generated_tokens"],
        native_probe_calls=probe_calls,
        native_rejections=0,
        cpu_threads=1,
        native_vote_events=6,
        native_event_compatibility=row["native_event_compatibility"],
        fixed_ranker=True,
        scripted_votes=True,
        model_downloads=0,
        benchmark_result=False,
    )
    save(output / "verification.json", result)
    print(json.dumps(result), flush=True)
    assert load(output / "labels.json")[0]["truth"] == 2


if __name__ == "__main__":
    main()
