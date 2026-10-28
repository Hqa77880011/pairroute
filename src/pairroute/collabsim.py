import copy
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

from .method import Ledger, Owners, Ranker
from .utils import load, save, seed_all


class NativeAgent:
    def __init__(self, owner, metadata, session):
        self.owner = owner
        self.metadata = metadata
        self.session = session
        self.sent = 0

    def reset(self, seed=None):
        self.sent = 0

    def serialize(self):
        return {"sent": self.sent}

    def load(self, state):
        self.sent = state.get("sent", 0)

    def context_update(self, observation):
        from src.agents.interface import ActionProposal

        session = self.session
        phase = observation.state["task_state"]["phase"]
        if phase == "discussion":
            session.discuss()
            messages = session.messages[self.owner]
            if self.sent < len(messages):
                payload = messages[self.sent]
                self.sent += 1
                return ActionProposal(action={"type": "message", "payload": payload})
            return ActionProposal(action={"type": "do_nothing", "payload": {}})
        stage = "initial_vote" if phase == "initial" else "final_vote"
        disclosed = []
        if phase == "final":
            for event in observation.visible_events:
                if event.get("event_type") == "message_delivered":
                    content = event["payload"]["content"]
                    try:
                        evidence = json.loads(content)
                    except json.JSONDecodeError:
                        evidence = {"source": "native_message", "text": content}
                    if "text" not in evidence:
                        evidence = {"source": "native_message", "text": content}
                    disclosed.append(evidence)
        try:
            vote = session.llm.json(
                session.owners.prompt(self.owner, disclosed)
                + "\nReturn {vote:integer option identifier}.",
                stage,
                32,
            ).get("vote")
        except (ValueError, RuntimeError):
            vote = None
        if type(vote) is not int or not 0 <= vote < len(session.task["options"]):
            vote = None
        message = json.dumps({"owner": self.owner, "vote": vote}, separators=(",", ":"))
        reserved = session.vote_cost * (len(session.names) - self.owner - 1)
        if stage == "initial_vote":
            reserved += session.vote_cost * len(session.names)
        if not session.ledger.send(message, ["coordinator"], stage, reserve=reserved):
            vote = None
        session.votes[stage][self.owner] = vote
        choice = session.task["options"][vote] if vote is not None else "abstain"
        return ActionProposal(
            action={
                "type": "decide",
                "payload": {"decision_id": stage, "choice": choice},
            }
        )

    def respond_probe(self, probe_id, prompt, construct, observation):
        from src.probe.probe import ProbeResponse

        try:
            response = self.session.llm.chat(
                prompt + "\nOBSERVATION=" + json.dumps(observation.state),
                "native_probe",
                64,
            )
        except RuntimeError:
            response = "unknown"
        return ProbeResponse(
            probe_id=probe_id,
            answer=response,
            confidence=None,
            structured_fields=None,
            raw_response=response,
        )


class Session:
    def __init__(self, task, names, llm, ranker, cfg, args, seed):
        self.task, self.names = task, names
        self.llm, self.ranker, self.cfg = llm, ranker, cfg
        self.args, self.seed = args, seed
        self.owners = Owners(task, llm)
        self.ledger = Ledger(llm.tokenizer, args.budget)
        self.vote_cost = max(
            llm.tokens(json.dumps({"owner": i, "vote": v}, separators=(",", ":")))
            for i in range(len(names))
            for v in [None, *range(len(task["options"]))]
        )
        self.votes = {
            phase: [None] * len(names) for phase in ("initial_vote", "final_vote")
        }
        self.messages = [[] for _ in names]
        self.routed = None

    def discuss(self):
        from .cli import route

        if self.routed is not None:
            return
        self.routed = route(
            self.task,
            self.llm,
            self.ranker,
            self.cfg,
            self.args.method,
            self.args.budget - self.ledger.spent,
            self.seed,
            self.args.device,
            reset=False,
            collect_votes=False,
            shuffle_owners=False,
        )
        self.ledger.events.extend(self.routed["events"])
        for event in self.routed["events"]:
            recipients = [i for i in event["recipients"] if isinstance(i, int)]
            if (
                event["stage"] not in ("relay", "direct", "discussion")
                or not recipients
            ):
                continue
            owner = next((i for i in range(len(self.names)) if i not in recipients), 0)
            self.messages[owner].append(
                {
                    "channel": "direct",
                    "content": event["message"],
                    "content_type": "json",
                    "recipients": [self.names[i] for i in recipients],
                }
            )


def run_native(args):
    import yaml

    root = Path(args.source_root).resolve()
    sys.path.insert(0, str(root))
    from src.agents.interface import AgentMetadata
    from src.controller.controller import ExperimentState
    from src.controller.factory import build_controller
    from src.probe.loader import load_probe_templates
    from .llm import LocalModel

    native = yaml.safe_load((root / args.native_config).read_text(encoding="utf-8"))
    cfg = load(args.config)
    names = [agent["id"] for agent in native["agents"]]
    source = native["task"]
    options = list(
        dict.fromkeys(re.findall(r"Candidate [A-Z]", "\n".join(source["shared_facts"])))
    )
    if not options:
        raise ValueError(
            "Native hidden-profile configuration needs named candidate options"
        )
    truth = source.get("phase_rules", {}).get("correct_answer")
    task = {
        "id": native["experiment"]["id"],
        "source": "collabsim",
        "split": "test",
        "public": "Select the best pilot.\n" + "\n".join(source["shared_facts"]),
        "options": options,
        "owners": [
            [
                {"id": str(j), "text": text, "source_id": f"{i}:{j}"}
                for j, text in enumerate(source["private_facts"][name])
            ]
            for i, name in enumerate(names)
        ],
    }
    ranker = Ranker().to(args.device)
    ranker.load_state_dict(
        torch.load(args.checkpoint, map_location=args.device, weights_only=True)[
            "model"
        ]
    )
    ranker.eval()
    llm = LocalModel(
        args.model,
        args.revision,
        max_generated=cfg["generation_cap"],
        max_total=cfg["inference_cap"],
        device=args.device,
    )
    rows = []
    for seed in args.seeds:
        seed_all(seed)
        llm.reset()
        session = Session(task, names, llm, ranker, cfg, args, seed)
        agents = {
            name: NativeAgent(
                i, AgentMetadata(name, "participant", "local", args.model), session
            )
            for i, name in enumerate(names)
        }
        config = copy.deepcopy(native)
        config["experiment"]["seed"] = seed
        config["task"].get("phase_rules", {}).pop("correct_answer", None)
        config["probe"]["questions_path"] = str(
            root / config["probe"]["questions_path"]
        )
        registry = load_probe_templates(str(root / "configs/probe_templates.yml"))
        config["protocol"]["termination"].pop("noop_consecutive_cycles", None)
        state = ExperimentState(
            agents=agents, task_state={}, resources={}, turn_state={}, buffers={}
        )
        controller = build_controller(state, config=config, probe_registry=registry)
        started = time.perf_counter()
        controller.run()
        votes = session.votes["final_vote"]
        counts = Counter(v for v in votes if v is not None)
        majority = next(
            (v for v, count in counts.items() if count > len(names) / 2), None
        )
        p, g = llm.used()
        row = dict(
            session.routed or {},
            id=task["id"],
            source="collabsim",
            method=args.method,
            budget=args.budget,
            seed=seed,
            votes=votes,
            initial_votes=session.votes["initial_vote"],
            majority=majority,
            events=session.ledger.events,
            communication_tokens=session.ledger.spent,
            prompt_tokens=p,
            generated_tokens=g,
            inference=llm.ledger,
            serial_seconds=time.perf_counter() - started,
            native_events=state.event_log,
            native_complete=state.task_state.get("complete", False),
            candidate_ids=(session.routed or {}).get("candidate_ids", []),
            delivered=(session.routed or {}).get("delivered", []),
        )
        rows.append(row)
        save(Path(args.output) / "traces.json", rows)
    if truth in options:
        save(
            Path(args.output) / "labels.json",
            [{"id": task["id"], "truth": options.index(truth)}],
        )
