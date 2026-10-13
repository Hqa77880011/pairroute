import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalModel:
    def __init__(
        self, name, revision="main", seed=17, max_generated=4096, max_total=65536
    ):
        torch.manual_seed(seed)
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=revision, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()
        self.max_generated = max_generated
        self.max_total = max_total
        self.ledger = []
        self.final_reservations = []

    def tokens(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def used(self):
        return sum(r["prompt_tokens"] for r in self.ledger), sum(
            r["generated_tokens"] for r in self.ledger
        )

    def reset(self):
        self.ledger = []
        self.final_reservations = []

    def reserve_final_calls(self, reservations):
        self.final_reservations = list(reservations)

    def encode(self, prompt):
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(self.model.device)

    def chat(self, prompt, stage, limit=512, temperature=0.2):
        ids = self.encode(prompt)
        p, g = self.used()
        if stage in ("final_vote", "single_reader") and self.final_reservations:
            self.final_reservations.pop(0)
        reserved_prompt = sum(p for p, g in self.final_reservations)
        reserved_generated = sum(g for p, g in self.final_reservations)
        allowance = min(
            limit,
            self.max_generated - g - reserved_generated,
            self.max_total
            - p
            - g
            - ids.shape[1]
            - reserved_prompt
            - reserved_generated,
        )
        if allowance < 1:
            raise RuntimeError("Inference token budget exhausted")
        start = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(
                ids,
                max_new_tokens=allowance,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        result = self.tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        self.ledger.append(
            {
                "stage": stage,
                "prompt_tokens": int(ids.shape[1]),
                "generated_tokens": int(out.shape[1] - ids.shape[1]),
                "seconds": time.perf_counter() - start,
            }
        )
        return result

    def json(self, prompt, stage, limit=512):
        answer = self.chat(prompt + "\nReturn a JSON object only.", stage, limit)
        start, end = answer.find("{"), answer.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Model response has no JSON object")
        return json.loads(answer[start : end + 1])

    def option_probs(self, prompt, count, stage="teacher"):
        prefix = self.encode(prompt + "\nAnswer using the option identifier only.")
        scores = []
        p, g = self.used()
        lengths = [
            self.tokenizer.encode(str(i), add_special_tokens=False)
            for i in range(count)
        ]
        needed = sum(prefix.shape[1] + len(t) for t in lengths)
        if (
            p + g + needed + sum(p + g for p, g in self.final_reservations)
            > self.max_total
        ):
            raise RuntimeError("Inference token budget exhausted")
        start = time.perf_counter()
        with torch.inference_mode():
            for suffix in lengths:
                tail = torch.tensor([suffix], device=self.model.device)
                ids = torch.cat([prefix, tail], 1)
                logp = (
                    self.model(ids).logits[0, prefix.shape[1] - 1 : -1].log_softmax(-1)
                )
                scores.append(logp.gather(1, tail[0, :, None]).sum())
        self.ledger.append(
            {
                "stage": stage,
                "prompt_tokens": needed,
                "generated_tokens": 0,
                "seconds": time.perf_counter() - start,
            }
        )
        return torch.stack(scores).softmax(0).cpu().numpy()
