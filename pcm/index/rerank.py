"""Reranker: scores (query, unit) pairs with the project's reranker model.

Usage follows the reranker's model card: a fixed system turn, a user turn
"<Instruct>: ...\\n<Query>: ...\\n<Document>: ...", an assistant header with an
empty thinking block, and the score read from the logits of "yes" and "no" at
the last position. The score returned is logit(yes) - logit(no): it orders pairs
exactly as the card's probability of "yes" does. The instruction and the two
token ids are read from the snapshot's own files.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import torch

from pcm.hub import model_path

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
          "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    def __init__(self, key: str = "reranker", dtype: torch.dtype = torch.bfloat16, device: str = "cuda:0",
                 max_length: int = 8192, batch_size: int = 32):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = model_path(key)
        self.revision = path.name
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype)
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        self.instruction = json.loads((path / "config_sentence_transformers.json").read_text(encoding="utf-8"))["prompts"]["query"]
        self.yes = self.tokenizer.convert_tokens_to_ids("yes")
        self.no = self.tokenizer.convert_tokens_to_ids("no")
        shipped = json.loads((path / "1_LogitScore" / "config.json").read_text(encoding="utf-8"))
        if (self.yes, self.no) != (shipped["true_token_id"], shipped["false_token_id"]):
            raise ValueError("the tokenizer's ids of 'yes' and 'no' differ from the ones the model ships")
        self.prefix = self.tokenizer.encode(PREFIX, add_special_tokens=False)
        self.suffix = self.tokenizer.encode(SUFFIX, add_special_tokens=False)
        self.max_length, self.batch_size = int(max_length), int(batch_size)

    def _pair_ids(self, query: str, doc: str) -> list[int]:
        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {doc}"
        room = self.max_length - len(self.prefix) - len(self.suffix)
        ids = self.tokenizer(body, add_special_tokens=False, truncation=True, max_length=room)["input_ids"]
        return self.prefix + ids + self.suffix

    @torch.no_grad()
    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        """logit(yes) - logit(no) of every (query, doc) pair, float32, in the order given."""
        if not len(docs):
            return np.zeros(0, dtype=np.float32)
        enc = [self._pair_ids(query, d) for d in docs]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))
        out = np.zeros(len(enc), dtype=np.float32)
        device = self.model.device
        pad = self.tokenizer.pad_token_id
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            width = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, width - len(enc[i]):] = torch.tensor(enc[i], dtype=torch.long)
                mask[r, width - len(enc[i]):] = 1
            hidden = self.model.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state[:, -1]
            logits = self.model.lm_head(hidden).float()  # the LM head at the last position only
            out[idx] = (logits[:, self.yes] - logits[:, self.no]).cpu().numpy()
        return out
