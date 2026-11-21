"""Unit and query embeddings with the project's embedder (specification, section 4.3).

Usage follows the embedder's model card: documents are embedded as they are,
queries behind the model's shipped query instruction ("Instruct: ...\\nQuery:"),
the tokenizer pads on the left, the embedding is the hidden state of the last
token, and vectors are normalised. The instruction is read from the snapshot's
config_sentence_transformers.json, not typed here.
"""
from __future__ import annotations

import json
from typing import Sequence

import numpy as np
import torch

from pcm.hub import model_path


class Embedder:
    def __init__(self, key: str = "embedder", dtype: torch.dtype = torch.bfloat16, device: str = "cuda:0",
                 max_length: int = 8192, batch_size: int = 128):
        from transformers import AutoModel, AutoTokenizer

        path = model_path(key)
        self.revision = path.name
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")
        try:
            self.model = AutoModel.from_pretrained(path, dtype=dtype)
        except TypeError:  # older transformers name for the same argument
            self.model = AutoModel.from_pretrained(path, torch_dtype=dtype)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        prompts = json.loads((path / "config_sentence_transformers.json").read_text(encoding="utf-8"))["prompts"]
        if prompts.get("document"):
            raise ValueError("the embedder ships a document instruction; documents are embedded without one here")
        self.query_prompt = prompts["query"]
        self.max_length, self.batch_size = int(max_length), int(batch_size)
        self.dim = int(self.model.config.hidden_size)
        self.dtype_name = str(dtype).replace("torch.", "")

    @property
    def tag(self) -> str:
        """Identifies the vectors this embedder writes: model revision and number format."""
        return embedder_tag(self.revision, self.dtype_name)

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Normalised embeddings, float32 [len(texts), dim], in the order given."""
        if not len(texts):
            return np.zeros((0, self.dim), dtype=np.float32)
        enc = self.tokenizer(list(texts), padding=False, truncation=True, max_length=self.max_length)["input_ids"]
        order = sorted(range(len(enc)), key=lambda i: len(enc[i]))  # similar lengths together: little padding
        out = np.zeros((len(enc), self.dim), dtype=np.float32)
        device = self.model.device
        pad = self.tokenizer.pad_token_id
        for s in range(0, len(order), self.batch_size):
            idx = order[s:s + self.batch_size]
            width = max(len(enc[i]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), width), dtype=torch.long)
            for r, i in enumerate(idx):  # left padding: the last column is every sequence's last token
                ids[r, width - len(enc[i]):] = torch.tensor(enc[i], dtype=torch.long)
                mask[r, width - len(enc[i]):] = 1
            hidden = self.model(input_ids=ids.to(device), attention_mask=mask.to(device)).last_hidden_state[:, -1]
            vec = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
            out[idx] = vec.cpu().numpy()
        return out

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts)

    def encode_query(self, query: str) -> np.ndarray:
        """One query behind the model's query instruction: float32 [dim]."""
        return self.encode([self.query_prompt + query])[0]


def embedder_tag(revision: str, dtype_name: str = "bfloat16") -> str:
    """The cache tag of an embedder (Embedder.tag) without loading it: model revision and number format."""
    return f"emb-{revision[:12]}-{dtype_name}"
