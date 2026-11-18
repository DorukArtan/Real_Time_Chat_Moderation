"""Inference only. No fit or training is invoked by any predictor."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from moderation.common import ARTIFACTS, ROOT, check_metadata, config, read_json
from moderation.preprocessing import normalize, words

MAIN_MODELS = ["tfidf", "neural", "transformer", "int8"]
ALL_MODELS = ["rules", *MAIN_MODELS, "onnx_fp32"]
DISPLAY = {
    "rules": "Edit-distance rules",
    "tfidf": "TF-IDF + Logistic Regression",
    "neural": "Own Word2Vec + Feedforward NN",
    "transformer": "DistilBERT FP32",
    "int8": "DistilBERT INT8 (ONNX CPU)",
    "onnx_fp32": "DistilBERT FP32 (ONNX CPU)",
}


def lexicon_from(path):
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


class RuleMatcher:
    def __init__(self, lexicon=None):
        self.lexicon = lexicon or lexicon_from(ROOT / "resources" / "offensive_lexicon.txt")

    def score(self, normalized):
        for token in words(normalized):
            for term in self.lexicon:
                if token == term:
                    return 1.0
                if len(term) >= 5 and len(token) >= 5 and abs(len(token) - len(term)) <= 1:
                    if Levenshtein.distance(token, term, score_cutoff=1) <= 1:
                        return 1.0
        return 0.0


class Predictor:
    def __init__(self, name, verify=True, allow_pending=False):
        if name not in ALL_MODELS:
            raise ValueError(f"Unknown model: {name}")
        self.name = name
        self.directory = ARTIFACTS / name
        self.meta = read_json(self.directory / "metadata.json")
        if verify:
            check_metadata(self.meta)
        if self.meta.get("stage_status") != "complete" and not allow_pending:
            raise RuntimeError(f"{name}: this artifact stage has not passed its checks.")
        self.cfg = config()
        self.threshold = self.meta["threshold"]
        if name == "rules":
            self.model = RuleMatcher(lexicon_from(self.directory / "lexicon.txt"))
        elif name == "tfidf":
            import joblib

            self.model = joblib.load(self.directory / "pipeline.joblib")
        elif name == "neural":
            import torch

            from moderation.neural import FeedForward

            torch.set_num_threads(self.cfg["cpu_threads"])
            self.vectors = np.load(self.directory / "vectors.npy", allow_pickle=False)
            self.vocabulary = {
                word: index for index, word in enumerate(read_json(self.directory / "vocabulary.json"))
            }
            self.model = FeedForward(self.vectors.shape[1], self.cfg["neural_hidden_dim"])
            self.model.load_state_dict(
                torch.load(self.directory / "network.pt", map_location="cpu", weights_only=True)
            )
            self.model.eval()
        elif name == "transformer":
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.set_num_threads(self.cfg["cpu_threads"])
            self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True)
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    self.directory, local_files_only=True, attn_implementation="eager"
                )
                .float()
                .cpu()
                .eval()
            )
        else:
            import onnxruntime as ort
            from transformers import AutoTokenizer

            options = ort.SessionOptions()
            options.intra_op_num_threads = self.cfg["cpu_threads"]
            options.inter_op_num_threads = 1
            self.model = ort.InferenceSession(
                str(self.directory / "model.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.directory, local_files_only=True)

    def scores(self, texts):
        normalized = [normalize(text, self.cfg["max_input_chars"]) for text in texts]
        if self.name == "rules":
            return np.asarray([self.model.score(text) for text in normalized])
        if self.name == "tfidf":
            return self.model.predict_proba(normalized)[:, 1]
        if self.name == "neural":
            import torch

            from moderation.neural import average_vectors

            inputs = average_vectors(normalized, self.vocabulary, self.vectors)
            with torch.inference_mode():
                return self.model(torch.from_numpy(inputs)).softmax(dim=-1)[:, 1].numpy()
        if self.name == "transformer":
            import torch

            inputs = self.tokenizer(
                normalized,
                padding=True,
                truncation=True,
                max_length=self.cfg["max_length"],
                return_tensors="pt",
            )
            with torch.inference_mode():
                return self.model(**inputs).logits.softmax(dim=-1)[:, 1].numpy()
        inputs = self.tokenizer(
            normalized, padding=True, truncation=True, max_length=self.cfg["max_length"], return_tensors="np"
        )
        inputs = {
            item.name: np.asarray(inputs[item.name], dtype=np.int64) for item in self.model.get_inputs()
        }
        logits = self.model.run(["logits"], inputs)[0]
        logits = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits)
        return (probs / probs.sum(axis=-1, keepdims=True))[:, 1]

    def classify(self, text):
        score = float(self.scores([text])[0])
        offensive = score >= self.threshold
        return {
            "label": "offensive" if offensive else "clean",
            "offensive_score": score,
            "confidence": score if offensive else 1 - score,
            "threshold": self.threshold,
        }
