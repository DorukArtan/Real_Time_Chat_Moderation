"""Use only the labeled original training CSV; create our own frozen holdout."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from moderation.common import DATA, HOME, config, digest, file_hash, implementation_hash, save_json
from moderation.preprocessing import normalize

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def clean_source(frame, max_chars=4000):
    required = {"id", "comment_text", *LABELS}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing dataset columns: {required - set(frame.columns)}")
    labels = frame[LABELS]
    if labels.isna().any().any() or not labels.isin([0, 1]).all().all():
        raise ValueError("Every source label must be 0 or 1; unlabeled test rows are not accepted.")
    result = frame[["id", "comment_text"]].copy()
    result["label"] = labels.max(axis=1).astype(int)
    result["normalized"] = result.comment_text.fillna("").map(lambda x: normalize(x, max_chars))
    result = result[result.normalized.str.len() > 0].copy()
    # Group by the exact text seen by the models, before any split or subsampling.
    conflicting = result.groupby("normalized").label.transform("nunique") > 1
    result = result[~conflicting].drop_duplicates("normalized").reset_index(drop=True)
    result["id"] = result.id.astype(str)
    return result


def make_splits(frame, cfg):
    total = sum(cfg[f"{name}_size"] for name in ("train", "validation", "test"))
    if total > len(frame):
        raise ValueError(f"Need {total} unique labeled rows; found {len(frame)}.")
    if total < len(frame):
        frame, _ = train_test_split(frame, train_size=total, stratify=frame.label, random_state=cfg["seed"])
    train_val, test = train_test_split(
        frame, test_size=cfg["test_size"], stratify=frame.label, random_state=cfg["seed"]
    )
    train, validation = train_test_split(
        train_val, test_size=cfg["validation_size"], stratify=train_val.label, random_state=cfg["seed"]
    )
    splits = {"train": train, "validation": validation, "test": test}
    sets = [set(part.normalized) for part in splits.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Normalized text leakage between splits.")
    return {name: part.sort_values("id").reset_index(drop=True) for name, part in splits.items()}


def prepare():
    from huggingface_hub import HfApi, hf_hub_download

    cfg = config()
    manifest_path = DATA / "manifest.json"
    if manifest_path.exists():
        from moderation.common import provenance

        provenance()
        print("Using existing frozen data:", manifest_path)
        return
    revision = HfApi().dataset_info(cfg["dataset_id"], revision=cfg["dataset_revision"]).sha
    path = hf_hub_download(
        cfg["dataset_id"],
        "train.csv",
        repo_type="dataset",
        revision=revision,
        cache_dir=str(HOME / ".cache" / "datasets"),
    )
    raw = pd.read_csv(path, dtype={"id": str})
    clean = clean_source(raw, cfg["max_input_chars"])
    splits = make_splits(clean, cfg)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part.to_csv(DATA / f"{name}.csv", index=False)
    hashes = {name: file_hash(DATA / f"{name}.csv") for name in splits}
    summary = {
        name: {"rows": len(part), "offensive": int(part.label.sum()), "clean": int((part.label == 0).sum())}
        for name, part in splits.items()
    }
    manifest = {
        "source": cfg["dataset_id"],
        "resolved_revision": revision,
        "original_source": "https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge",
        "source_file": "train.csv",
        "source_sha256": file_hash(path),
        "label_rule": "offensive = max(toxic, severe_toxic, obscene, threat, insult, identity_hate)",
        "raw_rows": len(raw),
        "clean_unique_rows": len(clean),
        "splits": summary,
        "split_sha256": hashes,
        "config_hash": digest(cfg),
        "implementation_hash": implementation_hash(),
        "data_fingerprint": digest(hashes),
    }
    save_json(manifest_path, manifest)
    print(summary)


def load_split(name):
    if name not in {"train", "validation", "test"}:
        raise ValueError("Unknown data split")
    return pd.read_csv(DATA / f"{name}.csv", dtype={"id": str}, keep_default_na=False)
