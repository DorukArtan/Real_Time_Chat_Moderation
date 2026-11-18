"""Every fit operation is guarded by the Colab-only training policy."""

from __future__ import annotations

import shutil
import time

import numpy as np
from sklearn.metrics import f1_score

from moderation.common import (
    ARTIFACTS,
    REPORTS,
    ROOT,
    check_metadata,
    config,
    hardware,
    provenance,
    read_json,
    require_cloud_training,
    save_json,
    seed_everything,
)
from moderation.data import load_split


def choose_threshold(labels, scores):
    candidates = np.linspace(0.05, 0.95, 37)
    # Prefer a threshold nearer 0.5 on a tied validation score.
    return float(
        max(
            candidates,
            key=lambda t: (f1_score(labels, scores >= t, average="macro", zero_division=0), -abs(t - 0.5)),
        )
    )


def completed(name):
    path = ARTIFACTS / name / "metadata.json"
    if path.exists():
        meta = read_json(path)
        check_metadata(meta)
        if meta.get("stage_status") != "complete":
            return False
        print(f"{name}: matching completed stage already exists; skipping", flush=True)
        return True
    return False


def metadata(name, threshold, **extra):
    value = {"name": name, "threshold": float(threshold), "stage_status": "complete", **provenance(), **extra}
    save_json(ARTIFACTS / name / "metadata.json", value)


def train_baselines():
    require_cloud_training()
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    cfg = config()
    provenance()
    if not completed("rules"):
        destination = ARTIFACTS / "rules"
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "resources" / "offensive_lexicon.txt", destination / "lexicon.txt")
        metadata("rules", 0.5, precision="deterministic rules", confidence_kind="binary match score")
    if completed("tfidf"):
        return
    train, validation = load_split("train"), load_split("validation")
    start = time.perf_counter()
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2), max_features=cfg["tfidf_max_features"], min_df=2, sublinear_tf=True
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, solver="liblinear", random_state=cfg["seed"]
                ),
            ),
        ]
    )
    model.fit(train.normalized, train.label)
    scores = model.predict_proba(validation.normalized)[:, 1]
    threshold = choose_threshold(validation.label.to_numpy(), scores)
    (ARTIFACTS / "tfidf").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACTS / "tfidf" / "pipeline.joblib")
    metadata(
        "tfidf",
        threshold,
        precision="FP64 sklearn",
        train_seconds=time.perf_counter() - start,
        validation_macro_f1=f1_score(validation.label, scores >= threshold, average="macro"),
    )
    print("TF-IDF fitted on training only; threshold selected on validation only.", flush=True)


def train_neural():
    require_cloud_training()
    import torch
    from gensim.models import Word2Vec
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from moderation.neural import FeedForward, average_vectors
    from moderation.preprocessing import words

    cfg = config()
    provenance()
    if completed("neural"):
        return
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    train, validation = load_split("train"), load_split("validation")
    start = time.perf_counter()
    sentences = [words(text) for text in train.normalized]
    # workers=1 and a fixed PYTHONHASHSEED make this small Word2Vec run reproducible.
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=cfg["word2vec_dim"],
        window=5,
        min_count=2,
        sg=1,
        negative=5,
        epochs=cfg["word2vec_epochs"],
        seed=cfg["seed"],
        workers=1,
    )
    vocabulary = {word: index for index, word in enumerate(w2v.wv.index_to_key)}
    vectors = np.asarray(w2v.wv.vectors, dtype=np.float32)
    x_train = average_vectors(train.normalized.tolist(), vocabulary, vectors)
    x_val = torch.from_numpy(average_vectors(validation.normalized.tolist(), vocabulary, vectors))
    y_train = torch.tensor(train.label.to_numpy(), dtype=torch.long)
    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), y_train),
        batch_size=cfg["neural_batch_size"],
        shuffle=True,
        generator=generator,
    )
    model = FeedForward(cfg["word2vec_dim"], cfg["neural_hidden_dim"])
    counts = np.bincount(train.label, minlength=2)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(train) / (2 * counts), dtype=torch.float32))
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["neural_learning_rate"], momentum=0.9)
    destination = ARTIFACTS / "neural"
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "vectors.npy", vectors, allow_pickle=False)
    save_json(destination / "vocabulary.json", w2v.wv.index_to_key)
    history, best_f1, best_threshold = [], -1, 0.5
    for epoch in range(cfg["neural_epochs"]):
        model.train()
        total_loss = 0.0
        for inputs, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        model.eval()
        with torch.inference_mode():
            scores = model(x_val).softmax(dim=-1)[:, 1].numpy()
        threshold = choose_threshold(validation.label.to_numpy(), scores)
        metric = f1_score(validation.label, scores >= threshold, average="macro", zero_division=0)
        if metric > best_f1:
            best_f1, best_threshold = metric, threshold
            torch.save(model.state_dict(), destination / "network.pt")
        entry = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train),
            "validation_macro_f1": float(metric),
        }
        history.append(entry)
        save_json(REPORTS / "neural_history.json", history)
        print(entry, flush=True)
    metadata(
        "neural",
        best_threshold,
        precision="FP32",
        optimizer="SGD with momentum 0.9",
        word2vec_source="training split only; trained from scratch",
        validation_macro_f1=best_f1,
        train_seconds=time.perf_counter() - start,
    )


def train_transformer():
    require_cloud_training()
    import torch
    from huggingface_hub import HfApi
    from torch import nn
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    cfg = config()
    provenance()
    if completed("transformer"):
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Select Runtime > Change runtime type > GPU in Colab before this stage.")
    seed_everything(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    train, validation = load_split("train"), load_split("validation")
    revision = HfApi().model_info(cfg["transformer_id"], revision=cfg["transformer_revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(cfg["transformer_id"], revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["transformer_id"],
        revision=revision,
        num_labels=2,
        use_safetensors=True,
        id2label={0: "clean", 1: "offensive"},
        label2id={"clean": 0, "offensive": 1},
        attn_implementation="eager",
    ).cuda()

    def examples(frame):
        encoded = tokenizer(frame.normalized.tolist(), truncation=True, max_length=cfg["max_length"])
        return [
            {**{key: value[i] for key, value in encoded.items()}, "labels": int(frame.label.iloc[i])}
            for i in range(len(frame))
        ]

    def collate(batch):
        return tokenizer.pad(batch, padding=True, return_tensors="pt")

    generator = torch.Generator().manual_seed(cfg["seed"])
    loader = DataLoader(
        examples(train),
        batch_size=cfg["transformer_batch_size"],
        shuffle=True,
        collate_fn=collate,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(examples(validation), batch_size=32, collate_fn=collate, num_workers=0)
    counts = np.bincount(train.label, minlength=2)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(len(train) / (2 * counts), dtype=torch.float32, device="cuda")
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["transformer_learning_rate"], weight_decay=0.01)
    total_steps = len(loader) * cfg["transformer_epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)
    scaler = torch.amp.GradScaler("cuda")
    destination = ARTIFACTS / "transformer"
    destination.mkdir(parents=True, exist_ok=True)
    history, best_f1, best_threshold = [], -1, 0.5
    start = time.perf_counter()
    for epoch in range(cfg["transformer_epochs"]):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, 1):
            batch = {key: value.cuda() for key, value in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(model(**batch).logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += float(loss.detach()) * len(labels)
            if step % 100 == 0:
                print(f"Transformer epoch {epoch + 1}, batch {step}/{len(loader)}", flush=True)
        model.eval()
        scores = []
        with torch.inference_mode():
            for batch in val_loader:
                batch.pop("labels")
                scores.extend(
                    model(**{k: v.cuda() for k, v in batch.items()}).logits.softmax(-1)[:, 1].cpu().tolist()
                )
        scores = np.asarray(scores)
        threshold = choose_threshold(validation.label.to_numpy(), scores)
        metric = f1_score(validation.label, scores >= threshold, average="macro", zero_division=0)
        if metric > best_f1:
            best_f1, best_threshold = float(metric), threshold
            model.save_pretrained(destination, safe_serialization=True)
            tokenizer.save_pretrained(destination)
        entry = {
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train),
            "validation_macro_f1": float(metric),
        }
        history.append(entry)
        save_json(REPORTS / "transformer_history.json", history)
        print(entry, flush=True)
    metadata(
        "transformer",
        best_threshold,
        precision="FP32 saved weights; FP16 mixed-precision GPU training",
        base_model=cfg["transformer_id"],
        resolved_revision=revision,
        optimizer="AdamW",
        validation_macro_f1=best_f1,
        train_seconds=time.perf_counter() - start,
        training_gpu=torch.cuda.get_device_name(0),
        environment=hardware(),
    )
