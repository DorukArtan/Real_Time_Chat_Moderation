"""Measured test metrics and CPU latency; never substitute model-card numbers."""

from __future__ import annotations

import gc
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from threadpoolctl import threadpool_limits

from moderation.common import (
    ARTIFACTS,
    REPORTS,
    config,
    hardware,
    provenance,
    require_cloud_training,
    save_json,
)
from moderation.data import load_split


def benchmark(predictor, messages, runs, warmup):
    if runs < 100:
        raise ValueError("Homework latency measurements require at least 100 runs.")
    for i in range(warmup):
        predictor.scores([messages[i % len(messages)]])
    samples = []
    for i in range(runs):
        start = time.perf_counter_ns()
        predictor.scores([messages[i % len(messages)]])
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "latency_mean_ms": float(np.mean(samples)),
        "latency_p50_ms": float(np.median(samples)),
        "latency_p95_ms": float(np.percentile(samples, 95)),
        "latency_runs": runs,
        "warmup_runs": warmup,
        "raw_latency_ms": samples,
    }


def evaluate():
    require_cloud_training()
    from moderation.models import ALL_MODELS, DISPLAY, MAIN_MODELS, Predictor

    cfg = config()
    source = provenance()
    test = load_split("test")
    y = test.label.to_numpy()
    # Same deterministic, mixed-class message order and batch size for every CPU benchmark.
    rng = np.random.default_rng(cfg["seed"])
    selected = rng.choice(len(test), size=min(cfg["latency_runs"], len(test)), replace=False)
    messages = test.comment_text.iloc[selected].tolist()
    results, details, failures = [], {}, []
    prediction_table = test[["id", "label"]].copy()
    for name in ALL_MODELS:
        print(f"Evaluating {name} on {len(test)} held-out messages (CPU)...", flush=True)
        predictor = Predictor(name)
        with threadpool_limits(limits=cfg["cpu_threads"]):
            scores = np.concatenate(
                [
                    predictor.scores(test.comment_text.iloc[i : i + 32].tolist())
                    for i in range(0, len(test), 32)
                ]
            )
            predictions = (scores >= predictor.threshold).astype(int)
            timing = benchmark(predictor, messages, cfg["latency_runs"], cfg["latency_warmup"])
        size_bytes = sum(path.stat().st_size for path in (ARTIFACTS / name).rglob("*") if path.is_file())
        row = {
            "model": name,
            "display_name": DISPLAY[name],
            "main_ladder": name in MAIN_MODELS,
            "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y, predictions)),
            "offensive_precision": float(precision_score(y, predictions, zero_division=0)),
            "offensive_recall": float(recall_score(y, predictions, zero_division=0)),
            "model_size_mb": size_bytes / 1_000_000,
            "latency_mean_ms": timing["latency_mean_ms"],
            "latency_p95_ms": timing["latency_p95_ms"],
            "threshold": predictor.threshold,
            "test_rows": len(test),
            "precision": predictor.meta["precision"],
        }
        results.append(row)
        details[name] = {
            **row,
            **timing,
            "confusion_matrix_true_rows_pred_columns": confusion_matrix(
                y, predictions, labels=[0, 1]
            ).tolist(),
        }
        prediction_table[f"{name}_prediction"] = predictions
        prediction_table[f"{name}_offensive_score"] = scores
        for i in np.flatnonzero(y != predictions)[:5]:
            failures.append(
                {
                    "model": name,
                    "id": test.id.iloc[i],
                    "text": test.comment_text.iloc[i],
                    "true_label": int(y[i]),
                    "predicted_label": int(predictions[i]),
                    "offensive_score": float(scores[i]),
                }
            )
        print(row, flush=True)
        del predictor
        gc.collect()
    REPORTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(REPORTS / "results.csv", index=False)
    prediction_table.to_csv(REPORTS / "test_predictions.csv", index=False)
    save_json(REPORTS / "failure_examples.json", failures)
    int8_row = next(row for row in results if row["model"] == "int8")
    save_json(
        REPORTS / "evaluation.json",
        {
            **source,
            "environment": hardware(),
            "cpu_threads": cfg["cpu_threads"],
            "batch_size_for_latency": 1,
            "latency_includes": "normalization, tokenization/vectorization, model forward, score postprocessing",
            "latency_excludes": "initial model loading, network, browser rendering",
            "latency_message_ids": test.id.iloc[selected].tolist(),
            "latency_budget_ms": cfg["latency_budget_ms"],
            "latency_budget_definition": "Mean CPU latency, batch size 1, fixed thread count, same held-out inputs",
            "int8_within_budget": int8_row["latency_mean_ms"] <= cfg["latency_budget_ms"],
            "models": details,
        },
    )
    print("Saved genuine measurements to", REPORTS, flush=True)
