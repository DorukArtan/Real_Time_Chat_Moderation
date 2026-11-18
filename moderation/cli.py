"""Explicit experiment stages; importing modules or opening the UI never trains."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from moderation.common import (
    ARTIFACTS,
    DATA,
    HOME,
    REPORTS,
    ROOT,
    config,
    file_hash,
    hardware,
    provenance,
    read_json,
    save_json,
)


def status():
    print("Work directory:", HOME)
    print("Training policy: Google Colab only")
    print("Frozen dataset:", "ready" if (DATA / "manifest.json").exists() else "not prepared")
    for name in ("rules", "tfidf", "neural", "transformer", "onnx_fp32", "int8"):
        path = ARTIFACTS / name / "metadata.json"
        state = read_json(path).get("stage_status", "incomplete") if path.exists() else "not trained"
        print(f"  {name:14} {state}")
    print("Measured results:", "available" if (REPORTS / "results.csv").exists() else "not measured")


def doctor():
    import torch

    print(json.dumps(hardware(), indent=2))
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Source:", ROOT)
    print("Work directory:", HOME)


def verify():
    from moderation.common import check_metadata
    from moderation.models import ALL_MODELS

    source = provenance()
    report = read_json(REPORTS / "evaluation.json")
    if any(report.get(key) != value for key, value in source.items()):
        raise RuntimeError("Results belong to a different data/configuration run.")
    for name in ALL_MODELS:
        meta = read_json(ARTIFACTS / name / "metadata.json")
        check_metadata(meta)
        if meta.get("stage_status") != "complete":
            raise RuntimeError(f"{name} is incomplete")
        row = report["models"][name]
        if row["test_rows"] != config()["test_size"] or row["latency_runs"] < 100:
            raise RuntimeError(f"Incomplete measurements for {name}")
    if not read_json(REPORTS / "quantization_checks.json")["integer_operator_counts"]:
        raise RuntimeError("Integer graph evidence is missing")
    checks = {
        "frozen_data_and_provenance": True,
        "six_measured_models": True,
        "same_heldout_test": True,
        "at_least_100_cpu_latency_runs": True,
        "quantization_graph_and_export_parity": True,
        "int8_within_declared_latency_budget": bool(report["int8_within_budget"]),
        "manual_checks_still_required": [
            "Live UI with two new inputs including a difficult case",
            "8-12-slide PDF with technique map, matching architecture and AI usage",
            "Narrated video no longer than 2 minutes",
            "Repository, weights and video links accessible to the instructor",
            "Fresh-environment setup from README",
        ],
    }
    save_json(REPORTS / "technical_checks.json", checks)
    print(json.dumps(checks, indent=2))
    if not report["int8_within_budget"]:
        raise RuntimeError(
            "The INT8 model missed the declared latency budget. Optimize and repeat measurement before submission."
        )


def bundle():
    # A bundle may preserve a run that missed its budget, but that run is never marked complete.
    provenance()
    files = []
    for directory in (DATA, ARTIFACTS, REPORTS):
        files.extend(
            path for path in directory.rglob("*") if path.is_file() and not path.name.endswith(".tmp")
        )
    manifest = {
        "files": {path.relative_to(HOME).as_posix(): file_hash(path) for path in files},
        "config": config(),
        "note": "Contains trained weights and held-out data. Do not commit this archive to Git.",
    }
    output = HOME / "moderation_run_bundle.zip"
    temporary = output.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.relative_to(HOME).as_posix())
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2))
    temporary.replace(output)
    print("Saved", output, flush=True)


def restore(path):
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("bundle_manifest.json"))
        if manifest["config"] != config():
            raise RuntimeError(
                "Use the exact config.json shipped with the Colab source before restoring this run."
            )
        for name, expected_hash in manifest["files"].items():
            relative = Path(name)
            destination = (HOME / relative).resolve()
            if (
                relative.is_absolute()
                or relative.parts[0] not in {"artifacts", "data", "reports"}
                or not destination.is_relative_to(HOME)
            ):
                raise ValueError(f"Unsafe bundle path: {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            digest = hashlib.sha256()
            with archive.open(name) as source, temporary.open("wb") as target:
                while block := source.read(1024 * 1024):
                    digest.update(block)
                    target.write(block)
            if digest.hexdigest() != expected_hash:
                temporary.unlink()
                raise RuntimeError(f"Checksum mismatch for {name}")
            temporary.replace(destination)
    provenance()
    print("Restored run to", HOME)


def main():
    parser = argparse.ArgumentParser(description="SWE015 English moderation experiment")
    parser.add_argument(
        "command",
        choices=[
            "status",
            "doctor",
            "prepare-data",
            "train-baselines",
            "train-neural",
            "train-transformer",
            "quantize",
            "evaluate",
            "verify",
            "bundle",
            "restore-bundle",
        ],
    )
    parser.add_argument("path", nargs="?")
    args = parser.parse_args()
    if args.command == "status":
        status()
    elif args.command == "doctor":
        doctor()
    elif args.command == "prepare-data":
        from moderation.data import prepare

        prepare()
    elif args.command.startswith("train-"):
        from moderation.training import train_baselines, train_neural, train_transformer

        {
            "train-baselines": train_baselines,
            "train-neural": train_neural,
            "train-transformer": train_transformer,
        }[args.command]()
    elif args.command == "quantize":
        from moderation.quantization import quantize

        quantize()
    elif args.command == "evaluate":
        from moderation.evaluation import evaluate

        evaluate()
    elif args.command == "verify":
        verify()
    elif args.command == "bundle":
        bundle()
    elif args.command == "restore-bundle":
        if not args.path:
            parser.error("restore-bundle requires a ZIP path")
        restore(args.path)


if __name__ == "__main__":
    main()
