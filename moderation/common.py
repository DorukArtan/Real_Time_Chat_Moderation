"""Paths, provenance and reproducibility shared by the command line and UI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path(os.environ.get("MODERATION_WORKDIR", ROOT)).resolve()
DATA = HOME / "data" / "processed"
ARTIFACTS = HOME / "artifacts"
REPORTS = HOME / "reports"
os.environ.setdefault("HF_HOME", str(HOME / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "false")


def config():
    return json.loads((ROOT / "config.json").read_text(encoding="utf-8"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def require_cloud_training():
    """The owner's policy: all fitting, including baselines, happens in Colab."""
    try:
        colab_module = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        colab_module = False
    if not (Path("/content").is_dir() and ("COLAB_RELEASE_TAG" in os.environ or colab_module)):
        raise RuntimeError("Training is restricted to Google Colab. Open notebooks/01_colab_training.ipynb.")


def implementation_hash():
    files = sorted((ROOT / "moderation").glob("*.py")) + sorted((ROOT / "resources").glob("*.txt"))
    return digest({str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in files})


def hardware():
    import importlib.metadata

    versions = {}
    for name in [
        "torch",
        "numpy",
        "pandas",
        "scikit-learn",
        "gensim",
        "transformers",
        "onnxruntime",
        "gradio",
    ]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "packages": versions,
    }


def provenance():
    manifest = read_json(DATA / "manifest.json")
    if manifest["config_hash"] != digest(config()):
        raise RuntimeError("config.json changed after data preparation. Use a new Colab RUN_NAME.")
    if manifest["implementation_hash"] != implementation_hash():
        raise RuntimeError("Source code changed after data preparation. Use a new Colab RUN_NAME.")
    for split in ("train", "validation", "test"):
        if file_hash(DATA / f"{split}.csv") != manifest["split_sha256"][split]:
            raise RuntimeError(f"The frozen {split} split was modified. Use a new run directory.")
    return {"data_fingerprint": manifest["data_fingerprint"], "config_hash": manifest["config_hash"]}


def check_metadata(meta):
    expected = provenance()
    if any(meta.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Model artifacts belong to a different data/configuration run.")
