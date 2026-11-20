"""Build a reviewable, self-contained Colab notebook from this repository's source.

The embedded ZIP is code only: no weights, data, secrets or machine environment.
Run after source changes so the notebook and repository stay identical.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import io
import json
import textwrap
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source_archive():
    paths = [
        ROOT / name
        for name in (
            "app.py",
            "run.cmd",
            "config.json",
            "pyproject.toml",
            "requirements.txt",
            "requirements-test.txt",
            "README.md",
            "AI_USAGE.md",
            ".gitignore",
            ".gitattributes",
        )
    ]
    for directory, pattern in [
        ("moderation", "*.py"),
        ("resources", "*.txt"),
        ("ui", "*.css"),
        ("docs", "*.md"),
        ("tests", "*.py"),
        ("scripts", "*.py"),
        (".vscode", "*.json"),
    ]:
        paths.extend(sorted((ROOT / directory).glob(pattern)))
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            info = zipfile.ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 11, 17, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def build():
    archive = source_archive()
    encoded = base64.b64encode(archive).decode()
    checksum = hashlib.sha256(archive).hexdigest()
    cells = []

    def markdown(source):
        cells.append(
            {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}
        )

    def code(source):
        source = textwrap.dedent(source).strip() + "\n"
        ast.parse(source)
        cells.append(
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source}
        )

    markdown("""
    # SWE015 English Chat Moderation

    **Use Case 4 · all fitting and experiment measurements run on Google Colab.**

    Model ladder: TF-IDF + Logistic Regression → our own Word2Vec + feedforward NN → fine-tuned DistilBERT → dynamic INT8 DistilBERT.
    An edit-distance rule baseline and an FP32 ONNX control are also measured.

    Choose **Runtime > Change runtime type > T4 GPU** (or another available GPU), then run cells in order.
    You must authorize your own Google Drive in step 2. The notebook contains the project source, so no repository URL is needed to start.

    Default split sizes: 20,000 train / 2,000 validation / 3,000 test. This notebook contains no precomputed performance claims.
    Keep the runtime open while working. GPU availability and session duration are subject to Colab limits.
    """)
    markdown("""
    ## 1 Extract the exact project source

    The embedded archive contains readable Python modules, configuration, tests and documentation. It contains no dataset or model weights.
    Extraction is confined to `/content/swe015_english_moderation`. Inspect the extracted files in Colab's Files panel.
    Rebuild this notebook with `scripts/build_colab_notebook.py` whenever repository source changes.
    """)
    code(
        """
    import base64, hashlib, io, json, os, pathlib, subprocess, sys, zipfile
    assert pathlib.Path('/content').is_dir(), 'Open this notebook in a hosted Google Colab runtime.'
    assert sys.version_info >= (3, 10), 'Use a Colab runtime with Python 3.10 or later.'
    PROJECT = pathlib.Path('/content/swe015_english_moderation')
    PROJECT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_B64 = '__ARCHIVE__'
    source_bytes = base64.b64decode(ARCHIVE_B64)
    assert hashlib.sha256(source_bytes).hexdigest() == '__CHECKSUM__'
    with zipfile.ZipFile(io.BytesIO(source_bytes)) as source:
        for member in source.infolist():
            target = (PROJECT / member.filename).resolve()
            assert target.is_relative_to(PROJECT), 'Unsafe archive path'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(member))
    print('Source extracted:', PROJECT)
    print('Source archive SHA-256:', hashlib.sha256(source_bytes).hexdigest())
    print((PROJECT / 'config.json').read_text())
    """.replace("__ARCHIVE__", encoded).replace("__CHECKSUM__", checksum)
    )
    markdown("""
    ## 2 Mount Drive and choose a run

    All persistent files are kept in your own `MyDrive/SWE015_Moderation/runs/RUN_NAME` folder.
    Keep `run_001` to rerun completed stages with identical source and settings. Change the run name after changing code or hyperparameters.
    An interrupted training stage restarts from its beginning; completed stages are reused.
    """)
    code("""
    from google.colab import drive
    drive.mount('/content/drive')
    RUN_NAME = 'run_001'
    assert RUN_NAME and '/' not in RUN_NAME and '\\\\' not in RUN_NAME and RUN_NAME not in {'.', '..'}
    WORK = pathlib.Path('/content/drive/MyDrive/SWE015_Moderation/runs') / RUN_NAME
    WORK.mkdir(parents=True, exist_ok=True)
    saved_source = WORK / 'source_archive.zip'
    if saved_source.exists():
        assert saved_source.read_bytes() == source_bytes, 'Source changed: choose a new RUN_NAME.'
    else:
        saved_source.write_bytes(source_bytes)
    saved_config = WORK / 'config.json'
    if saved_config.exists():
        assert saved_config.read_bytes() == (PROJECT / 'config.json').read_bytes(), 'Configuration changed: choose a new RUN_NAME.'
    else:
        saved_config.write_bytes((PROJECT / 'config.json').read_bytes())
    print('Persistent experiment directory:', WORK)
    """)
    markdown("""
    ## 3 Install an isolated experiment environment

    This virtual environment uses compatible Colab system packages, including CUDA PyTorch, and installs the project's pinned dependencies separately.
    It is created without pip bootstrapping, so it does not depend on the runtime shipping `ensurepip`. The notebook's pip manages this separate interpreter explicitly.
    Commands run as fresh subprocesses, so installing NumPy does not require restarting the notebook kernel.
    """)
    code("""
    import os, pathlib, subprocess, sys
    assert 'PROJECT' in globals() and 'WORK' in globals(), 'Run sections 1 and 2 first.'

    def setup_checked(command):
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Setup command failed: ' + ' '.join(command) + '\\n'
                               + result.stdout + '\\n' + result.stderr)
        if result.stdout.strip():
            print(result.stdout, flush=True)

    def create_experiment_environment(target):
        # Reconfigure even a partially created environment; never clear/delete it.
        setup_checked([sys.executable, '-m', 'venv', '--without-pip',
                       '--system-site-packages', str(target)])
        executable = target / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        setup_checked([sys.executable, '-m', 'pip', '--python', str(executable), '--version'])
        return str(executable)

    VENV = pathlib.Path('/content/swe015_env')
    PYTHON = create_experiment_environment(VENV)
    print('Installing project packages into', VENV, flush=True)
    setup_checked([sys.executable, '-m', 'pip', '--python', PYTHON,
                   'install', '-r', str(PROJECT / 'requirements.txt')])
    ENV = os.environ.copy()
    ENV.update({'MODERATION_WORKDIR': str(WORK), 'HF_HOME': '/content/swe015_hf_cache',
                'PYTHONHASHSEED': '42', 'TOKENIZERS_PARALLELISM': 'false',
                'GRADIO_ANALYTICS_ENABLED': 'false', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})

    def run_stage(name):
        print('Running:', name, flush=True)
        subprocess.run([PYTHON, '-u', '-m', 'moderation', name], cwd=PROJECT, env=ENV, check=True)

    run_stage('doctor')
    subprocess.run([PYTHON, '-c', "import torch; assert torch.cuda.is_available(), 'Select a GPU runtime before Transformer training'"], env=ENV, check=True)
    """)
    markdown("""
    ## 4 Prepare frozen data

    Downloads the labeled Jigsaw training CSV from the documented mirror, pins its resolved commit, converts six labels to a binary target,
    removes normalized duplicates/conflicting labels **before splitting**, and writes immutable stratified subsets and their hashes.
    The original competition test CSV is not used. No model sees our held-out test subset during fitting.
    """)
    code("""
    run_stage('prepare-data')
    manifest = json.loads((WORK / 'data/processed/manifest.json').read_text())
    print(json.dumps(manifest, indent=2))
    """)
    markdown("""
    ## 5 Classical baselines

    Create the fixed lexicon/edit-distance control and fit TF-IDF + Logistic Regression using training only.
    The decision threshold is selected on validation data. This fitting happens on the Colab CPU.
    """)
    code("run_stage('train-baselines')")
    markdown("""
    ## 6 Our Word2Vec and feedforward network

    Train Word2Vec from scratch on training messages. Average its vectors and train our own feedforward network with SGD and backpropagation.
    The best validation epoch is saved; no external pretrained embeddings are substituted.
    """)
    code("run_stage('train-neural')")
    markdown("""
    ## 7 Fine-tune DistilBERT on the Colab GPU

    Load the general pretrained English base model and train its classification head and encoder on our training split.
    This is our own fine-tuning, not use of a ready-made toxicity classifier. DistilBERT is a smaller BERT-family replacement justified by CPU deployment.
    Best validation weights and epoch history are persisted in Drive. An interruption requires rerunning this stage, not retraining completed baselines.
    """)
    code("run_stage('train-transformer')")
    markdown("""
    ## 8 Export and quantize

    Export these exact fine-tuned weights to FP32 ONNX, check parity on validation examples, then apply dynamic per-channel INT8 to eligible MatMul/Gemm operations.
    Other operations may remain FP32. Integer graph nodes are checked. Reuse the FP32 decision threshold for both ONNX versions.
    """)
    code("""
    run_stage('quantize')
    print((WORK / 'reports/quantization_checks.json').read_text())
    """)
    markdown("""
    ## 9 Measure the same held-out test data and CPU latency

    All six models use the same 3,000-row test split. Macro-F1, accuracy, precision/recall and complete inference-artifact size are measured.
    Each model gets the same message order, batch size 1, two CPU threads, 10 warmups and 200 measured calls. CPU execution is explicit even in a GPU runtime.
    The declared target is mean INT8 latency <=150 ms. A missed target is reported, never silently relabeled as success.
    """)
    code("""
    run_stage('evaluate')
    import pandas as pd
    display(pd.read_csv(WORK / 'reports/results.csv'))
    report = json.loads((WORK / 'reports/evaluation.json').read_text())
    print('CPU:', report['environment']['cpu'])
    print('INT8 within declared latency budget:', report['int8_within_budget'])
    """)
    markdown("""
    ## 10 Review failure examples and technical checks

    Inspect real mistakes and explain at least one in the presentation. The verification command fails if technical measurements are incomplete or the CPU budget is missed.
    Passing it does not replace the manual app, presentation, video, repository-link and fresh-environment checks.
    """)
    code("""
    failures = json.loads((WORK / 'reports/failure_examples.json').read_text())
    display(pd.DataFrame(failures))
    run_stage('verify')
    """)
    markdown("""
    ## 11 Run the actual interactive application inside Colab

    The default is **INT8 ONNX CPU**, with live new-input classification, confidence, masking, model switching and per-message timings.
    This embedded app is for interactive notebook work; no public tunnel is opened automatically. The notebook must stay connected.
    Type at least two new messages, including a difficult case. Printed predictions alone are not the homework demo.
    """)
    code("""
    import socket, time
    from google.colab import output
    if 'app_process' in globals() and app_process.poll() is None:
        app_process.terminate()
        app_process.wait(timeout=20)
    app_log_path = WORK / 'app.log'
    app_log = open(app_log_path, 'w')
    app_process = subprocess.Popen([PYTHON, '-u', 'app.py', '--host', '0.0.0.0', '--port', '7860'],
                                   cwd=PROJECT, env=ENV, stdout=app_log, stderr=subprocess.STDOUT)
    for _ in range(60):
        if app_process.poll() is not None:
            raise RuntimeError(app_log_path.read_text())
        try:
            with socket.create_connection(('127.0.0.1', 7860), timeout=1):
                break
        except OSError:
            time.sleep(1)
    else:
        raise RuntimeError('App startup timed out. Inspect ' + str(app_log_path))
    output.serve_kernel_port_as_iframe(7860, height='850')
    """)
    markdown("""
    ## 12 Save the run bundle and submission evidence

    The bundle contains weights, required embeddings/tokenizers, frozen splits and measured reports with checksums.
    It is saved in your Drive run folder. Do not commit it to Git. Share its link with the instructor separately after verifying permissions.
    The source archive and exact config are saved alongside it. The final presentation, narrated video and repository still need to be completed.
    """)
    code("""
    run_stage('bundle')
    run_stage('status')
    print('Run bundle:', WORK / 'moderation_run_bundle.zip')
    print('Results table:', WORK / 'reports/results.csv')
    print('Hardware and raw timings:', WORK / 'reports/evaluation.json')
    print('Actual failure examples:', WORK / 'reports/failure_examples.json')
    print('Submission requirements:', PROJECT / 'docs/requirements.md')
    """)
    markdown("""
    ## References and AI usage

    - Original dataset: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    - Download mirror: https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge
    - Base model: https://huggingface.co/distilbert/distilbert-base-uncased
    - Quantization: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
    - Colab persistence and limits: https://research.google.com/colaboratory/faq.html

    ChatGPT / Codex assisted requirement interpretation, planning, implementation, notebook creation, tests and documentation.
    The student must inspect the code and explain every component. Only our own saved runs may supply quantitative claims.
    If using Colab Gemini or other assistants, record those uses too. Include the final AI Usage slide.
    """)
    for index, cell in enumerate(cells):
        cell["id"] = f"swe015-{index:02d}"
        cell["metadata"]["id"] = f"swe015-{index:02d}"
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {"name": "SWE015 English Chat Moderation.ipynb", "provenance": []},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    output = ROOT / "notebooks" / "01_colab_training.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    (dist / "moderation_source.zip").write_bytes(archive)
    print(f"Built {output} ({len(cells)} cells, source SHA-256 {checksum})")


if __name__ == "__main__":
    build()
