# English Chat Moderation

SWE015 individual homework, Use Case 4. This project compares English chat moderation across TF-IDF, a feedforward network using our own Word2Vec, a fine-tuned DistilBERT, and its INT8 version. A separate edit-distance rule baseline and an FP32 ONNX control are also evaluated.

**Start with the trained model bundle below to run the application on CPU.** Retraining is optional; see [the Colab instructions at the end](#optional-retrain-in-google-colab).

## Run with trained models

### 1. Get the source and download the models

Clone the repository, then enter its folder. If you already have the project, open your existing project folder and skip these two commands.

```powershell
git clone https://github.com/DorukArtan/Real_Time_Chat_Moderation.git
cd Real_Time_Chat_Moderation
```

**Download:** [moderation_run_bundle.zip - Google Drive](https://drive.google.com/file/d/1sybY7Hv8NzawH67hxUHRxOdOxfRfbzfr/view?usp=drive_link) (**710.6 MB**, approximately 711 MB). Save the ZIP intact in your **Downloads** folder with that filename.

The bundle contains all six model artifacts, required tokenizer/vocabulary files, frozen data splits and their manifest, and the original evaluation reports. A new clone needs this bundle before the application can run. Model weights and dataset files are excluded from Git. The current development checkout already has the bundle restored.

Expected SHA-256:

```text
80c7427b94b8a74e72a16fb78765f153d813b4fb054b58f2917f30e62f020bb7
```

### 2. Set up the Python environment

For a new installation, install **64-bit Python 3.12**, then run the following commands from the project folder in PowerShell. If this checkout already has a working `.venv`, use that environment and continue to step 3.

```powershell
py -3.12 --version
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

The version check must succeed before creating the environment. Python 3.12 and CPU PyTorch 2.6.0 match the verified local demo. These commands name the project interpreter directly, so environment activation is unnecessary. Use the same `.venv` in your IDE and terminal.

### 3. Verify and restore the model bundle

Compare the hash printed by the first command with the SHA-256 above, then restore:

```powershell
Get-FileHash "$env:USERPROFILE\Downloads\moderation_run_bundle.zip" -Algorithm SHA256
.\.venv\Scripts\python.exe -m moderation restore-bundle "$env:USERPROFILE\Downloads\moderation_run_bundle.zip"
```

If you saved the ZIP elsewhere, replace its path in both commands. The restore command checks the archived file hashes and experiment configuration. It restores the saved artifacts without training. Skip this step if this checkout already has the trained artifacts restored.

### 4. Start the application

Single run command on Windows:

```powershell
.\run.cmd
```

Open <http://127.0.0.1:7860>. The default model is **DistilBERT INT8 (ONNX CPU)**. Type a message and press **Enter** or click **Send message**. Use **Shift+Enter** for a new line. Empty or whitespace-only input is ignored. The application shows clean/offensive labels and confidence, masks flagged messages, provides four model choices, and logs response time.

The launcher selects this project's `.venv\Scripts\python.exe` and working directory explicitly. To use another port when 7860 is occupied:

```powershell
.\run.cmd --port 7861
```

Then open <http://127.0.0.1:7861>. The equivalent direct launch is `.\.venv\Scripts\python.exe app.py`.

Confidence scores are uncalibrated model scores. Each model classifies only the current message. The UI rejects inputs longer than 4,000 characters, and the Transformer uses at most 128 tokens. Missing model files produce an explicit setup error.

### Use the same environment in an IDE or Codex

- **VS Code:** open the project folder and select `.venv\Scripts\python.exe` with **Python: Select Interpreter**. The checked-in `.vscode` files provide the environment default and an **English Chat Moderation** launch configuration for F5 / Run Without Debugging. Install the recommended Python and Python Debugger extensions if absent. A previously selected workspace interpreter may need to be changed explicitly.
- **Antigravity, PyCharm or another IDE:** select the existing interpreter `<project>\.venv\Scripts\python.exe`, set `app.py` as the script and the project folder as the working directory. You can also run `.\run.cmd` from the IDE's PowerShell terminal.
- **Codex:** use the same `.\run.cmd` launcher or `.\.venv\Scripts\python.exe app.py`.

Setup was verified on Windows from a fresh GitHub clone in an isolated Python 3.12.14 environment. After installing CPU PyTorch and the project requirements, `pip check` passed. The model bundle matched the SHA-256 above, and `.\run.cmd` started the app with INT8 selected. The app displayed a clean message, masked an offensive one, and returned predictions from all three other models.

VS Code configuration follows its [debugger documentation](https://code.visualstudio.com/docs/python/debugging) and [interpreter setting documentation](https://code.visualstudio.com/docs/python/settings-reference). See also the [Python virtual environment documentation](https://docs.python.org/3.12/library/venv.html).

### Linux/macOS paths

Use Python 3.12 to create an environment with `python3.12 -m venv .venv`. Install the project dependencies and an appropriate CPU PyTorch build for your platform. The verified local demo uses Windows; other platforms have not been independently checked.

Replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`, and use your actual ZIP path:

```sh
.venv/bin/python -m moderation restore-bundle /path/to/moderation_run_bundle.zip
.venv/bin/python app.py
```

For the SHA-256 check, use `sha256sum /path/to/moderation_run_bundle.zip` on Linux or `shasum -a 256 /path/to/moderation_run_bundle.zip` on macOS.

## Dataset and task definition

Original source: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge).

Download mirror: [thesofakillers/Jigsaw on Hugging Face](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge). This is a third-party distribution of the original competition data, not a new dataset authored by this project. The mirror lists CC BY-SA 3.0. Retain the source attribution and check the source terms when redistributing data.

We download **only the labeled original `train.csv`**, record its resolved Hub commit and SHA-256, and create our own held-out subsets. We do not use the competition's partially unlabeled `test.csv` as if it were labeled.

The binary target is `offensive = 1` when any of `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, or `identity_hate` is 1; otherwise it is `clean = 0`. This is an explicit project adaptation, not the original six-label task.

Before splitting, empty inputs are removed, texts are normalized, conflicting labels for identical normalized inputs are removed, and remaining normalized duplicates are deduplicated. A seeded, stratified subset contains 20,000 training, 2,000 validation and 3,000 test examples by default. The exact retained IDs, class counts and file hashes are written to `data/processed/manifest.json`.

All learned vocabularies and embeddings are fitted on training only. Validation determines the best epoch and decision threshold. Test data never determines thresholds or training choices. The same Transformer threshold is reused for FP32 ONNX and INT8 so quantization does not receive a separate threshold advantage.

## Model ladder and technique map

| Area | Implementation | Relevant source |
| --- | --- | --- |
| B Classical NLP | Regex noise normalization, leetspeak, repeated letters, word tokenization | `moderation/preprocessing.py` |
| B Classical NLP | Small hand-specified lexicon and minimum Levenshtein distance; auxiliary rule baseline | `moderation/models.py`, `resources/offensive_lexicon.txt` |
| B Classical NLP | TF-IDF word unigrams/bigrams + Logistic Regression; main classical baseline | `moderation/training.py` |
| A Neural networks | Word2Vec trained from scratch on training messages; averaged vectors; our feedforward network trained with SGD and backpropagation | `moderation/neural.py`, `moderation/training.py` |
| C Transformers | Fine-tune English DistilBERT with a two-class head | `moderation/training.py` |
| D Efficient models | ONNX dynamic per-channel INT8 for MatMul/Gemm; inspect integer graph operators | `moderation/quantization.py` |

DistilBERT replaces the suggested BERT with another model in the same Transformer area, as the homework allows when justified. The reason is reduced training and CPU inference cost. We use an existing pretrained DistilBERT and fine-tune it ourselves; we **do not claim to perform knowledge distillation**. Base model: [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased), Apache 2.0.

Quantization is partial: eligible MatMul/Gemm weights are INT8 while other operations, including embeddings where applicable, remain floating point. The UI default is the real quantized ONNX graph on `CPUExecutionProvider`.

## Results and latency protocol

The completed Colab run is recorded in `reports/results.csv` and `reports/evaluation.json`, with one row for each of the four main models and two auxiliary controls. The four main rows are:

| Model | Test macro-F1 | Complete inference size (MB) | Mean CPU latency (ms) |
| --- | ---: | ---: | ---: |
| TF-IDF + Logistic Regression | 0.8489 | 1.89 | 1.15 |
| Own Word2Vec + Feedforward NN | 0.8213 | 10.58 | 0.45 |
| DistilBERT FP32 (PyTorch) | 0.8894 | 268.78 | 47.25 |
| DistilBERT INT8 (ONNX) | 0.8942 | 140.00 | 14.96 |

All use the same 3,000-example held-out test. Benchmark CPU: Intel Xeon @ 2.20 GHz, two inference threads, batch size 1, 10 warmups and 200 measured calls. The FP32 ONNX control measured 33.06 ms: INT8 is **2.21x faster than that control**, isolating quantization from the runtime conversion. The 3.16x comparison with PyTorch includes both effects. INT8 reduces complete inference size by about 47.9%. Both FP32 and INT8 meet the declared 150 ms mean budget. The small network does not beat TF-IDF in this run.

Every row reports macro-F1, accuracy, offensive precision/recall, model size in decimal MB, mean CPU latency and p95 CPU latency. Model size is the complete inference directory including required tokenizer/vocabulary/embedding files. A separate raw latency list, CPU name, thread count, versions, confusion matrices, held-out predictions and actual failure examples are retained.

Latency uses batch size 1, the same seeded message order for every model, 10 unmeasured warmups and 200 measured calls. It includes normalization, tokenization/vectorization, forward pass and score postprocessing. It excludes initial model loading, network requests and browser rendering. The declared budget is **mean CPU latency <=150 ms per message with two threads and maximum Transformer length 128**. The saved INT8 result of 14.96 ms meets it. If a later run misses the target, the verification command fails; disclose any changed experiment configuration.

## Repository and submission

Repository: [DorukArtan/Real_Time_Chat_Moderation](https://github.com/DorukArtan/Real_Time_Chat_Moderation).

Keep weights, raw/processed data, virtual environments and bundles out of Git. The `.gitignore` covers them. The trained bundle is available through the Google Drive link above, with its checksum and restore commands. The source repository and this external bundle together provide the saved experiment and application artifacts.

See `docs/requirements.md` for R1-R8 and grading gates, `docs/architecture.md` for the matching data flow, `docs/presentation_plan.md` for the 12-slide plan, and `AI_USAGE.md` for the current tool-use record.

Cloud training, exported metrics, artifact restoration and local CPU browser verification are complete. See [the demo guide](docs/demo_guide.md) for the tested inputs, actual failure and narration draft. Press **Enter** or click **Send message** to submit; **Shift+Enter** inserts a new line.

The 12-slide presentation links to the repository and the narrated demo on its first slide. The video runs for 1 minute 58.5 seconds and shows live moderation, model switching, the architecture diagram and the benchmark results. The video and trained model bundle were verified to be accessible without signing in. [Watch the demo on Google Drive](https://drive.google.com/file/d/1VztToKFtg89EG3ea7TkQJVKHPfDaw8-1/view?usp=sharing).

## Code checks

Using the same project environment on Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.venv\Scripts\python.exe -m pytest -q
```

These tests verify target mapping, split isolation, noise normalization, local-training restrictions, edit-distance behavior, threshold selection and the >=100-run benchmark contract. They do not train models or manufacture project measurements.

## References

- [Original Jigsaw competition](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- [Download mirror and dataset card](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge)
- [DistilBERT base model](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Hugging Face classification guide](https://huggingface.co/docs/transformers/tasks/sequence_classification)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html)
- [Colab resource limits and persistence](https://research.google.com/colaboratory/faq.html)

## Optional: retrain in Google Colab

This section is for reproducing or changing the experiment. The trained bundle above is sufficient to run the application. All fitting and experiment benchmarking run in Colab; local training is blocked. Opening the application never starts training.

1. Open [Google Colab](https://colab.research.google.com/) and choose **File > Upload notebook**.
2. Upload `notebooks/01_colab_training.ipynb`. It contains a source archive, so a published GitHub repository is not needed to start.
3. Choose **Runtime > Change runtime type > T4 GPU**, or another available GPU.
4. Run the numbered cells in order. Authorize your own Google Drive when the mount cell requests access. For a new experiment, choose an unused `RUN_NAME`, such as `run_002`, in the run-selection cell.
5. Models, frozen splits, training logs and measured reports are stored under `MyDrive/SWE015_Moderation/runs/<RUN_NAME>/`.
6. Run the interactive Gradio application cell to test new messages inside Colab.

The notebook uses an isolated virtual environment and reuses Colab's CUDA PyTorch when compatible. GPU training is used for the Transformer; the Word2Vec and small network stages use the Colab CPU. **All inference benchmarks explicitly use CPU**, including when the Colab runtime has a GPU.

The environment is created with `venv --without-pip --system-site-packages`, then the notebook's pip uses `--python` to install into that environment. This avoids depending on `ensurepip`, which some hosted Python installations omit. Creation is repeated safely on rerun to repair a partially created environment; existing packages and Drive outputs are not deleted. Setup failures include their underlying stdout/stderr in the error message. See [pip's interpreter-management documentation](https://pip.pypa.io/en/stable/topics/python-option/).

Completed model stages are skipped on rerun. An interrupted stage restarts from its beginning; this is not an optimizer-state resume system. Best epoch weights and per-epoch logs are saved to Drive, but do not reuse an incomplete stage as final evidence. If source or configuration changes, choose a new `RUN_NAME` to avoid mixing experiments.

Colab GPU availability and session duration are not guaranteed. No paid resource is purchased by this project.

### Experiment stages

The notebook executes these same repository commands:

```text
python -m moderation doctor
python -m moderation prepare-data
python -m moderation train-baselines
python -m moderation train-neural
python -m moderation train-transformer
python -m moderation quantize
python -m moderation evaluate
python -m moderation verify
python -m moderation bundle
```

Training commands are restricted to Colab. Data preparation, status inspection and inference do not train models.
