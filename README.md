# English Chat Moderation

SWE015 individual homework, Use Case 4. This project compares an English chat moderation task across TF-IDF, a feedforward network using our own Word2Vec, a fine-tuned DistilBERT, and its INT8 version. A separate edit-distance rule baseline and an FP32 ONNX control are also evaluated.

All model fitting and experiment benchmarking run in **Google Colab**. Local training is deliberately blocked. Opening the application never starts training.

## Start in Colab

1. Open [Google Colab](https://colab.research.google.com/) and choose **File > Upload notebook**.
2. Upload `notebooks/01_colab_training.ipynb`. It contains a source archive, so a published GitHub repository is not needed to start.
3. Choose **Runtime > Change runtime type > T4 GPU**, or another available GPU.
4. Run the numbered cells in order. Authorize your own Google Drive when the mount cell requests access.
5. Models, frozen splits, training logs and measured reports are stored under `MyDrive/SWE015_Moderation/runs/run_001/`.
6. Run the interactive Gradio application cell to test new messages inside Colab.

The notebook uses an isolated virtual environment and reuses Colab's CUDA PyTorch when compatible. GPU training is used for the Transformer; the Word2Vec and small network stages use the Colab CPU. **All inference benchmarks explicitly use CPU**, including when the Colab runtime has a GPU.

The environment is created with `venv --without-pip --system-site-packages`, then the notebook's pip uses `--python` to install into that environment. This avoids depending on `ensurepip`, which some hosted Python installations omit. Creation is repeated safely on rerun to repair a partially created environment; existing packages and Drive outputs are not deleted. Setup failures include their underlying stdout/stderr in the error message. See [pip's interpreter-management documentation](https://pip.pypa.io/en/stable/topics/python-option/).

Completed model stages are skipped on rerun. An interrupted stage restarts from its beginning; this is not an optimizer-state resume system. Best epoch weights and per-epoch logs are saved to Drive, but do not reuse an incomplete stage as final evidence. If source or configuration changes, choose a new `RUN_NAME` to avoid mixing experiments.

Colab GPU availability and session duration are not guaranteed. No paid resource is purchased by this project.

## Experiment stages

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

## Run the application after cloud training

**Trained model download: an instructor-accessible link is still pending.** A new clone needs the separately exported `moderation_run_bundle.zip` before the application can run. The current development computer already has it restored. Model weights and dataset files are deliberately excluded from Git.

The easiest first demo is the final app cell in Colab, embedded in the notebook. No public tunnel is opened automatically. For a local CPU demonstration after training:

### Existing project: the same environment in an IDE and Codex

The current `.venv` already has the packages and the trained artifacts have been restored. Both the IDE and Codex can use this same environment. Its base Python is supplied with Codex, but running the application does not require launching it through Codex's chat.

From a PowerShell terminal in this project, run:

```powershell
.\run.cmd
```

The launcher selects this project's `.venv\Scripts\python.exe` and project working directory explicitly, regardless of the terminal's default Python. It forwards application arguments, for example `.\run.cmd --port 7861` for a second instance while port 7860 is already occupied.

- **VS Code:** open the `LLM_HW1` folder. The checked-in `.vscode` files provide the environment default and an **English Chat Moderation** launch configuration for F5 / Run Without Debugging. Install the recommended Python and Python Debugger extensions if absent. If the workspace already selected a different interpreter, use **Python: Select Interpreter** to choose `.venv\Scripts\python.exe`; VS Code does not replace a previous selection merely because the default setting changes. The launch configuration also names the project interpreter explicitly.
- **PyCharm or another Python IDE:** select the existing interpreter `C:\Users\USER\Desktop\LLM_HW1\.venv\Scripts\python.exe`, set `app.py` as the script and `C:\Users\USER\Desktop\LLM_HW1` as the working directory. No new virtual environment is needed for this project on this computer.
- **Codex:** use the same launcher or `.\.venv\Scripts\python.exe app.py`.

VS Code configuration follows its [Python debugger documentation](https://code.visualstudio.com/docs/python/debugging) and [interpreter setting documentation](https://code.visualstudio.com/docs/python/settings-reference).

### New checkout or another computer

1. For a new installation, install an independent **64-bit Python 3.12** first. Verify it with `py -3.12 --version`; this command must succeed before creating the environment. The existing system Python 3.8 is not suitable. Python 3.12 is the version used for the verified local CPU demo; the full supported-version range has not been independently tested.
2. In the project folder, create a new virtual environment and install CPU PyTorch and the remaining dependencies. These commands are for a new checkout without an existing `.venv`; use the existing working environment directly rather than recreating it over itself. On Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

This pins the CPU PyTorch version used by the local demo; Colab keeps its separate CUDA-enabled PyTorch. No activation is required below because every command names the environment's interpreter explicitly. Select that same `.venv/Scripts/python.exe` in the IDE.

3. Download `moderation_run_bundle.zip` from your run folder in Drive. Preserve the exact source and `config.json` used to produce it:

```powershell
.\.venv\Scripts\python.exe -m moderation restore-bundle "PATH_TO_DOWNLOADED_ZIP"
```

4. Single run command:

```powershell
.\.venv\Scripts\python.exe app.py
```

Virtual environments depend on their base Python installation and are not portable folders. Do not copy or commit `.venv`; recreate it on the target computer. The current development `.venv` uses Codex's bundled Python 3.12.14 and passes imports, `pip check` and real CPU inference. It is suitable for the shared IDE/Codex workflow above while that base installation remains available. A fresh installation on another computer is still pending. See the [Python venv documentation](https://docs.python.org/3.12/library/venv.html).

Open `http://127.0.0.1:7860`. The application accepts new messages, shows labels and confidence, masks messages flagged offensive, switches between four main models and logs each response time. Confidence scores are not calibrated probabilities. Inputs longer than 4,000 characters are rejected in the UI; training preprocessing clips to that length and the Transformer truncates to 128 tokens. Each model classifies the current message without conversation history.

Missing model files produce an explicit setup error; there is no fake prediction fallback. Only restore artifact bundles you produced and trust, because the sklearn pipeline uses Python serialization.

For Linux/macOS, use an independently installed Python 3.12 with `python3.12 -m venv .venv`, then replace `.\.venv\Scripts\python.exe` in the commands above with `.venv/bin/python`.

## Repository and submission

Repository: [DorukArtan/Real_Time_Chat_Moderation](https://github.com/DorukArtan/Real_Time_Chat_Moderation).

Keep weights, raw/processed data, virtual environments and bundles out of Git. The `.gitignore` covers them. An instructor-accessible weight-bundle link from Drive or Hugging Face is still pending; the restore commands are documented above. Source code alone is not a complete submission.

See `docs/requirements.md` for R1-R8 and grading gates, `docs/architecture.md` for the matching data flow, `docs/presentation_plan.md` for the 12-slide plan, and `AI_USAGE.md` for the current tool-use record.

Cloud training, exported metrics, artifact restoration and local CPU browser verification are complete. See [the demo guide](docs/demo_guide.md) for the tested inputs, actual failure and narration draft. Click **Send message** to submit: Enter inserted a newline in the tested multiline textbox despite its placeholder.

Pending deliverables: the final 8-12-slide PDF, the student's narrated <=2-minute video, accessible weights/video links, and a fresh-environment setup check. Successful local inference does not complete these submission gates.

## Code checks

```text
python -m pip install -r requirements-test.txt
python -m pytest -q
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

