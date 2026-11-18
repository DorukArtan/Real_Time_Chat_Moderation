# Live demo and narration guide

Verified on 2026-11-17 with the actual restored Colab artifacts on Windows CPU. Training remains in Colab. Formal results remain the original Colab CPU measurements in `reports/results.csv`; the few live timings below are separate observations.

## Open and rehearse

From the project folder, run `.\.venv\Scripts\python.exe app.py` and open <http://127.0.0.1:7860>. If the server is already running, use its existing tab. **DistilBERT INT8 (ONNX CPU)** is the default. Click **Send message** after typing. In the tested multiline textbox, Enter inserted a newline despite the placeholder.

Use **Clear conversation** before recording. Type the first two examples live, rather than only clicking the built-in examples or showing a screenshot. Show the selected model and the response log. Four model choices were tested successfully: TF-IDF, own Word2Vec + FFN, DistilBERT FP32 and INT8.

| Input | Observed INT8 result | What to show |
| --- | --- | --- |
| `Thanks for the match. Let's play again tomorrow.` | Clean, 99.9% displayed confidence | Message remains visible |
| `You are a useless idiot. Get lost.` | Offensive, 99.9% displayed confidence | Whole message masked |
| `Y0u are a c0mplete 1d10t.` | Offensive, 99.9% displayed confidence | Noise normalization and masking |

These are live smoke checks, not a new accuracy test. Displayed scores and timings can vary between runtimes. Confidence is uncalibrated, and the classifier sees only the current message.

## Actual failure to discuss

Held-out test ID: `b79f828bb11b371f`.

> Thats what yopur mom said last night oooh

The retained dataset label is **offensive**. The frozen Colab INT8 prediction is **clean**, with offensive score **0.018513653**. The local single-message application also predicted **clean**, displaying **97.1% clean confidence** (table value 0.9713), and left the message visible. See `reports/live_validation/selected_failure.json` and `int8_failure_example.jpg`.

This is a false negative relative to the dataset annotation. An interpretation is that indirect innuendo, spelling noise and missing context make the example difficult; this is a hypothesis, not an established causal explanation. It also shows why a high confidence score does not establish correctness. The live score and frozen batch-evaluation score are separate measurements: dynamic quantization can depend on input batching, and these runs also used different environments. Preserve the original metrics rather than replacing them with the live score.

## Narration draft, target about 1:45-1:55

Rehearse with a stopwatch and trim if typing or switching slides takes longer. This text has not been recorded or timed by the student.

**0:00-0:12 — introduction**

“This is an English chat moderation application. It classifies each message as clean or offensive, hides flagged messages, and shows confidence and response time.”

**0:12-0:55 — type two live inputs**

“The active model is our quantized DistilBERT running on CPU. I will type a new friendly message. It stays visible. Now I will type an insult using numbers in place of letters. The message is flagged and masked. The side panel records the model, label, confidence and latency. The selector also provides our three other main models.”

**0:55-1:12 — architecture slide**

“We normalize noisy text, then run the selected classifier and apply its validation threshold. All fitting took place in Colab. Our four stages are TF-IDF, a feedforward network using our own Word2Vec, fine-tuned DistilBERT, and its partially quantized INT8 ONNX version.”

**1:12-1:35 — results slide**

“All models share the same three-thousand-message test set. INT8 achieved macro-F1 0.8942, a complete inference size of 140 megabytes, and mean CPU latency 14.96 milliseconds over 200 calls. It was 2.21 times faster than the FP32 ONNX control. Both versions met our 150-millisecond budget.”

**1:35-1:55 — conclusion/failure slide**

“The small neural network did not beat TF-IDF. Our selected failure is an indirect offensive remark that INT8 labeled clean with high confidence. Context, domain shift from Wikipedia comments to chat, and uncalibrated confidence remain limitations. The system supports moderation decisions, but this evaluation does not establish reliable performance for every real chat.”

## Submission checks still pending

The recording must include the student's own narration, live new input, architecture and measured results, and last at most two minutes. It must accompany an 8-12-slide PDF and accessible repository/weights/video links. This guide and the saved screenshots do not complete those deliverables.
