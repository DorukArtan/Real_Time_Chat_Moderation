# Homework requirements and evidence

This is a traceability plan. An implementation location is not proof that an unexecuted stage passed.

| Requirement | Planned evidence | Current evidence source |
| --- | --- | --- |
| R1 Working application | New typed input, label, confidence, masking, timing panel and model switch | Passed local CPU browser check on 2026-11-17; `reports/live_validation/` |
| R2 Four technique areas | A/B/C/D slide with actual component names | README map; final slide pending |
| R3 Model ladder | TF-IDF, own Word2Vec + NN, DistilBERT FP32, quantized DistilBERT on identical frozen holdout | Colab run complete; restored split IDs and normalized texts are disjoint; predictions match the 3,000-row test |
| R4 Results table | Macro-F1, complete artifact MB, mean latency >=100 calls, hardware | Actual `reports/results.csv` and `evaluation.json`: six models, 200 CPU calls each, hardware recorded |
| R5 Quantized Transformer in app | INT8 is selected by default; explicit CPU ONNX inference; missing weights fail | Real trained INT8 default, masking and new-input inference verified in browser; `reports/live_validation/README.md` and screenshots |
| R6 Matching architecture | Names, precision, preprocessing, postprocessing, hosting and user interface | `docs/architecture.md` matches CPU app/model switching; final slide pending |
| R7 Repository | Accessible GitHub/GitLab, README, requirements, single app command, weights linked externally | Repository prepared: `https://github.com/DorukArtan/Real_Time_Chat_Moderation`; external weight link and fresh setup check pending |
| R8 AI usage | Accurate tool-and-purpose slide | `AI_USAGE.md`; final slide pending |

## Use Case 4 component checklist

- Regex noisy-text normalization, leetspeak and repeated letters. English has no Turkish-specific character normalization requirement.
- Edit-distance lexicon matcher as a separate rule baseline.
- TF-IDF classification baseline.
- Our own Word2Vec trained only on training messages; averaged embeddings and a feedforward network trained with SGD/backpropagation.
- Our own fine-tuning of an English BERT-family Transformer (DistilBERT), justified by the smaller CPU deployment.
- Quantization, a declared CPU latency budget and measured verification of whether that budget is met.
- Chat-like UI, confidence, whole-message masking, per-message latency log and model selection.
- Macro-F1 for the imbalanced binary classes and CPU latency.

## Gate 1 risk of grade 0

Missing/unreadable presentation, video or repository; no live new-input demo; no Transformer or no quantized model; unrelated use case; fabricated results, plagiarism or copied work. Every quantitative claim must come from this project's own saved run.

## Gate 2 risk of grade 50

Missing technique area/map/model; unequal held-out evaluation; incomplete results table; missing/mismatched architecture; incorrect presentation structure; quantized model absent from the running app; broken README setup; missing AI usage slide; or unnarrated video without architecture and results.

## Manual release checklist

- [ ] Record student name and ID in the first slide.
- [x] Run all experiment stages in Colab and save outputs to Drive.
- [x] Inspect the actual confusion matrices and at least one failure example.
- [x] Meet and report the declared CPU latency budget honestly (14.96 ms mean vs 150 ms target; FP32 also meets it).
- [x] Test INT8 default inference with at least two new inputs, one difficult.
- [x] Inspect model switching, masking and timing log in a real browser.
- [ ] Prepare 8-12 slides following Introduction / Development / Conclusion.
- [ ] Include the technique map, matching architecture, own measured results and AI usage.
- [ ] Record a narrated video of at most 2:00 with live app, architecture and results.
- [ ] Publish/share the repository and weights; add tested setup and artifact links.
- [ ] Test setup in a fresh environment without training locally.
- [ ] Check video, repository and weight links with instructor-equivalent access.

The original Colab `reports/technical_checks.json` remains unchanged as run evidence. Its manual-check list reflects the time of that export. The later browser check is recorded separately in `reports/live_validation/`. This does not substitute for the student's narrated video or a fresh-environment installation check. Use **Send message** during the demo: Enter inserted a newline in the tested multiline textbox despite its placeholder.

