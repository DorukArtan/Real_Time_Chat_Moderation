# Saved experiment evidence

The numeric reports here come from the project's completed Colab run. They are preserved as exported, rather than regenerated using the developer's Windows CPU.

- `results.csv`: four main models and the two auxiliary controls; macro-F1, accuracy, precision/recall, complete inference size and CPU latency.
- `evaluation.json`: hardware, fixed test provenance, confusion matrices and the 200 raw timing measurements per model.
- `test_predictions.csv`: original test IDs, labels, predicted labels and offensive scores, without raw comment texts.
- `neural_history.json` and `transformer_history.json`: our own per-epoch logs.
- `quantization_checks.json`: integer graph operators and FP32 export parity checks.
- `technical_checks.json`: checks performed when the original Colab run was exported. Its pending manual list is historical.
- `live_validation/`: selected later browser evidence and an actual false-negative example.

Model weights, frozen split texts, the full raw failure-example list and the run bundle are not committed. They remain in the original `moderation_run_bundle.zip`; an instructor-accessible model download link is still pending. Local machine/account diagnostics and the Drive authorization screenshot are also excluded.

The dataset source and binary-label adaptation are documented in the main README. The single comment quoted in the selected failure is an example from that attributed dataset, not a message authored by the student.
