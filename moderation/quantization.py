"""Export the same fine-tuned weights to FP32 ONNX and dynamic INT8 ONNX."""

from __future__ import annotations

import collections
import time

import numpy as np

from moderation.common import ARTIFACTS, REPORTS, provenance, read_json, require_cloud_training, save_json


def export_graphs(model, tokenizer, fp32, int8):
    """Inference-only conversion, also exercised with a tiny untrained test fixture."""
    import onnx
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class LogitsOnly(torch.nn.Module):
        def __init__(self, classifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, input_ids, attention_mask):
            return self.classifier(input_ids=input_ids, attention_mask=attention_mask).logits

    fp32.mkdir(parents=True, exist_ok=True)
    int8.mkdir(parents=True, exist_ok=True)
    dummy = tokenizer("Thank you for helping the team.", return_tensors="pt")
    raw = fp32 / "export.onnx"
    start = time.perf_counter()
    wrapper = LogitsOnly(model).eval()
    torch.onnx.export(
        wrapper,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(raw),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    quant_pre_process(str(raw), str(fp32 / "model.onnx"), skip_optimization=False, auto_merge=True)
    onnx.checker.check_model(str(fp32 / "model.onnx"))
    quantize_dynamic(
        str(fp32 / "model.onnx"),
        str(int8 / "model.onnx"),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )
    quantized = onnx.load(str(int8 / "model.onnx"))
    onnx.checker.check_model(quantized)
    ops = collections.Counter(node.op_type for node in quantized.graph.node)
    if not any("Integer" in op or op.startswith("QLinear") for op in ops):
        raise RuntimeError("No integer computation nodes found; refusing to label this model INT8.")
    raw.unlink()  # Generated intermediate, not a deployed model.
    for directory in (fp32, int8):
        tokenizer.save_pretrained(directory)
    return {"conversion_seconds": time.perf_counter() - start, "integer_operator_counts": dict(ops)}


def quantize():
    require_cloud_training()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from moderation.common import check_metadata
    from moderation.models import Predictor
    from moderation.training import completed, metadata

    provenance()
    if completed("int8") and completed("onnx_fp32"):
        return
    parent = ARTIFACTS / "transformer"
    parent_meta = read_json(parent / "metadata.json")
    check_metadata(parent_meta)
    model = (
        AutoModelForSequenceClassification.from_pretrained(
            parent, local_files_only=True, attn_implementation="eager"
        )
        .cpu()
        .float()
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(parent, local_files_only=True)
    evidence = export_graphs(model, tokenizer, ARTIFACTS / "onnx_fp32", ARTIFACTS / "int8")
    metadata(
        "onnx_fp32",
        parent_meta["threshold"],
        precision="FP32 ONNX CPU",
        parent="transformer",
        stage_status="pending",
    )
    metadata(
        "int8",
        parent_meta["threshold"],
        precision="dynamic INT8 MatMul/Gemm; remaining operators FP32",
        parent="transformer",
        method="ONNX Runtime dynamic per-channel QInt8",
        **evidence,
        stage_status="pending",
    )

    # Inspect parity on validation messages only, never on the held-out test set.
    from moderation.data import load_split

    messages = load_split("validation").comment_text.iloc[:64].tolist()
    reference = Predictor("transformer").scores(messages)
    exported = Predictor("onnx_fp32", allow_pending=True).scores(messages)
    compressed = Predictor("int8", allow_pending=True).scores(messages)
    max_error = float(np.max(np.abs(reference - exported)))
    if max_error > 0.001:
        raise RuntimeError(f"FP32 export parity check failed (score error {max_error}).")
    save_json(
        REPORTS / "quantization_checks.json",
        {
            "validation_samples": len(messages),
            "fp32_onnx_max_score_error": max_error,
            "int8_mean_score_change": float(np.mean(np.abs(reference - compressed))),
            "threshold_policy": "The FP32 validation threshold is reused unchanged for both ONNX models.",
            "int8_is_partial_quantization": True,
            "integer_operator_counts": evidence["integer_operator_counts"],
        },
    )
    for name in ("onnx_fp32", "int8"):
        path = ARTIFACTS / name / "metadata.json"
        info = read_json(path)
        info["stage_status"] = "complete"
        save_json(path, info)
    print("FP32 export parity and integer graph checks passed.", flush=True)
