"""No fitting: tiny randomly initialized fixtures test export/runtime compatibility."""

import numpy as np
import pytest


@pytest.mark.integration
def test_dynamic_onnx_shapes_and_real_int8_cpu_graph(tmp_path):
    torch = pytest.importorskip("torch")
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("transformers")
    from transformers import BertTokenizerFast, DistilBertConfig, DistilBertForSequenceClassification

    from moderation.quantization import export_graphs

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nteam\nthank\nyou\nfor\nhelping\n.\n", encoding="utf-8"
    )
    tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
    torch.manual_seed(42)
    cfg = DistilBertConfig(
        vocab_size=12,
        n_layers=1,
        n_heads=2,
        dim=16,
        hidden_dim=32,
        max_position_embeddings=128,
        num_labels=2,
        dropout=0.0,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    model = DistilBertForSequenceClassification(cfg).eval()
    evidence = export_graphs(model, tokenizer, tmp_path / "fp32", tmp_path / "int8")
    assert any("Integer" in key for key in evidence["integer_operator_counts"])
    for texts in (["hello"], ["hello team", "thank you for helping"]):
        inputs = tokenizer(texts, padding=True, return_tensors="pt")
        with torch.inference_mode():
            reference = model(**{k: v for k, v in inputs.items() if k != "token_type_ids"}).logits.numpy()
        for precision in ("fp32", "int8"):
            session = ort.InferenceSession(
                str(tmp_path / precision / "model.onnx"), providers=["CPUExecutionProvider"]
            )
            feeds = {item.name: inputs[item.name].numpy() for item in session.get_inputs()}
            output = session.run(["logits"], feeds)[0]
            assert output.shape == (len(texts), 2)
            np.testing.assert_allclose(output, reference, atol=0.02 if precision == "int8" else 1e-5)


@pytest.mark.integration
def test_app_default_is_int8_and_has_four_model_choices(tmp_path, monkeypatch):
    pytest.importorskip("gradio")
    import app as launcher
    import moderation.app as ui

    (tmp_path / "int8").mkdir()
    (tmp_path / "int8" / "metadata.json").write_text("{}")
    loaded = []

    class InferenceFixture:
        def __init__(self, name):
            loaded.append(name)

    monkeypatch.setattr(ui, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(ui, "Predictor", InferenceFixture)
    app = launcher.build_app()
    dropdowns = [item for item in app.config["components"] if item["type"] == "dropdown"]
    assert loaded == ["int8"]
    assert dropdowns[0]["props"]["value"] == "int8"
    assert len(dropdowns[0]["props"]["choices"]) == 4
    examples = next(item for item in app.config["components"] if item["type"] == "dataset")
    textbox = next(item for item in app.config["components"] if item["type"] == "textbox")
    for field in ("lines", "elem_id", "placeholder"):
        assert examples["props"]["component_props"][0][field] == textbox["props"][field]
    app.close()
