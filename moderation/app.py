"""Interactive chat interface. Defaults to the actual quantized Transformer."""

from __future__ import annotations

import argparse
import time

from moderation.common import ARTIFACTS, REPORTS, config
from moderation.models import DISPLAY, MAIN_MODELS, Predictor


def build_app():
    import gradio as gr

    cfg = config()
    if not (ARTIFACTS / "int8" / "metadata.json").exists():
        raise RuntimeError(
            "Trained INT8 artifacts are missing. Complete the Colab notebook and restore its run bundle first."
        )
    predictors = {"int8": Predictor("int8")}

    def moderate(message, history, model_name, log):
        if not message or not message.strip():
            raise gr.Error("Enter a message first.")
        if len(message) > cfg["max_input_chars"]:
            raise gr.Error(f"Use at most {cfg['max_input_chars']} characters per message.")
        if model_name not in predictors:
            predictors[model_name] = Predictor(model_name)
        start = time.perf_counter_ns()
        result = predictors[model_name].classify(message)
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        shown = "[Message hidden: offensive language]" if result["label"] == "offensive" else message
        history = list(history or [])
        history.extend(
            [
                {"role": "user", "content": shown},
                {
                    "role": "assistant",
                    "content": f"{result['label'].upper()} · confidence {result['confidence']:.1%} · {elapsed:.1f} ms",
                },
            ]
        )
        log = list(log or [])
        log.append(
            [
                len(log) + 1,
                DISPLAY[model_name],
                result["label"],
                round(result["confidence"], 4),
                round(elapsed, 2),
            ]
        )
        return "", history, log, log

    with gr.Blocks(title="English Chat Moderation", theme=gr.themes.Soft()) as app:
        gr.Markdown("# English Chat Moderation\nType a message to check it before it appears in the chat.")
        log_state = gr.State([])
        with gr.Row():
            with gr.Column(scale=3):
                chat = gr.Chatbot(
                    type="messages",
                    label="Moderated conversation",
                    height=440,
                    sanitize_html=True,
                    render_markdown=False,
                )
                message = gr.Textbox(
                    label="Your message", placeholder="Type a message and press Enter", lines=2, max_lines=5
                )
                send = gr.Button("Send message", variant="primary")
                gr.Examples(
                    examples=[
                        ["Thanks for helping our team!"],
                        ["You are a complete idiot."],
                        ["That was a stupid mistake, but we can fix it."],
                    ],
                    inputs=message,
                )
            with gr.Column(scale=2):
                selector = gr.Dropdown(
                    choices=[(DISPLAY[name], name) for name in MAIN_MODELS],
                    value="int8",
                    label="Active model",
                )
                gr.Markdown(
                    "Confidence is the model's score, not a guarantee of correctness. Only the current message is classified."
                )
                log_view = gr.Dataframe(
                    headers=["Message", "Model", "Label", "Confidence", "Latency ms"],
                    value=[],
                    interactive=False,
                    label="Response time log",
                )
                clear = gr.Button("Clear conversation")
        inputs = [message, chat, selector, log_state]
        outputs = [message, chat, log_view, log_state]
        send.click(moderate, inputs, outputs)
        message.submit(moderate, inputs, outputs)
        clear.click(lambda: ([], [], []), outputs=[chat, log_view, log_state])
        if (REPORTS / "results.csv").exists():
            import pandas as pd

            with gr.Accordion("Measured model comparison", open=False):
                gr.Dataframe(value=pd.read_csv(REPORTS / "results.csv"), interactive=False)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True
    )
