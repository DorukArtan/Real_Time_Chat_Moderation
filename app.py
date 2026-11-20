"""Single run command: python app.py. Inference only; never fits a model."""

import argparse
from html import escape
from pathlib import Path

from moderation.app import build_app as build_moderation_app
from moderation.models import DISPLAY

UI = Path(__file__).resolve().parent / "ui"
INT8_LABEL = "Quantized DistilBERT (INT8 CPU)"


def render_log(rows):
    """Render the session's numeric log as a compact, accessible read-only table."""
    body = []
    for number, model, label, confidence, latency in rows:
        name = INT8_LABEL if model == DISPLAY["int8"] else model
        body.append(
            f"<tr><td>{int(number)}</td><td>{escape(name)}</td>"
            f'<td data-verdict="{escape(label)}">{escape(label)}</td>'
            f"<td>{confidence:.1%}</td><td>{latency:.1f} ms</td></tr>"
        )
    if not body:
        body.append('<tr><td class="log-empty" colspan="5">No messages yet.</td></tr>')
    headings = ("Message", "Model", "Label", "Confidence", "Latency")
    header = "".join(f'<th scope="col">{title}</th>' for title in headings)
    return (
        '<h2 id="log-title">Moderation log</h2><div class="log-scroll" tabindex="0" '
        'role="region" aria-labelledby="log-title"><table aria-labelledby="log-title">'
        f"<thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def build_app():
    import gradio as gr

    app = build_moderation_app()
    blocks = list(app.blocks.values())
    chat = next(block for block in blocks if isinstance(block, gr.Chatbot))
    message = next(block for block in blocks if isinstance(block, gr.Textbox))
    send, clear = [block for block in blocks if isinstance(block, gr.Button)]
    selector = next(block for block in blocks if isinstance(block, gr.Dropdown))
    examples = next(block for block in blocks if isinstance(block, gr.Dataset))
    log_state = next(block for block in blocks if isinstance(block, gr.State))
    raw_log = next(block for block in blocks if isinstance(block, gr.Dataframe) and block.label)
    comparison = next((block for block in blocks if isinstance(block, gr.Accordion)), None)
    intro, note = [block for block in blocks if isinstance(block, gr.Markdown)]

    intro.elem_id = "moderation-intro"
    intro.value = "# English Chat Moderation\nDetect offensive language in real time."
    chat.elem_id = "conversation"
    chat.height = "clamp(325px, 30.6vw, 445px)"
    chat.parent.parent.elem_id = "moderation-workspace"
    inspector = raw_log.parent
    inspector.elem_id = "model-panel"
    selector.label = "Model"
    selector.elem_id = "model-selector"
    selector.choices = [(INT8_LABEL if key == "int8" else name, key) for name, key in selector.choices]
    note.elem_id = "confidence-note"
    note.value = "Confidence is an uncalibrated model score."
    raw_log.visible = False
    message.lines = 1
    message.elem_id = "moderation-message"
    message.placeholder = "Type a message. Enter to send; Shift+Enter for a new line."
    # Gradio replays its saved textbox properties when an example is selected.
    examples.component_props[0].update(
        lines=message.lines, elem_id=message.elem_id, placeholder=message.placeholder
    )
    send.value = "Send"
    send.elem_id = "moderation-send"
    send.min_width = 88
    send.scale = 0
    clear.elem_id = "clear-conversation"
    examples.elem_id = "message-examples"
    app.fill_width = True
    app.css = (UI / "moderation.css").read_text(encoding="utf-8")

    # Move the existing controls with their Gradio events and session state intact.
    with app:
        with chat.parent:
            with gr.Column(elem_id="composer-panel", min_width=0):
                with gr.Row(elem_id="composer-row"):
                    message.parent.unrender().render()
                    send.unrender().render()
                examples.unrender().render()
                if comparison is not None:
                    comparison.elem_id = "model-comparison"
                    comparison.unrender().render()
        with inspector:
            log_table = gr.HTML(render_log([]), elem_id="moderation-log")
            clear.unrender().render()
        log_state.change(render_log, inputs=log_state, outputs=log_table, show_progress="hidden")
        chat.clear(lambda: ([], []), outputs=[raw_log, log_state], show_progress="hidden")
        app.load(fn=None, js="""
        () => {
            const conversation = document.querySelector("#conversation");
            const formatVerdicts = () => {
                for (const text of conversation.querySelectorAll('[data-testid="bot"] .md')) {
                    if (text.querySelector("strong")) continue;
                    const match = /^(CLEAN|OFFENSIVE) · confidence (.+)$/.exec(text.textContent);
                    if (!match) continue;
                    const label = document.createElement("strong");
                    label.textContent = match[1];
                    text.replaceChildren(label, " · " + match[2]);
                }
            };
            if (conversation) {
                formatVerdicts();
                new MutationObserver(formatVerdicts).observe(conversation, {
                    childList: true, subtree: true, characterData: true
                });
            }
            const stopEmptySubmission = (event, input) => {
                if (input && !input.value.trim()) {
                    event.preventDefault();
                    event.stopImmediatePropagation();
                    input.focus();
                }
            };
            document.addEventListener("keydown", (event) => {
                if (event.key !== "Enter" || event.shiftKey) return;
                const input = event.composedPath().find(node =>
                    node instanceof HTMLTextAreaElement &&
                    node.closest("#moderation-message")
                );
                stopEmptySubmission(event, input);
            }, {capture: true});
            document.addEventListener("click", (event) => {
                const button = event.composedPath().find(node =>
                    node instanceof Element && node.id === "moderation-send"
                );
                if (!button) return;
                const input = button.getRootNode().querySelector("#moderation-message textarea");
                stopEmptySubmission(event, input);
            }, {capture: true});
        }
        """)
    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
