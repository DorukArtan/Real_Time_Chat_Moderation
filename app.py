"""Single run command: python app.py. Inference only; never fits a model."""

import argparse

from moderation.app import build_app as build_moderation_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = build_moderation_app()
    from gradio import Button, Textbox

    message = next(
        block for block in app.blocks.values()
        if isinstance(block, Textbox) and block.label == "Your message"
    )
    message.lines = 1
    message.elem_id = "moderation-message"
    message.placeholder = "Type a message. Enter to send; Shift+Enter for a new line."
    send = next(
        block for block in app.blocks.values()
        if isinstance(block, Button) and block.value == "Send message"
    )
    send.elem_id = "moderation-send"
    with app:
        app.load(fn=None, js="""
        () => {
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
    # Keep settings and API controls available without the Gradio branding link.
    app.css = (app.css or "") + """
    footer .built-with,
    footer .divider {
        display: none !important;
    }
    """
    app.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
