"""Single run command: python app.py. Inference only; never fits a model."""

import argparse

from moderation.app import build_app as build_moderation_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    app = build_moderation_app()
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
