# Live CPU application check

The actual restored Colab models were tested in the browser on 2026-11-17. The default and final selected model was **DistilBERT INT8 (ONNX CPU)**. All four main model choices returned real predictions. Later, the shared Windows launcher was also checked by starting a second server and submitting a real message through its API.

| Typed input | Observed INT8 result | Displayed confidence |
| --- | --- | ---: |
| `Thanks for the match. Let's play again tomorrow.` | Clean; visible | 99.9% |
| `You are a useless idiot. Get lost.` | Offensive; masked | 99.9% |
| `Y0u are a c0mplete 1d10t.` | Offensive; masked | 99.9% |

The label, confidence and response-time log appeared in the UI. Clearing the conversation removed chat and log entries. An empty submission was rejected. The measured-comparison panel showed the six original result rows.

`selected_failure.json` records an actual test false negative, including its frozen test ID and score. `int8_failure_example.jpg`, `verified_int8_app.jpg` and `final_demo_ax.txt` show the later live result. This case also demonstrates that a high confidence score need not be correct.

The few Windows per-message timings are smoke-check observations; they do not replace the formal 200-call Colab CPU benchmark. Confidence is uncalibrated. Only the current message is classified. In the tested multiline textbox, Enter inserted a newline despite the placeholder; use **Send message** to submit.

Machine-specific diagnostic reports remain local. These screenshots and checks do not replace the student's required narrated demo video. A fresh-environment setup check and final submission-link access check remain pending.
