# English moderation architecture

The offline experiment runs in Google Colab. Trained artifacts and evidence persist in the student's Drive. The app can run inside Colab for interactive checking, or on a CPU after restoring the exported bundle. No model is trained on the student's computer.

```mermaid
flowchart TD
    J[Jigsaw labeled original train.csv] --> N[English regex normalization]
    N --> D[Remove normalized duplicates and label conflicts]
    D --> S[Seeded stratified train / validation / test]
    S --> T[Training split only]
    T --> LR[TF-IDF + Logistic Regression]
    T --> W[Own Word2Vec from scratch]
    W --> M[Averaged vectors + own feedforward NN\nSGD and backpropagation]
    T --> B[Pretrained English DistilBERT\nFine-tune two-class head on GPU]
    B --> F[Saved FP32 weights]
    F --> O[FP32 ONNX CPU control]
    O --> Q[Dynamic INT8 MatMul/Gemm\nOther operations remain floating point]
    V[Validation only: epoch and threshold selection] --> LR
    V --> M
    V --> F
    F --> H[Reuse the same threshold in both ONNX versions]
    H --> O
    H --> Q
    L[Hand-specified lexicon] --> R[Minimum-edit-distance rule baseline]
    S --> E[Shared frozen test evaluation\nCPU only, fixed threads, identical latency inputs]
    LR --> E
    M --> E
    F --> E
    O --> E
    Q --> E
    R --> E
    E --> A[Measured results, sizes, raw latencies,\nhardware, failure examples and hashes in Drive]
```

```mermaid
flowchart LR
    U[New English message in Gradio] --> P[Validate length and normalize]
    P --> C{Selected model\nDefault: INT8}
    C --> L[TF-IDF + LR\nsklearn CPU]
    C --> N[Word2Vec mean + FFN\nFP32 PyTorch CPU]
    C --> B[DistilBERT\nFP32 PyTorch CPU]
    C --> Q[DistilBERT INT8 ONNX\nCPUExecutionProvider]
    L --> S[Model score + validation threshold]
    N --> S
    B --> S
    Q --> S
    S --> V[Label and confidence\nMask flagged message]
    V --> UI[Chat + response time log]
```

The UI classifies one message at a time and does not infer from chat history. Model loading is excluded from both displayed response timing and the formal warm benchmark. Formal latency includes preprocessing and score postprocessing but excludes browser/network time. Artifacts include tokenizer and Word2Vec dependencies when size is measured.

