# Presentation and video plan

The final deliverable is an 8-12-slide PDF. This 12-slide outline is a plan, not a submitted presentation. Insert only results measured in the saved Colab run.

1. **Title**: student name, ID, Use Case 4, repository link and video link.
2. **Introduction**: gaming/chat moderation problem, intended users, binary labels and success criteria.
3. **Development Data**: original source and mirror, six-to-two label mapping, exact retained sizes/class ratios, preprocessing and frozen splits.
4. **Development Architecture**: actual user-input flow, model names/precision, postprocessing, CPU deployment and Colab training.
5. **Development Technique Map**: explicitly map A/B/C/D to components in the code.
6. **Development Classical Models**: edit-distance rule control and TF-IDF + Logistic Regression; training-only vocabulary.
7. **Development Neural Network**: our own training-only Word2Vec, mean pooling, network layers, SGD/backpropagation and validation selection.
8. **Development Transformer**: pretrained DistilBERT source, why it replaces BERT, fine-tuning settings, training GPU and validation selection.
9. **Development Quantization**: FP32 ONNX control, dynamic INT8 scope, unchanged threshold, integer-operator/parity checks, declared CPU budget.
10. **Development Results**: own macro-F1, model MB and mean latency for all four models, CPU hardware and >=100 runs; auxiliary controls if space permits. Explain trade-offs without assuming the Transformer wins.
11. **Conclusion**: measured findings, one actual failure example, domain shift from Wikipedia comments to chat, truncation, uncalibrated confidence and next steps.
12. **Conclusion AI Usage**: tools and actual purposes, verification responsibility, references. Verify the source list is legible, or distribute citations to relevant slides.

## Narrated video at most two minutes

- 0:00-0:15: describe the problem and target users.
- 0:15-1:15: live application with at least two newly typed messages, including a difficult case; show the INT8 selection, masking, confidence and latency log.
- 1:15-1:45: matching architecture and actual measured results table.
- 1:45-2:00: main finding, limitation and conclusion.

Only the first two minutes are watched. Test the uploaded link in a private browser or with access equivalent to the instructor. Slides, screenshots or notebook printed predictions alone are insufficient evidence of a working app.

