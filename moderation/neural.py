"""Our own feedforward classifier, trained with SGD and backpropagation."""

import numpy as np
from torch import nn

from moderation.preprocessing import words


class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_dim, 2)
        )

    def forward(self, inputs):
        return self.layers(inputs)


def average_vectors(texts, vocabulary, vectors):
    output = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
    for index, text in enumerate(texts):
        indices = [vocabulary[word] for word in words(text) if word in vocabulary]
        if indices:
            output[index] = vectors[indices].mean(axis=0)
    return output
