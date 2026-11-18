import pytest

from moderation.common import require_cloud_training
from moderation.evaluation import benchmark
from moderation.models import RuleMatcher
from moderation.training import choose_threshold


def test_local_fitting_is_blocked():
    with pytest.raises(RuntimeError, match="restricted to Google Colab"):
        require_cloud_training()


def test_rule_matcher_handles_one_edit_without_substring_false_positive():
    model = RuleMatcher(["idiot", "ass"])
    assert model.score("you are an idoit") == 0  # Transposition costs two Levenshtein edits.
    assert model.score("you are an idiott") == 1
    assert model.score("an assessment of the class") == 0


def test_validation_threshold_can_address_imbalanced_scores():
    import numpy as np

    labels = np.asarray([0, 0, 0, 1])
    scores = np.asarray([0.02, 0.08, 0.15, 0.35])
    threshold = choose_threshold(labels, scores)
    assert ((scores >= threshold) == labels).all()


def test_latency_requires_hundred_real_runs():
    class Counter:
        calls = 0

        def scores(self, messages):
            self.calls += 1
            return [0.1] * len(messages)

    model = Counter()
    with pytest.raises(ValueError, match="at least 100"):
        benchmark(model, ["Hello"], runs=99, warmup=0)
    measured = benchmark(model, ["Hello", "Thanks"], runs=100, warmup=3)
    assert model.calls == 103
    assert len(measured["raw_latency_ms"]) == 100
    assert measured["latency_mean_ms"] >= 0
