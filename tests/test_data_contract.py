import pandas as pd
import pytest

from moderation.data import LABELS, clean_source, make_splits
from moderation.preprocessing import normalize


def source_row(identifier, text, positive=False):
    return {
        "id": identifier,
        "comment_text": text,
        **{label: int(positive and label == "insult") for label in LABELS},
    }


def test_binary_target_covers_any_of_six_labels():
    rows = [source_row("0", "Friendly message")]
    for i, label in enumerate(LABELS):
        row = source_row(str(i + 1), f"Different message {i}")
        row[label] = 1
        rows.append(row)
    assert clean_source(pd.DataFrame(rows)).label.tolist() == [0, 1, 1, 1, 1, 1, 1]


def test_conflicting_normalized_duplicates_removed_before_splitting():
    rows = [
        source_row("a", "HELLO   THERE"),
        source_row("b", "hello there", True),
        source_row("c", "Other text"),
        source_row("d", "OTHER TEXT"),
    ]
    cleaned = clean_source(pd.DataFrame(rows))
    assert cleaned.id.tolist() == ["c"]


def test_unlabeled_official_test_rows_rejected():
    row = source_row("a", "A message")
    row["toxic"] = -1
    with pytest.raises(ValueError, match="0 or 1"):
        clean_source(pd.DataFrame([row]))


def test_stratified_frozen_splits_are_disjoint_and_repeatable():
    rows = [source_row(str(i), f"Unique comment number {i}", i % 5 == 0) for i in range(300)]
    cleaned = clean_source(pd.DataFrame(rows))
    cfg = {"train_size": 100, "validation_size": 25, "test_size": 25, "seed": 42}
    first, second = make_splits(cleaned, cfg), make_splits(cleaned, cfg)
    seen = set()
    for split in first:
        pd.testing.assert_frame_equal(first[split], second[split])
        assert len(first[split]) == cfg[f"{split}_size"]
        assert set(first[split].label) == {0, 1}
        assert not seen.intersection(first[split].normalized)
        seen.update(first[split].normalized)


@pytest.mark.parametrize(
    "text", ["SH!T", "you are stuuuupid", "@someone hello https://example.com", "hello!", "Room 123"]
)
def test_normalization_is_idempotent(text):
    assert normalize(normalize(text)) == normalize(text)


def test_leetspeak_and_punctuation_do_not_corrupt_normal_words():
    assert normalize("h3llo sh!t") == "hello shit"
    assert normalize("hello!") == "hello!"
    assert normalize("Room 123") == "room 123"
