"""Deterministic English noise normalization; no statistics learned from test data."""

import html
import re
import unicodedata

LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t"})


def normalize(text, max_chars=4000):
    text = unicodedata.normalize("NFKC", html.unescape(str(text)[:max_chars])).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url ", text)
    text = re.sub(r"(?<!\w)@[a-z0-9_]+", " user ", text)
    text = re.sub(r"(?<=[a-z])[@$!](?=[a-z])", lambda m: {"@": "a", "$": "s", "!": "i"}[m[0]], text)
    text = re.sub(r"[a-z0-9]+", lambda m: m[0].translate(LEET) if re.search("[a-z]", m[0]) else m[0], text)
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)
