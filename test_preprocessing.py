import pytest

from claim_fraud.preprocessing.features import FEATURE_NAMES, build_matrix, extract
from claim_fraud.preprocessing.text import clean_text


def test_clean_text_lowercases_and_normalises():
    out = clean_text("Visit HTTP://x.com NOW! Loss of $5000 @ 9pm.")
    assert "http" not in out and "$" not in out and "@" not in out
    assert out == out.lower()
    assert "5000" not in out


def test_clean_text_handles_non_strings():
    assert clean_text(None) == "none"
    assert clean_text(123) == "0"


def test_feature_vector_length_and_names():
    assert len(extract("A short claim.")) == len(FEATURE_NAMES) == 12
    assert list(build_matrix(["a", "b c"]).columns) == FEATURE_NAMES


def test_repetition_features_detect_repeated_sentences():
    rep = dict(zip(FEATURE_NAMES, extract("I need money now. I need money now. I need money now.")))
    plain = dict(zip(FEATURE_NAMES, extract("The car hit a pole. A witness called the police. Repairs cost little.")))
    assert rep["repeated_sentence_ratio"] > 0 == plain["repeated_sentence_ratio"]
    assert rep["repeated_trigram_ratio"] > plain["repeated_trigram_ratio"]
    assert rep["type_token_ratio"] < plain["type_token_ratio"]


def test_sentiment_and_urgency_features():
    f = dict(zip(FEATURE_NAMES, extract("Everything is destroyed! Pay me urgently and immediately!")))
    assert f["neg_ratio"] > 0 and f["urgency_ratio"] > 0 and f["exclaim_ratio"] > 0
    assert f["sentiment_score"] < 0


def test_empty_text_does_not_crash():
    vals = extract("")
    assert all(v == v for v in vals)  # no NaN
