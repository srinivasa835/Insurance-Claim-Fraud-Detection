"""Hand-crafted signals that mirror the problem statement:
repetitive patterns, sentiment anomalies and urgency/inconsistency cues.

Everything is lexicon/regex based so no extra downloads are needed.
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

NEG = {"destroyed", "gone", "huge", "severe", "badly", "stolen", "burnt", "lost",
       "serious", "total", "terrible", "worst", "ruined", "damage", "hurt", "injury"}
POS = {"fine", "minor", "safe", "ok", "okay", "attached", "confirmed", "report", "witness"}
URGENT = {"urgent", "urgently", "immediately", "fast", "asap", "now", "quickly", "maximum"}
VAGUE = {"cannot", "remember", "nobody", "nothing", "unknown", "somewhere", "no", "without"}

FEATURE_NAMES = [
    "n_words", "n_sentences", "type_token_ratio", "repeated_trigram_ratio",
    "repeated_sentence_ratio", "exclaim_ratio", "neg_ratio", "pos_ratio",
    "sentiment_score", "urgency_ratio", "vague_ratio", "digit_ratio",
]


def _tokens(text):
    return re.findall(r"[a-z']+", text.lower())


def extract(text):
    toks = _tokens(text)
    n = max(len(toks), 1)
    sents = [s.strip().lower() for s in re.split(r"[.!?]+", text) if s.strip()]
    ns = max(len(sents), 1)

    trigrams = list(zip(toks, toks[1:], toks[2:]))
    tri_counts = Counter(trigrams)
    rep_tri = sum(c - 1 for c in tri_counts.values() if c > 1) / max(len(trigrams), 1)

    sent_counts = Counter(sents)
    rep_sent = sum(c - 1 for c in sent_counts.values() if c > 1) / ns

    neg = sum(t in NEG for t in toks) / n
    pos = sum(t in POS for t in toks) / n
    return [
        len(toks), len(sents), len(set(toks)) / n, rep_tri, rep_sent,
        text.count("!") / ns, neg, pos, pos - neg,
        sum(t in URGENT for t in toks) / n,
        sum(t in VAGUE for t in toks) / n,
        len(re.findall(r"\d", text)) / max(len(text), 1),
    ]


def build_matrix(texts):
    return pd.DataFrame([extract(t) for t in texts], columns=FEATURE_NAMES)
