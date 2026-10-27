"""Data loading, cleaning and splitting for narrative insurance-claim fraud detection.

Expected CSV format: one row per claim with
  - a free-text narrative column  (default name: "narrative")
  - a binary label column         (default name: "fraud", 1 = fraudulent, 0 = genuine)

If you do not have a real narrative dataset yet, `make_synthetic_dataset()` builds
a small demo corpus so the whole pipeline can be run end to end. Results on the
synthetic data are only for checking that the code works - report results on a real
dataset (Kaggle Insurance Claim Fraud, UCI, ISO ClaimSearch) in your final poster.
"""
import random
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

# --------------------------------------------------------------------------- #
# Loading / cleaning
# --------------------------------------------------------------------------- #

def load_csv(path, text_col="narrative", label_col="fraud"):
    df = pd.read_csv(path)
    df = df.rename(columns={text_col: "narrative", label_col: "fraud"})
    df = df[["narrative", "fraud"]].dropna()
    df["fraud"] = df["fraud"].map(
        lambda v: 1 if str(v).strip().lower() in {"1", "y", "yes", "true", "fraud"} else 0
    )
    return df.reset_index(drop=True)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\d+", " 0 ", text)           # keep "a number was here" signal
    text = re.sub(r"[^a-z0-9!?.,' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split(df, test_size=0.2, val_size=0.1):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["fraud"], random_state=SEED
    )
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["fraud"], random_state=SEED
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic demo data (pipeline check only)
# --------------------------------------------------------------------------- #

_GENUINE = [
    "I was driving home on {road} when another vehicle struck the rear of my car at a red light.",
    "The accident happened around {time}. I called the police and they filed a report.",
    "My car sustained damage to the rear bumper and tail lamp. I have attached the repair estimate.",
    "A witness stayed at the scene and gave their contact details to the officer.",
    "I took photographs of both vehicles and exchanged insurance details with the other driver.",
    "The repair shop inspected the car and confirmed the damage matches the collision.",
    "No one was seriously hurt, though I visited the clinic the next day as a precaution.",
]
_FRAUD = [
    "My car was totally destroyed, everything is gone, the loss is huge and I need payment urgently!",
    "It happened late at night, nobody saw anything, there was no one around and no cameras.",
    "I lost my expensive jewellery, laptop and phone in the same incident, all brand new.",
    "I cannot remember the exact place but the damage is very serious and costs a lot.",
    "I was hurt badly, the injury is severe, I need the maximum amount immediately.",
    "The car was stolen and later found burnt, I have no receipts but it was worth a lot.",
    "Please settle my claim fast, I will not provide more documents, the loss is total!",
]
_ROADS = ["NH16", "Main Road", "Ring Road", "MG Road", "Bypass Road"]
_TIMES = ["8 am", "6 pm", "noon", "9 pm", "7:30 am"]


def make_synthetic_dataset(n=1500, fraud_rate=0.3, seed=SEED):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        is_fraud = rng.random() < fraud_rate
        pool = _FRAUD if is_fraud else _GENUINE
        k = rng.randint(3, 6)
        sents = rng.choices(pool, k=k) if is_fraud else rng.sample(pool, k=min(k, len(pool)))
        if rng.random() < 0.25:  # label noise: mix in some sentences of the other class
            sents.append(rng.choice(_GENUINE if is_fraud else _FRAUD))
        text = " ".join(s.format(road=rng.choice(_ROADS), time=rng.choice(_TIMES)) for s in sents)
        rows.append((text, int(is_fraud)))
    return pd.DataFrame(rows, columns=["narrative", "fraud"])
