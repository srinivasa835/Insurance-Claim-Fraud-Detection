"""TF-IDF baseline: TF-IDF (word 1-2 grams) + hand-crafted signals -> Logistic Regression.

Usage:
    python src/train_tfidf.py --synthetic
    python src/train_tfidf.py --data data/claims.csv --text-col narrative --label-col fraud
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
import data as D
import features as F
from evaluate import plot_curves, save_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--text-col", default="narrative")
    ap.add_argument("--label-col", default="fraud")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = D.make_synthetic_dataset() if args.synthetic else D.load_csv(args.data, args.text_col, args.label_col)
    df["clean"] = df["narrative"].map(D.clean_text)
    train, val, test = D.split(df)
    print(f"train={len(train)} val={len(val)} test={len(test)} fraud_rate={df['fraud'].mean():.2%}")

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
    scaler = StandardScaler()

    def feats(frame, fit=False):
        extra = F.build_matrix(frame["narrative"]).values
        if fit:
            x_text = tfidf.fit_transform(frame["clean"]); extra = scaler.fit_transform(extra)
        else:
            x_text = tfidf.transform(frame["clean"]); extra = scaler.transform(extra)
        return hstack([x_text, csr_matrix(extra)]).tocsr()

    X_tr, X_va, X_te = feats(train, True), feats(val), feats(test)

    # Learning curve over training-set size (used as the "accuracy/loss curve" for the classical model)
    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    from sklearn.metrics import log_loss
    for frac in [0.2, 0.4, 0.6, 0.8, 1.0]:
        n = int(X_tr.shape[0] * frac)
        m = LogisticRegression(max_iter=2000, class_weight="balanced")
        m.fit(X_tr[:n], train["fraud"][:n])
        hist["train_loss"].append(log_loss(train["fraud"][:n], m.predict_proba(X_tr[:n])))
        hist["val_loss"].append(log_loss(val["fraud"], m.predict_proba(X_va)))
        hist["train_acc"].append(m.score(X_tr[:n], train["fraud"][:n]))
        hist["val_acc"].append(m.score(X_va, val["fraud"]))
    plot_curves(hist, "tfidf_logreg_learning_curve", args.out)

    model = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X_tr, train["fraud"])
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = save_report("tfidf_logreg", test["fraud"].values, pred, prob, test["narrative"].values, args.out)
    print({k: round(v, 4) for k, v in metrics.items()})

    # Most informative n-grams
    names = list(tfidf.get_feature_names_out()) + F.FEATURE_NAMES
    coef = model.coef_[0]
    top = np.argsort(coef)
    pd.DataFrame({"fraud_indicators": [names[i] for i in top[::-1][:20]],
                  "genuine_indicators": [names[i] for i in top[:20]]}).to_csv(
        os.path.join(args.out, "tfidf_top_terms.csv"), index=False)

    os.makedirs("models", exist_ok=True)
    joblib.dump({"tfidf": tfidf, "scaler": scaler, "model": model}, "models/tfidf_logreg.joblib")


if __name__ == "__main__":
    main()
