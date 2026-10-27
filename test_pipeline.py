import json
import os

import numpy as np
import pytest

from claim_fraud.data.loader import split
from claim_fraud.data.synthetic import make_synthetic_dataset
from claim_fraud.evaluation.metrics import compute_metrics, plot_curves, save_evaluation
from claim_fraud.explainability.occlusion import sentence_occlusion, split_sentences
from claim_fraud.explainability.tfidf_terms import top_terms
from claim_fraud.models import tfidf_model as T


def test_compute_metrics_known_values():
    m = compute_metrics([1, 1, 0, 0], [1, 0, 0, 0], [0.9, 0.4, 0.2, 0.1])
    assert m["accuracy"] == 0.75 and m["precision"] == 1.0 and m["recall"] == 0.5
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == 1.0


def test_compute_metrics_single_class_has_no_auc():
    assert "roc_auc" not in compute_metrics([0, 0], [0, 0], [0.1, 0.2])


def test_save_evaluation_writes_expected_files(tmp_path):
    y = np.array([1, 0, 1, 0, 1, 0])
    prob = np.array([0.9, 0.2, 0.6, 0.4, 0.3, 0.1])
    save_evaluation("demo", y, (prob >= 0.5).astype(int), prob, ["t"] * 6, str(tmp_path))
    for rel in ["evaluation/demo_confusion_matrix.png", "evaluation/demo_confusion_matrix_normalized.png",
                "evaluation/demo_pr_curve.png", "evaluation/demo_roc_curve.png",
                "evaluation/demo_f1_threshold_curve.png", "metrics/demo_test_metrics.json",
                "sample_predictions/demo_sample_predictions.csv"]:
        assert os.path.exists(tmp_path / rel), rel
    data = json.load(open(tmp_path / "metrics/demo_test_metrics.json"))
    assert data["n_test"] == 6 and set(data["per_class"]) == {"genuine", "fraud"}


def test_plot_curves_writes_png_and_history(tmp_path):
    h = {"train_loss": [1, .5], "val_loss": [1.1, .7], "train_acc": [.5, .8], "val_acc": [.4, .7]}
    plot_curves(h, "roberta", str(tmp_path))
    assert os.path.exists(tmp_path / "training/roberta_curves.png")
    assert os.path.exists(tmp_path / "training/roberta_history.csv")


@pytest.fixture(scope="module")
def fitted():
    df = make_synthetic_dataset(600)
    tr, _, te = split(df)
    return T.fit_bundle(tr["narrative"], tr["fraud"]), te


def test_tfidf_model_beats_chance_and_saves(fitted, tmp_path):
    bundle, te = fitted
    prob = T.predict_proba(bundle, te["narrative"])
    assert prob.shape == (len(te),) and ((prob >= 0) & (prob <= 1)).all()
    assert ((prob >= 0.5).astype(int) == te["fraud"].values).mean() > 0.8
    path = tmp_path / "m.joblib"
    T.save_bundle(bundle, path)
    assert np.allclose(T.predict_proba(T.load_bundle(path), te["narrative"]), prob)


def test_top_terms_shape(fitted):
    bundle, _ = fitted
    assert len(top_terms(bundle, 10)) == 10


def test_sentence_occlusion_flags_the_fraud_sentence(fitted):
    bundle, _ = fitted
    text = ("A witness stayed at the scene and gave their contact details to the officer. "
            "My car was totally destroyed, everything is gone, the loss is huge and I need payment urgently!")
    df = sentence_occlusion(lambda t: T.predict_proba(bundle, t), text)
    assert len(df) == 2
    assert "destroyed" in df.iloc[0]["sentence"]          # most influential sentence first
    assert df.iloc[0]["importance"] > 0


def test_split_sentences():
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []
