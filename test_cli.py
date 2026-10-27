import os

import pandas as pd

from claim_fraud.evaluation import evaluate, inference
from claim_fraud.training import train_tfidf


def test_train_evaluate_infer_end_to_end(tmp_path, capsys):
    out = str(tmp_path / "results")
    train_tfidf.main(["--synthetic", "--out", out])
    weights = os.path.join(out, "models", "tfidf_logreg.joblib")
    for rel in ["training/tfidf_logreg_learning_curve_curves.png", "evaluation/tfidf_logreg_confusion_matrix.png",
                "metrics/tfidf_logreg_test_metrics.json", "explainability/tfidf_top_terms.csv", "models/tfidf_logreg.joblib"]:
        assert os.path.exists(os.path.join(out, rel)), rel

    evaluate.main(["--weights", weights, "--synthetic", "--out", str(tmp_path / "eval")])
    assert "accuracy" in capsys.readouterr().out

    csv = tmp_path / "new.csv"
    pd.DataFrame({"narrative": ["A witness gave details to the police and the repair estimate is attached.",
                                "Everything is destroyed and gone, pay me urgently!"]}).to_csv(csv, index=False)
    pred_csv = tmp_path / "pred.csv"
    inference.main(["--weights", weights, "--source", str(csv), "--out", str(pred_csv)])
    pred = pd.read_csv(pred_csv)
    assert list(pred["prediction"]) == ["genuine", "fraud"]
