import pandas as pd
import pytest

from claim_fraud.data.loader import get_dataframe, load_csv, split
from claim_fraud.data.synthetic import make_synthetic_dataset


def test_synthetic_is_reproducible_and_binary():
    a, b = make_synthetic_dataset(200), make_synthetic_dataset(200)
    assert a.equals(b)
    assert set(a["fraud"].unique()) <= {0, 1}
    assert 0.1 < a["fraud"].mean() < 0.5


def test_split_sizes_and_no_overlap():
    df = make_synthetic_dataset(500).drop_duplicates("narrative").reset_index(drop=True)
    tr, va, te = split(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert not (set(tr["narrative"]) & set(te["narrative"]))
    assert not (set(tr["narrative"]) & set(va["narrative"]))
    assert not (set(va["narrative"]) & set(te["narrative"]))


def test_split_is_stratified_and_deterministic():
    df = make_synthetic_dataset(600)
    tr1, _, te1 = split(df)
    tr2, _, te2 = split(df)
    assert tr1.equals(tr2) and te1.equals(te2)
    assert abs(tr1["fraud"].mean() - te1["fraud"].mean()) < 0.05


def test_load_csv_maps_labels_and_drops_missing(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({"text": ["a claim", "b claim", None, "c claim", "a claim"],
                  "label": ["Y", "no", "yes", 1, "Y"]}).to_csv(p, index=False)
    df = load_csv(p, text_col="text", label_col="label")
    assert list(df.columns) == ["narrative", "fraud"]
    assert len(df) == 3                      # missing row and duplicate removed
    assert dict(zip(df["narrative"], df["fraud"])) == {"a claim": 1, "b claim": 0, "c claim": 1}


def test_load_csv_reports_missing_column(tmp_path):
    p = tmp_path / "c.csv"
    pd.DataFrame({"x": [1], "y": [0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="not found"):
        load_csv(p)


def test_get_dataframe_requires_source():
    with pytest.raises(ValueError):
        get_dataframe()
