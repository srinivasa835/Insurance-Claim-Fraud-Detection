# Financial Insurance Claim Fraud Detection from Narrative Logs

Deep Learning mini project. Fraudulent claims often hide in long free-text claim
descriptions, and manual review of those logs is slow and expensive. This project
classifies claim narratives as **fraudulent** or **genuine** using two approaches
and compares them:

1. **TF-IDF + hand-crafted signals + Logistic Regression** (classical baseline)
2. **RoBERTa-base fine-tuned** for sequence classification (deep model)

The hand-crafted signals target the patterns named in the problem statement:
repetitive phrasing (repeated trigrams / sentences), sentiment anomalies
(negative-vs-positive word balance, exclamation use), urgency and vagueness cues.

## Team
`<Add team member names and register numbers here>`

## Project structure
```
insurance-claim-fraud-detection/
├── README.md
├── requirements.txt
├── data/                  # put the dataset CSV here (not committed)
├── src/
│   ├── data.py            # loading, cleaning, stratified split, synthetic demo data
│   ├── features.py        # repetition / sentiment / urgency features
│   ├── train_tfidf.py     # baseline model
│   ├── train_roberta.py   # RoBERTa fine-tuning
│   └── evaluate.py        # metrics + plots
├── results/               # generated: curves, confusion matrices, ROC, predictions
└── models/                # generated: saved models
```

## Dataset
The problem statement lists these sources:

| Dataset | Link / note |
|---|---|
| Kaggle - Insurance Claim Fraud | `<paste the exact Kaggle dataset URL you use>` |
| UCI - Fraud Detection in Insurance | `<paste URL>` |
| ISO ClaimSearch | Industry/proprietary - not publicly downloadable |

**Important:** the code expects a CSV with a free-text column (`narrative`) and a
binary label column (`fraud`). Many public insurance-fraud datasets are mostly
tabular and do **not** contain real narrative text. Check your chosen dataset
before training. If it only has tabular fields, either use a dataset that has a
text description column, or state clearly in your report how narratives were built
from the available fields. Column names can be changed with `--text-col` and
`--label-col`.

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
# 1. Quick check that everything works (synthetic demo data, NOT for reporting)
python src/train_tfidf.py --synthetic

# 2. Real data - baseline
python src/train_tfidf.py --data data/claims.csv --text-col narrative --label-col fraud

# 3. Real data - RoBERTa (use a GPU, e.g. free Colab T4)
python src/train_roberta.py --data data/claims.csv --epochs 4 --max-len 256
```

## Methodology
1. **Preprocessing** - lowercase, strip URLs/special characters, normalise numbers,
   drop missing rows, map labels to 0/1.
2. **Split** - stratified 70% train / 10% validation / 20% test (seed 42).
3. **Baseline** - TF-IDF (1-2 grams, sublinear TF) concatenated with 12 scaled
   hand-crafted features, classified by class-weighted Logistic Regression.
4. **RoBERTa** - `roberta-base`, max length 256, AdamW (lr 2e-5), linear warm-up
   schedule, class-weighted cross-entropy, best epoch chosen on validation accuracy.
5. **Evaluation** - accuracy, precision, recall, F1, ROC-AUC on the held-out test
   set. For fraud, recall and F1 on the fraud class matter more than raw accuracy
   because fraud cases are the minority.

## Results
Fill this table from `results/*_report.txt` after running on your real dataset.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| TF-IDF + LogReg | | | | | |
| RoBERTa-base | | | | | |

Generated plots (in `results/`):
- `tfidf_logreg_learning_curve_curves.png` - loss/accuracy vs training size
- `roberta_curves.png` - loss/accuracy per epoch
- `*_confusion_matrix.png`, `*_roc.png`
- `*_sample_predictions.csv` - example predictions
- `tfidf_top_terms.csv` - words most associated with fraud / genuine claims

## Key findings
`<Write 3-4 points after you have real results: which model won, what the top
fraud indicator terms were, where each model failed.>`

## Limitations and future scope
- RoBERTa truncates narratives beyond 512 tokens; chunking long logs is a next step.
- Add explainability (SHAP / attention) so reviewers see why a claim was flagged.
- Combine text with tabular claim fields in one model.
- Cost-sensitive thresholds tuned to the insurer's review capacity.

## References
1. Liu, Y. et al. (2019). *RoBERTa: A Robustly Optimized BERT Pretraining Approach.* arXiv:1907.11692.
2. Devlin, J. et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers.* NAACL.
3. Salton, G. & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval.
4. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12.
5. Wolf, T. et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP (Demos).
6. Dataset pages listed above.
