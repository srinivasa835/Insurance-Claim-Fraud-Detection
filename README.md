# Financial Insurance Claim Fraud Detection from Narrative Logs

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red) ![Transformers](https://img.shields.io/badge/HuggingFace-RoBERTa-yellow) ![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-orange) ![License: MIT](https://img.shields.io/badge/License-MIT-green)

An automated text-classification system that flags potentially fraudulent insurance claims from their written narrative. It compares a classical **TF-IDF + Logistic Regression** baseline with a fine-tuned **RoBERTa-base** transformer, and adds interpretable signals for repetitive patterns, sentiment anomalies, urgency and vagueness. The TF-IDF model is trained and evaluated on the Kaggle Insurance Claim Fraud dataset (1,000 claims) with verified results; the RoBERTa trainer is included and ready to run on a GPU.

> **Note**
>
> The Kaggle file contains structured fields only, with no free-text column. Narratives were generated from each claim's recorded fields by `claim_fraud/data/build_narratives.py` (Section 4). The fraud label is never used to build the text.

## 1. Project Overview

Insurance providers receive claims with long textual descriptions, and manual review is slow and expensive. This project provides an end-to-end, reproducible pipeline that:

- Builds claim narratives, cleans them and creates a stratified train / validation / test split.
- Trains a TF-IDF + Logistic Regression baseline and a fine-tuned RoBERTa model on identical splits.
- Extracts interpretable signals (repetition, sentiment, urgency, vagueness) and reports the words most associated with fraud.
- Saves metrics, confusion matrices, ROC / precision-recall curves, learning curves and sample predictions.

## 2. Problem Statement

Fraudulent claims submitted through long textual logs cost insurance providers significant manual review time. Manual review suffers from:

- **Volume and fatigue:** reviewers must read many narratives, so decisions become inconsistent.
- **Subtle linguistic cues:** semantic inconsistencies, sentiment anomalies and repetitive patterns are hard to spot by eye across thousands of claims.
- **Class imbalance:** genuine claims far outnumber fraudulent ones (75.3% vs 24.7% here), so accuracy alone is a misleading measure.

## 3. Objectives

- **Binary Fraud Classification:** classify each claim narrative as fraudulent or genuine.
- **Model Comparison:** compare a sparse-feature baseline against a pretrained transformer on identical splits.
- **Imbalance-Aware Evaluation:** report precision, recall, F1, ROC-AUC and average precision, with class-weighted training.
- **Interpretability:** surface the n-grams that push a claim toward fraud or genuine.
- **Reproducibility:** fixed random seed (42), stratified splits and command-line entry points.

## 4. Dataset Description

The project uses the Kaggle **Auto Insurance Claims Data** (`insurance_claims.csv`), listed in the problem statement as Kaggle Insurance Claim Fraud.

- **Source:** https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
- **Total claims:** 1,000 (753 genuine, 247 fraudulent; fraud rate 24.70%)
- **Original fields:** 39 columns, including incident type and severity, collision type, authorities contacted, witnesses, injuries, police report, claim amounts, vehicle details and the insured's occupation and hobby. The label is `fraud_reported` (Y / N).
- **Split scheme:** stratified 70% train (700), 10% validation (100), 20% independent test (200 claims, 49 fraudulent), seed 42.
- **Narrative construction:** each claim is written out as plain English, for example: *"The insured, a 48 year old craft repair and a long-standing customer, reported a single vehicle collision in the early hours of the night in Columbus, SC. ... The damage was described as major damage. ... A police report was yes. ..."* Exact money amounts become descriptive bands (low, moderate, high, very high) so that text models can read them. Policy numbers, dates, ZIP codes and the label are not included.
- **Other datasets:** UCI Fraud Detection in Insurance and ISO ClaimSearch were not used. ISO ClaimSearch is proprietary and not publicly downloadable.

Because the narratives are generated from structured fields, they are templated and repeat the same wording across claims. The results therefore show how well the text pipeline learns from these fields, not how it would handle free-form human-written narratives.

## 5. Engineered Signals

| Signal | What it captures | Why it may indicate fraud |
|---|---|---|
| Repeated trigram / sentence ratio | Copy-pasted or repetitive phrasing | Templated or coached stories |
| Negative / positive word ratio and sentiment score | Sentiment anomalies | Over-dramatic loss descriptions |
| Exclamation ratio | Emotional emphasis | Pressure to settle quickly |
| Urgency ratio | Words such as "urgent", "immediately", "maximum" | Push for fast payout |
| Vagueness ratio | Words such as "cannot remember", "nobody", "no" | Missing verifiable detail |
| Length, type-token ratio, digit ratio | Writing style and detail level | Unusually short or detail-free claims |

## 6. Technology Stack

- **Deep Learning Framework:** PyTorch
- **Transformer Model:** Hugging Face Transformers (`roberta-base`)
- **Classical ML and Text Features:** scikit-learn (TF-IDF, Logistic Regression), SciPy
- **Data Analysis:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Model Persistence:** joblib
- **Quality Assurance:** Pytest (tests in `tests/`)

## 7. Project Directory Structure

```
insurance-claim-fraud-detection/
├── LICENSE                          # MIT License
├── README.md                        # Project documentation
├── REFERENCES.md                    # Citations and dataset attribution
├── pyproject.toml                   # Package setup and CLI entry points
├── requirements.txt                 # Runtime dependencies
├── data/
│   ├── insurance_claims.csv         # Raw Kaggle dataset (1,000 claims)
│   └── claims.csv                   # Generated narratives + 0/1 fraud label
├── results/                         # Outputs from the TF-IDF run
│   ├── models/
│   │   └── tfidf_logreg.joblib      # Trained TF-IDF + LogReg model
│   ├── training/
│   │   ├── tfidf_logreg_learning_curve_curves.png
│   │   └── tfidf_logreg_learning_curve_history.csv
│   ├── evaluation/
│   │   ├── tfidf_logreg_confusion_matrix.png
│   │   ├── tfidf_logreg_confusion_matrix_normalized.png
│   │   ├── tfidf_logreg_roc_curve.png
│   │   ├── tfidf_logreg_pr_curve.png
│   │   └── tfidf_logreg_f1_threshold_curve.png
│   ├── metrics/
│   │   └── tfidf_logreg_test_metrics.json   # Machine-readable test metrics
│   ├── explainability/
│   │   ├── tfidf_top_terms.csv
│   │   └── tfidf_top_terms.png
│   └── sample_predictions/
│       └── tfidf_logreg_sample_predictions.csv
├── src/claim_fraud/                 # Reusable Python package
│   ├── cli/                         # Commands: claim-train, claim-infer
│   ├── data/                        # Loading, splitting, narrative builder, demo data
│   ├── preprocessing/               # Text cleaning and engineered signals
│   ├── models/                      # TF-IDF model bundle
│   ├── training/                    # TF-IDF and RoBERTa training
│   ├── evaluation/                  # Metrics, plots, batch inference
│   └── explainability/              # TF-IDF top terms, occlusion analysis
└── tests/                           # Unit tests
```

Running `claim-train --model roberta` adds the RoBERTa files (`roberta_*` plots, metrics and `results/models/roberta/`) to the same folders.

## 8. Installation & Environment Setup

**1. Get the project.** On the GitHub page click **Code → Download ZIP**, extract it, and open a terminal inside the extracted folder. If you use git, `git clone` the repository URL from that page instead.

**2. Set up a virtual environment (recommended)**

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
pip install -e .
```

## 9. Dataset Preparation

The raw dataset is already included at `data/insurance_claims.csv`. Generate the narrative file used for training:

```bash
python -m claim_fraud.data.build_narratives --raw data/insurance_claims.csv --out data/claims.csv
```

This writes `data/claims.csv` with two columns: `narrative` (text) and `fraud` (1 = fraudulent, 0 = genuine). To use a different dataset, give any CSV with a text column and a label column and pass `--text-col` and `--label-col` to the training commands.

## 10. Training Instructions

**TF-IDF baseline** (runs in seconds on a CPU)

```bash
claim-train --model tfidf --data data/claims.csv
# or: python -m claim_fraud.training.train_tfidf --data data/claims.csv
```

**RoBERTa fine-tuning** (a GPU is strongly recommended, for example a free Google Colab T4)

```bash
claim-train --model roberta --data data/claims.csv --epochs 4 --max-len 256
# or: python -m claim_fraud.training.train_roberta --data data/claims.csv --epochs 4 --max-len 256
```

**Key training settings**

| Setting | TF-IDF baseline | RoBERTa |
|---|---|---|
| Model | Logistic Regression (class-weighted) | `roberta-base` |
| Features / input | TF-IDF (1-2 grams, max 30,000 terms, sublinear TF) + 12 scaled signals | Tokenized text, max length 256 |
| Optimizer | L-BFGS (max 2,000 iterations) | AdamW, learning rate 2e-5, weight decay 0.01 |
| Schedule | n/a | Linear warm-up (10%) then decay |
| Batch size / epochs | n/a | 16 / 4 |
| Loss | Log loss, balanced class weights | Class-weighted cross-entropy |
| Model selection | Fixed | Best epoch on validation accuracy |
| Random seed | 42 | 42 |

## 11. Prediction & Inference

Score new claim narratives with a trained model:

```bash
# Single narrative
claim-infer --weights results/models/tfidf_logreg.joblib --text "The damage was described as major damage. The insured's hobby is chess."

# A whole CSV of claims
claim-infer --weights results/models/tfidf_logreg.joblib --source new_claims.csv --text-col narrative
```

Fifty example test-set predictions, with the actual label, predicted label and fraud probability, are in `results/sample_predictions/tfidf_logreg_sample_predictions.csv`. After a RoBERTa run, point `--weights` at `results/models/roberta`.

## 12. Test Evaluation

Evaluation runs automatically at the end of each training command on the held-out test split (200 claims, 49 fraudulent) that was never used for training or model selection. Results are written to `results/metrics/tfidf_logreg_test_metrics.json` and the plots in `results/evaluation/`.

## 13. Training and Validation Graphs

The classical model has no epochs, so its learning curve plots loss and accuracy against the share of the training set used (20% to 100%).

![TF-IDF learning curve](results/training/tfidf_logreg_learning_curve_curves.png)

- **Validation accuracy** rose from 75.0% (20% of the training data) to 85.0% (all 700 training claims).
- **Validation loss** fell from 0.574 to 0.470 and was still decreasing at the end, so more data would likely help.
- **Training accuracy** reached 86.4%, close to validation accuracy, so the model is not badly overfitting.

## 14. Independent Test Results

Test split: 200 claims, 49 fraudulent. Decision threshold 0.5.

**Overall test metrics**

| Model | Accuracy | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC | Average precision |
|---|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 82.50% | 62.50% | 71.43% | 66.67% | 0.8213 | 0.5285 |
| RoBERTa-base | Needs a GPU run (command in Section 10) | | | | | |

**Per-class test performance (TF-IDF + Logistic Regression)**

| Class | Claims | Precision | Recall | F1 |
|---|---|---|---|---|
| Genuine | 151 | 90.28% | 86.09% | 88.14% |
| Fraud | 49 | 62.50% | 71.43% | 66.67% |

**Confusion matrix (counts):** 130 genuine claims correctly passed, 21 genuine claims wrongly flagged, 14 fraudulent claims missed and 35 fraudulent claims caught.

**Evaluation visualizations**

| Normalized Confusion Matrix | ROC Curve |
|---|---|
| ![Normalized confusion matrix](results/evaluation/tfidf_logreg_confusion_matrix_normalized.png) | ![ROC curve](results/evaluation/tfidf_logreg_roc_curve.png) |

| Precision-Recall Curve | F1 vs Threshold |
|---|---|
| ![PR curve](results/evaluation/tfidf_logreg_pr_curve.png) | ![F1 threshold curve](results/evaluation/tfidf_logreg_f1_threshold_curve.png) |

For fraud detection, recall and F1 on the fraud class matter more than overall accuracy. Always predicting "genuine" would score 75.5% accuracy on this test set but catch no fraud, so the 82.50% accuracy should be read together with the 71.43% fraud recall.

## 15. Interpretability Analysis

The TF-IDF model is linear, so its largest coefficients show what drives each prediction (`results/explainability/tfidf_top_terms.csv`).

![Top terms](results/explainability/tfidf_top_terms.png)

- **Strongest fraud indicators:** "major damage" (weight 2.72), the insured's hobby being "chess" (2.61) or "cross fit" (1.67), and the presence of witnesses (0.79).
- **Strongest genuine indicators:** "total loss" (-1.36), "minor damage" (-0.77), "trivial damage" (-0.74), and hobbies such as golf (-0.68).
- **Reading the result:** the model relies on the claim's recorded severity and a few categorical details. The hobby terms in particular are patterns of this one dataset and are unlikely to be real fraud signals elsewhere, which is a reason to audit any such model before use.

## 16. Limitations & Production Readiness Disclaimer

> **Warning**
>
> This system is NOT ready for automated claim rejection. It should only be used to prioritise claims for human review.

- **Generated narratives:** the text is built from structured fields, so it is templated and does not contain the semantic inconsistencies or emotional wording of human-written claims. The repetition, sentiment and urgency signals therefore have little to work with on this dataset.
- **Small dataset:** 1,000 claims (200 in the test set, 49 fraudulent) means each metric can move several points with a different split.
- **False alarms:** 21 of 151 genuine test claims (about 14%) were flagged as fraud, which would waste reviewer time.
- **Missed fraud:** 14 of 49 fraudulent test claims (about 29%) were not flagged.
- **Dataset artefacts:** some learned terms (such as hobbies) reflect quirks of this dataset rather than genuine fraud indicators.
- **RoBERTa not yet evaluated:** the transformer comparison requires running the provided trainer on a GPU.
- **Fairness:** attributes such as occupation and hobby appear in the text and could cause unfair flagging of some groups; audit before any real use.

## 17. Future Improvements

- **Real Narratives:** train on claim descriptions written by adjusters or claimants, where RoBERTa's language understanding and the repetition and sentiment signals can show their value.
- **Run the RoBERTa Comparison:** fine-tune on a GPU and compare against the baseline on the same test split.
- **Cross-Validation:** report mean and spread over several splits to give more reliable estimates on a small dataset.
- **Explainability:** add SHAP or occlusion analysis (`explainability/occlusion.py`) so reviewers see which words drove each RoBERTa prediction.
- **Multimodal Model:** combine the narrative with the numeric claim fields (amounts, policy age, claim history) in one model.
- **Cost-Sensitive Thresholds:** choose the decision threshold from the cost of a missed fraud versus a wasted review, using the F1-threshold curve.

## 18. References & Attribution

- **Dataset:** Kaggle, Auto Insurance Claims Data (`insurance_claims.csv`). https://www.kaggle.com/datasets/buntyshah/auto-insurance-claims-data
- **RoBERTa:** Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach. arXiv:1907.11692.
- **BERT:** Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL-HLT 2019.
- **TF-IDF:** Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. Information Processing & Management, 24(5), 513-523.
- **scikit-learn:** Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.
- **Transformers library:** Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. EMNLP 2020 (System Demonstrations).

See `REFERENCES.md` for the full list.

## 19. Unit Testing

Run the automated tests covering data loading, preprocessing, the training pipeline and the command-line tools:

```bash
python -m pip install pytest
python -m pytest tests/ -v
```
