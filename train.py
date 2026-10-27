"""`claim-train --model tfidf|roberta [options]` - options are passed to the chosen trainer."""
import argparse


def main():
    ap = argparse.ArgumentParser(description="Train a claim-fraud model", add_help=False)
    ap.add_argument("--model", choices=["tfidf", "roberta"], default="tfidf")
    known, rest = ap.parse_known_args()
    if known.model == "tfidf":
        from claim_fraud.training.train_tfidf import main as run
    else:
        from claim_fraud.training.train_roberta import main as run
    run(rest)


if __name__ == "__main__":
    main()
