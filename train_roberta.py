"""Fine-tune RoBERTa for claim-narrative fraud classification.

Usage (GPU strongly recommended - e.g. Google Colab / Kaggle T4):
    python src/train_roberta.py --synthetic --epochs 3
    python src/train_roberta.py --data data/claims.csv --epochs 4 --max-len 256

Long narratives: texts longer than --max-len tokens are truncated (head). For very
long logs, increase --max-len (RoBERTa supports at most 512 tokens).
"""
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, os.path.dirname(__file__))
import data as D
from evaluate import plot_curves, save_report


class ClaimDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.enc = tok(list(texts), truncation=True, padding="max_length",
                       max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.enc["input_ids"][i],
                "attention_mask": self.enc["attention_mask"][i],
                "labels": self.labels[i]}


def run_epoch(model, loader, device, loss_fn, optimizer=None, scheduler=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    probs, preds = [], []
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
            loss = loss_fn(logits, batch["labels"])
            if train:
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
            p = torch.softmax(logits, dim=-1)[:, 1]
            pred = logits.argmax(-1)
            total_loss += loss.item() * len(pred)
            correct += (pred == batch["labels"]).sum().item(); n += len(pred)
            probs.extend(p.detach().cpu().numpy()); preds.extend(pred.cpu().numpy())
    return total_loss / n, correct / n, np.array(preds), np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--text-col", default="narrative")
    ap.add_argument("--label-col", default="fraud"); ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256); ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    torch.manual_seed(D.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    df = D.make_synthetic_dataset() if args.synthetic else D.load_csv(args.data, args.text_col, args.label_col)
    train, val, test = D.split(df)

    tok = AutoTokenizer.from_pretrained(args.model)
    mk = lambda f: DataLoader(ClaimDataset(f["narrative"], f["fraud"], tok, args.max_len),
                              batch_size=args.batch_size, shuffle=False)
    tr_loader = DataLoader(ClaimDataset(train["narrative"], train["fraud"], tok, args.max_len),
                           batch_size=args.batch_size, shuffle=True)
    va_loader, te_loader = mk(val), mk(test)

    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=train["fraud"].values)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tr_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)

    hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val, best_state = -1, None
    for ep in range(1, args.epochs + 1):
        tl, ta, _, _ = run_epoch(model, tr_loader, device, loss_fn, optimizer, scheduler)
        vl, va, _, _ = run_epoch(model, va_loader, device, loss_fn)
        for k, v in zip(hist, (tl, vl, ta, va)):
            hist[k].append(v)
        print(f"epoch {ep}: train_loss={tl:.4f} train_acc={ta:.4f} val_loss={vl:.4f} val_acc={va:.4f}")
        if va > best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    plot_curves(hist, "roberta", args.out)
    _, _, pred, prob = run_epoch(model, te_loader, device, loss_fn)
    metrics = save_report("roberta", test["fraud"].values, pred, prob, test["narrative"].values, args.out)
    print({k: round(v, 4) for k, v in metrics.items()})

    os.makedirs("models/roberta", exist_ok=True)
    model.save_pretrained("models/roberta"); tok.save_pretrained("models/roberta")


if __name__ == "__main__":
    main()
