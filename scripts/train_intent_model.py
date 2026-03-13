from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def main(args):
    df = pd.read_csv(args.csv_path)
    df = df[[args.text_col, args.label_col]].dropna()
    df[args.text_col] = df[args.text_col].astype(str)
    df[args.label_col] = df[args.label_col].astype(str)

    X_train, X_val, y_train, y_val = train_test_split(
        df[args.text_col],
        df[args.label_col],
        test_size=0.2,
        random_state=42,
        stratify=df[args.label_col],
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    preds = clf.predict(X_val_vec)
    print(classification_report(y_val, preds))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out / "vectorizer.joblib")
    joblib.dump(clf, out / "classifier.joblib")
    with (out / "labels.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(df[args.label_col].unique().tolist()), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--text_col", required=True)
    parser.add_argument("--label_col", required=True)
    parser.add_argument("--output_dir", default="models/intent_classifier")
    main(parser.parse_args())
