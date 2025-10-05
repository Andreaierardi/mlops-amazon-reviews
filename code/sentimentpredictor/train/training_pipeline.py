# train.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse, joblib, json
import numpy as np
import pandas as pd
import yaml
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from preprocessing.datapreparator import DataPreparator          
from preprocessing.transformation import TextTfidfTransformer    

def main(args):
    root_path = Path(__file__).parent.parent.parent

    config_path = root_path / args.config_path
    prep_path = config_path / 'dataprep_config.yaml'
    transf_path = config_path / 'transformation_config.yaml'

    # --- Load YAML config properly ---
    with open(prep_path, "r", encoding="utf-8") as f:
        prep_config = yaml.safe_load(f)   # safe_load is recommended (prevents code execution)

    # --- Load YAML config properly ---
    with open(transf_path, "r", encoding="utf-8") as f:
        transf_config = yaml.safe_load(f)   # safe_load is recommended (prevents code execution)


    # 1) load data
    print('Data file path:', args.train_file)
    df = pd.read_json(args.train_file, lines=True)
    # 2) prepare sentences + labels
    prep = DataPreparator(prep_config)  # supports explode_sentences, etc.
    df_prepared = prep.transform(df)            # → columns: ['sentence', 'sentiment']

    TEXT_COL, LABEL_COL = "sentence", "sentiment"

    X_train, X_val, y_train, y_val = train_test_split(
        df_prepared[TEXT_COL], df_prepared[LABEL_COL], test_size=0.2, random_state=42, stratify=df_prepared[LABEL_COL]
    )

    X = df_prepared[TEXT_COL]
    y = df_prepared[LABEL_COL]

    # 3) build end-to-end pipeline (featureizer + classifier)
    pipe = Pipeline([
        ("tfidf", TextTfidfTransformer(transf_config)),
        ("clf", LogisticRegression(
            solver="saga", max_iter=2000, class_weight="balanced", n_jobs=-1, C=1.0
        )),
    ])

    # 4) (optional) hold-out validation
    # For simplicity assume a provided dev set; else do train_test_split here
    metrics = {}

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)
    
    print(pipe.score(X_val, y_val))
    print("\nValidation report:\n", classification_report(y_val, preds))
    metrics["val_report"] = classification_report(y_val, preds, output_dict=True)

    pipe.fit(X, y)

    # 5) save artifacts

    # --- create timestamped output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.outdir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, run_dir / "pipeline.joblib")
    with open(run_dir / "training_meta.json", "w") as f:
        json.dump({
            "config_path": str(args.config_path),
            "n_train": int(len(X_train)),
            "classes": [0,1,2],
            "model": "logreg_saga_tfidf",
            "metrics": metrics,
        }, f, indent=2)
    print(f"\nSaved model to {run_dir / 'pipeline.joblib'}")

    df_prepared.to_csv(run_dir / "data_prep.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train-file", type=str, required=True)
    p.add_argument("--config-path", type=str, required=True)

    p.add_argument("--outdir", type=Path, default=Path("artifacts"))
    main(p.parse_args())