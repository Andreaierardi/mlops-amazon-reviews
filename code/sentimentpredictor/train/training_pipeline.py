from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# local modules
from preprocessing.datapreparator import DataPreparator
from preprocessing.transformation import TextTfidfTransformer

# MLflow
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    if path.suffix.lower() in {".jsonl", ".json"}:
        # try line-delimited first
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")



def main(args: argparse.Namespace) -> None:
    root_path = Path(__file__).parent.parent.parent
    DATAPREP_FILE_NAME = 'dataprep_config.yaml'
    TRANSF_FILE_NAME = 'transformation_config.yaml'
    TRAINING_FILE_NAME = 'train_config.yaml'
    config_path = root_path / args.config_path
    prep_path = config_path / DATAPREP_FILE_NAME
    transf_path = config_path / TRANSF_FILE_NAME
    training_path = config_path / TRAINING_FILE_NAME

    train_file = Path(args.train_file)
    prep_config = yaml.safe_load(Path(prep_path).read_text())
    transf_config =  yaml.safe_load(Path(transf_path).read_text())
    train_config =  yaml.safe_load(Path(training_path).read_text())

    outdir: Path = Path(args.outdir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = outdir / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # =============== Data ===============
    df = load_table(train_file)

    TEXT_COL = train_config.get("text_col", "sentence")
    LABEL_COL = train_config.get("label_col", "sentiment")

    # Prepare sentences + labels
    prep = DataPreparator(prep_config)  
    df_prepared = prep.transform(df)  

    # Keep only needed columns to avoid accidental leakage
    df_prepared = df_prepared[[TEXT_COL, LABEL_COL]].dropna()

    X_train, X_val, y_train, y_val = train_test_split(
        df_prepared[TEXT_COL],
        df_prepared[LABEL_COL],
        test_size=0.2,
        random_state=train_config.get("random_state", 42),
        stratify=df_prepared[LABEL_COL],
    )

    # =============== Pipeline ===============
    pipe = Pipeline(
        steps=[
            ("tfidf", TextTfidfTransformer(transf_config)),
            ("clf", LogisticRegression(**train_config.get('model'))),
        ]
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_val)

    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=sorted(np.unique(y_train)))

    metrics = {
        "accuracy": float(report["accuracy"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
    }

    # =============== Write local artifacts (unchanged behavior) ===============
    joblib.dump(pipe, run_dir / "pipeline.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    df_prepared.head(100).to_csv(run_dir / "preview_data.csv", index=False)

    print(f"[local] Saved model → {run_dir / 'pipeline.joblib'}")
    print(f"[local] Metrics     → {run_dir / 'metrics.json'}")

    # =============== MLflow logging & model saving ===============
    # Tracking URI & experiment from env, with safe defaults
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-training"))

    run_name = f"train_{ts}"
    with mlflow.start_run(run_name=run_name) as run:
        # ---- Params (data/tfidf/model) ----
        mlflow.log_params(
            {
                "text_col": TEXT_COL,
                "label_col": LABEL_COL,
                "explode_sentences": prep_config.get("explode_sentences", True),
                "min_text_len": prep_config.get("min_text_len", 0),
                "dropna": prep_config.get("dropna", True),
                "dedupe": prep_config.get("dedupe", True),
                # tfidf
                "tfidf_ngram_range": str(transf_config.get("ngram_range", [1, 1])),
                "tfidf_min_df": transf_config.get("min_df", 1),
                "tfidf_max_df": transf_config.get("max_df", 1.0),
                "tfidf_max_features": transf_config.get("max_features", None),
                # model
                "model_type": "LogisticRegression",
                **{f"model_{k}": v for k, v in train_config.items()},
            }
        )

        # ---- Metrics ----
        mlflow.log_metrics(metrics)
        # per-class f1 (if numeric classes 0/1/2)
        for label, stats in report.items():
            if label in {"accuracy", "macro avg", "weighted avg"}:
                continue
            try:
                mlflow.log_metric(f"class_{label}_f1", float(stats["f1-score"]))
            except Exception:
                pass

        # ---- Artifacts ----
        mlflow.log_artifact(run_dir / "metrics.json")
        mlflow.log_artifact(run_dir / "confusion_matrix.csv")

        # ---- Model (with signature & example) ----
        # Build a small input example consistent with the pipeline interface
        sample_in = pd.DataFrame({TEXT_COL: X_val.iloc[:5].tolist()})
        sample_out = pipe.predict(sample_in[TEXT_COL])
        signature = infer_signature(sample_in, sample_out)

        registered_name = train_config['mlflow']['registered_model_name']

        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=sample_in,
            pip_requirements=["-r requirements.txt"],  # ensures env parity
            registered_model_name=registered_name,
        )

        print(f"[mlflow] run_id: {run.info.run_id}")
        if registered_name:
            print(f"[mlflow] registered as: {registered_name}")

    print("Done.")

def main2(args: argparse.Namespace) -> None:
    root_path = Path(__file__).parent.parent.parent

    config_path = root_path / args.config_path
    prep_path = config_path / DATAPREP_FILE_NAME
    transf_path = config_path / TRANSF_FILE_NAME

    # --- Load YAML config properly ---
    with open(prep_path, "r", encoding="utf-8") as f:
        prep_config = yaml.safe_load(f)   # safe_load is recommended (prevents code execution)

    # --- Load YAML config properly ---
    with open(transf_path, "r", encoding="utf-8") as f:
        transf_config = yaml.safe_load(f)   # safe_load is recommended (prevents code execution)


    # 1) load data
    print('Data file path:',args.train_file)
    df = pd.read_json(args.train_file, lines=True)
    # 2) prepare sentences + labels
    prep = DataPreparator(prep_config)  # supports explode_sentences, etc.
    df_prepared = prep.transform(df)            # → columns: ['sentence', 'sentiment']


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--registered-model-name",
        type=str,
        default=None,
        help="If set, registers in MLflow Model Registry under this name.",
    )
    main(parser.parse_args())