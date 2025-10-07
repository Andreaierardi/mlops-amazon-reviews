from __future__ import annotations
import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Local imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from preprocessing.datapreparator import DataPreparator
from preprocessing.transformation import TextTfidfTransformer


# ============================================================
# Logging setup
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("training-pipeline")


# ============================================================
# Utility Functions
# ============================================================

def load_yaml_config(path: Path) -> dict:
    """Safely load a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        logger.error(f"Config file not found: {path}")
        raise FileNotFoundError(f"Config file not found: {path}")
    logger.info(f"Loading configuration: {path}")
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.debug(f"Loaded keys: {list(cfg.keys())}")
    return cfg


def load_table(path: Path) -> pd.DataFrame:
    """Load CSV or JSON/JSONL file into a DataFrame."""
    path = Path(path)
    logger.info(f"Loading dataset from: {path}")
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        try:
            df = pd.read_json(path, lines=True)
        except ValueError:
            df = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported data format: {path.suffix}")
    logger.info(f"Loaded {len(df):,} rows")
    return df


def prepare_data(df: pd.DataFrame, prep_config: dict, text_col: str, label_col: str) -> pd.DataFrame:
    """Run data preparation pipeline and return cleaned DataFrame."""
    logger.info("Preparing data...")
    prep = DataPreparator(prep_config)
    df_prepared = prep.transform(df)
    #df_prepared = df_prepared[[text_col, label_col]].dropna()
    logger.info(f"Prepared {len(df_prepared):,} rows after preprocessing")
    return df_prepared


def build_pipeline(transf_config: dict, model_config: dict) -> Pipeline:
    """Build sklearn pipeline for text TF-IDF + Logistic Regression."""
    logger.info("Building training pipeline (TF-IDF + LogisticRegression)")
    pipe = Pipeline([
        ("tfidf", TextTfidfTransformer(transf_config)),
        ("clf", LogisticRegression(**model_config)),
    ])
    return pipe


def evaluate_model(pipe: Pipeline, X_val, y_val, y_train=None) -> tuple[dict, np.ndarray, dict]:
    """Evaluate model and return metrics, confusion matrix, and report."""
    logger.info("Evaluating model...")
    y_pred = pipe.predict(X_val)
    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    if y_train is None or y_train.empty:
        cm = pd.Series()
    else:
        cm = confusion_matrix(y_val, y_pred, labels=sorted(np.unique(y_train)))
    
    metrics = {
        "accuracy": float(report["accuracy"]),
        "f1_macro": float(report["macro avg"]["f1-score"]),
        "precision_macro": float(report["macro avg"]["precision"]),
        "recall_macro": float(report["macro avg"]["recall"]),
    }

    logger.info(
        "Evaluation complete — "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"F1_macro: {metrics['f1_macro']:.4f}"
    )
    return metrics, cm, report


def save_local_artifacts(run_dir: Path, pipe: Pipeline, metrics: dict, cm: np.ndarray, df_prepared: pd.DataFrame):
    """Save model, metrics, and data preview locally."""
    logger.info(f"Saving artifacts to {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, run_dir / "pipeline.joblib")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savetxt(run_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    df_prepared.head(100).to_csv(run_dir / "preview_data.csv", index=False)
    logger.info("Artifacts saved successfully.")


def log_to_mlflow(pipe: Pipeline, X_val, metrics, report, prep_config, transf_config, train_config, run_dir, ts):
    """Log parameters, metrics, and model to MLflow."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-training")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info(f"Logging to MLflow → URI: {tracking_uri}, Experiment: {experiment_name}")

    run_name = f"train_{ts}"
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")

        # Parameters
        mlflow.log_params({
            "explode_sentences": prep_config.get("explode_sentences", True),
            "min_text_len": prep_config.get("min_text_len", 0),
            "tfidf_ngram_range": str(transf_config.get("ngram_range", [1, 1])),
            "tfidf_min_df": transf_config.get("min_df", 1),
            "tfidf_max_df": transf_config.get("max_df", 1.0),
            "tfidf_max_features": transf_config.get("max_features", None),
            "model_type": "LogisticRegression",
            **{f"model_{k}": v for k, v in train_config.items()},
        })

        # Metrics
        mlflow.log_metrics(metrics)
        for label, stats in report.items():
            if label not in {"accuracy", "macro avg", "weighted avg"}:
                try:
                    mlflow.log_metric(f"class_{label}_f1", float(stats["f1-score"]))
                except Exception:
                    pass

        # Artifacts
        mlflow.log_artifact(run_dir / "metrics.json")
        mlflow.log_artifact(run_dir / "confusion_matrix.csv")

        # Model
        TEXT_COL = train_config.get("text_col", "sentence")
        sample_in = pd.DataFrame({TEXT_COL: X_val.iloc[:5].tolist()})
        sample_out = pipe.predict(sample_in[TEXT_COL])
        signature = infer_signature(sample_in, sample_out)
        registered_name = train_config.get("mlflow", {}).get("registered_model_name")

        mlflow.sklearn.log_model(
            sk_model=pipe,
            name="model",
            signature=signature,
            input_example=sample_in,
            pip_requirements=["-r requirements.txt"],
            registered_model_name=registered_name,
        )

        logger.info(f"Model logged to MLflow run {run.info.run_id}")
        if registered_name:
            logger.info(f"Registered model name: {registered_name}")


# ============================================================
# Main Training Pipeline
# ============================================================

def main(args: argparse.Namespace):
    """Full training workflow orchestrator."""
    logger.info("=== Starting training pipeline ===")
    root_path = Path(__file__).parent.parent.parent
    config_path = root_path / args.config_path
    logger.info(f"Using configuration directory: {config_path}")

    prep_config = load_yaml_config(config_path / "dataprep_config.yaml")
    transf_config = load_yaml_config(config_path / "transformation_config.yaml")
    train_config = load_yaml_config(config_path / "train_config.yaml")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.outdir) / f"run_{ts}"

    df = load_table(Path(args.train_file))
    text_col = train_config.get("text_col", "sentence")
    text_col_original = train_config.get("text_col_original", "text")
    label_col = train_config.get("label_col", "sentiment")

    df_prepared = prepare_data(df, prep_config, text_col, label_col)

    training_set, validation_set, training_target, validation_target = train_test_split(
        df_prepared,
        df_prepared[label_col],
        test_size=0.2,
        random_state=train_config.get("random_state", 42),
        stratify=df_prepared[label_col],
    )

    logger.info(f"Columns: {training_set.columns}")
    cols = ['asin', 'user_id']
    df_val_ids_user = validation_set.merge(training_set[cols], on=cols, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)


    X_train = training_set[text_col]
    y_train = training_target
    X_val = validation_set[text_col]
    y_val = validation_target

    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"X_val shape: {X_val.shape}")
    logger.info(f"y_val shape: {y_val.shape}")


    pipe = build_pipeline(transf_config, train_config.get("model"))
    logger.info("Fitting model...")
    pipe.fit(X_train, y_train)

    metrics, cm, report = evaluate_model(pipe, X_val, y_val, y_train)
    
    save_local_artifacts(run_dir, pipe, metrics, cm, df_prepared)
    log_to_mlflow(pipe, X_val, metrics, report, prep_config, transf_config, train_config, run_dir, ts)

    ### Results on the original text ###
    logger.info("=== Computing results on original dataset ===")

    prep_config_orig = prep_config.copy()
    prep_config_orig['filters']['explode_sentences'] = False

    df_prepared_orig = prepare_data(df, prep_config_orig, text_col_original, label_col)
    logger.info("Filtering original by selected row for validation")   
    df_prepared_orig = df_prepared_orig.merge(df_val_ids_user[cols], on=cols, how='inner')

    X_val_orig = df_prepared_orig[text_col_original]
    y_val_orig = df_prepared_orig[label_col]

    metrics_orig, cm_orig, report_orig = evaluate_model(pipe, X_val_orig, y_val_orig)
    
    save_local_artifacts(run_dir, pipe, metrics_orig, cm_orig, X_val_orig)
    log_to_mlflow(pipe, X_val_orig, metrics_orig, report_orig, prep_config_orig, transf_config, train_config, run_dir, ts)
    
    logger.info("✅ Training pipeline completed successfully.")


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    main(args)