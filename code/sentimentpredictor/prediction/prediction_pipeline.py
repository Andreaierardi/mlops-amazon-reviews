# predict.py
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse, joblib
import pandas as pd
from preprocessing.datapreparator import DataPreparator          
from preprocessing.transformation import TextTfidfTransformer  

def main(args):
    pipe = joblib.load(args.artifact_path)  
    df = pd.read_csv(args.input_file)

    # the pipeline expects a DF with a 'sentence' column (or whatever your config maps)
    preds = pipe.predict(df[["sentence"]])
    df["sentiment_pred"] = preds
    if args.output_file:
        df.to_csv(args.output_file, index=False)
        print(f"Wrote predictions to {args.output_file}")
    else:
        print(df[["sentence", "pred_sentiment"]].head(20))

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-path", type=str, required=True)   # artifacts/pipeline.joblib
    p.add_argument("--input-file", type=str, required=True)
    p.add_argument("--output-file", type=str)
    main(p.parse_args())