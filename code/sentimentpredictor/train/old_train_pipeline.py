# train_pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from code.sentimentpredictor.dataprocessing.utils.textcleaner import TextCleaner  # <-- import from module

# code/train_pipeline.py (snippet)
import argparse
import pandas as pd
import joblib
from pathlib import Path
# ... imports for sklearn and your TextCleaner ...

parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

print("Loading dataset")
df = pd.read_json(args.data, lines=True)
print("Dataset shape", df.shape)
print("Dataset columns", df.columns)

TEXT_COL, LABEL_COL = "text", "sentiment"

def map_sent(r): 
    return 0 if r<=2 else 1 if r==3 else 2

if LABEL_COL not in df.columns:
    df[LABEL_COL] = df["rating"].apply(map_sent)
df = df[[TEXT_COL, LABEL_COL]].dropna()

X_train, X_test, y_train, y_test = train_test_split(
    df[TEXT_COL], df[LABEL_COL], test_size=0.2, random_state=42, stratify=df[LABEL_COL]
)

print("Start training")
pipe = Pipeline([
    ("clean", TextCleaner()),
    ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])

pipe.fit(X_train, y_train)
print("Test acc:", pipe.score(X_test, y_test))
print("Classification report:\n", )

print("Saved model to:", args.output)
joblib.dump(pipe, args.output)
