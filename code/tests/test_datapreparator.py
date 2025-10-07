import pandas as pd
from sentimentpredictor.preprocessing.datapreparator import DataPreparator
import yaml 
from pathlib import Path

def test_datapreparator_basic_mapping_no_explode():
    # Minimal dataset
    df = pd.read_json('data/test.jsonl', lines=True)
    
    cfg = yaml.safe_load(Path('code/sentimentpredictor/config/dataprep_config.yaml').read_text())
    cfg['filters']['keep_only_sentence_and_label'] = True
    cfg['filters']['explode_sentences'] = False

    prep = DataPreparator(cfg)
    out = prep.transform(df)

    # Columns present
    assert set(out.columns) == {"sentence", "sentiment"}

    # Sizes consistent, no explosion
    assert len(out) == 3

    # Correct mapping 5->2, 3->1, 1->0
    assert out["sentiment"].tolist() == [2, 1, 0]
    print(out)
    # Sentence column mirrors original text when not exploding
    assert out["sentence"].iloc[0] == "Amazing quality and service. Very good!"

def test_datapreparator_basic_mapping_explode():
    # Minimal dataset
    df = pd.read_json('data/test.jsonl', lines=True)
    
    cfg = yaml.safe_load(Path('code/sentimentpredictor/config/dataprep_config.yaml').read_text())
    cfg['filters']['keep_only_sentence_and_label'] = True
    cfg['filters']['explode_sentences'] = True

    prep = DataPreparator(cfg)
    out = prep.transform(df)

    # Columns present
    assert set(out.columns) == {"sentence", "sentiment"}

    # Sizes consistent, no explosion
    assert len(out) == 5

    # Correct mapping 5->2, 3->1, 1->0
    assert out["sentiment"].tolist() == [2, 2, 1, 0, 0]
    print(out)
