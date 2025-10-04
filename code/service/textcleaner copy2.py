import re, string, html

def clean_text(t: str) -> str:
    if not isinstance(t, str): return ""
    t = html.unescape(t).lower()
    t = re.sub(r"http\S+|www\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    t = re.sub(r"\s+", " ", t).strip()
    return t