import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from pathlib import Path

URL = "http://localhost:8000/predict"
JSON_PATH = Path("data/Books_10k.jsonl") 


def load_sentences(n=100):
    """Load up to n sentences from a JSON file into a list."""
    df = pd.read_json(JSON_PATH, lines=True)
    if "text" not in df.columns:
        raise ValueError("JSON file must contain a 'sentence' column or key.")
    return df["text"].head(n).tolist()

def send_request(session, sentence, idx):
    """Send one POST request to the FastAPI endpoint."""
    payload = {"sentences": [sentence]}
    start = time.time()
    try:
        resp = session.post(URL, json=payload, timeout=3)
        latency = (time.time() - start) * 1000  # ms
        return idx, latency, resp.status_code
    except Exception as e:
        return idx, None, "error", str(e)

def run_benchmark(sentences, concurrency=10):
    """Run concurrent requests benchmark."""
    latencies = []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(send_request, session, sentence, i): i
                for i, sentence in enumerate(sentences)
            }

            for future in as_completed(futures):
                result = future.result()
                idx, latency, status, *extra = result
                if latency:
                    latencies.append(latency)
                print(f"[{idx}] {status} - {latency:.2f} ms")

    if not latencies:
        print("No successful requests.")
        return

    latencies.sort()
    def percentile(p): return latencies[int(len(latencies) * p) - 1]
    print("\n--- Summary ---")
    print(f"Total requests: {len(latencies)}")
    print(f"Average latency: {sum(latencies)/len(latencies):.2f} ms")
    print(f"p90: {percentile(0.90):.2f} ms")
    print(f"p95: {percentile(0.95):.2f} ms")
    print(f"p99: {percentile(0.99):.2f} ms")

if __name__ == "__main__":
    sentences = load_sentences(100)
    print(f"Loaded {len(sentences)} sentences from {JSON_PATH}")
    run_benchmark(sentences, concurrency=20)