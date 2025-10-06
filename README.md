# MLOps Reviews Sentiment Prediction
This project trains and serves a **sentiment classifier** that categorizes review sentences into **positive**, **neutral**, or **negative** classes.  
Each review is split into individual sentences, and each sentence inherits the review’s star rating as its label.


## 🧠 Project Overview

The project includes:
1. **Training pipeline** to preprocess data, extract features, and train a classifier.
2. **FastAPI app** that serves predictions locally with **p99 latency ≤ 300 ms**.
3. **Containerization** for easy local or cloud deployment.


## How to run:
### Requirements
- python3.10
- docker

### Run:
- Inference Server API
```bash
docker build -t tfidf-sentiment -f inference.dockerfile .
docker run -p 8000:8000 tfidf-sentiment
```
- Test deployed server:
```bash
curl -X POST localhost:8002/predict -H 'Content-Type: application/json' \
-d '{"sentences":["I loved this", "meh", "terrible product", "product is ok, can be better"]}'

```
- Output
```json
{
  "sentiments": [
    "positive","negative","negative","neutral"
  ]
}
```

- Model Training 
```bash
docker build -t tfidf-sentiment-train -f training.dockerfile .
docker run docker run tfidf-sentiment-train
```
<br>



## ⚡ Performance Benchmark
```bash
hey -z 30s -c 20 -m POST -H "Content-Type: application/json" \
  -d '{"sentences": ["great book!", "meh", "terrible"]}' \
  http://localhost:8000/predict
```
**Load test tool:** [`hey`](https://github.com/rakyll/hey)  
**Endpoint tested:** `POST /predict`  
**Payload:** `{"sentences": ["great book!", "meh", "terrible"]}`  
**Test duration:** 30 seconds  
**Concurrency:** 20 clients  

### 📊 Summary
| Metric | Result |
|---------|---------|
| **Total duration** | 30.08 seconds |
| **Requests/sec** | 215.75 |
| **Average latency** | 92.7 ms |
| **Fastest request** | 13.3 ms |
| **Slowest request** | 405.1 ms |
| **Total responses** | 6,489 |
| **Successful (200)** | 100% |
---
### ⏱️ Latency distribution

| Percentile | Latency |
|-------------|----------|
| 10% | 54.9 ms |
| 25% | 66.8 ms |
| 50% (median) | 83.8 ms |
| 75% | 108.8 ms |
| 90% | 142.0 ms |
| 95% | 165.3 ms |
| **99% (p99)** | **236.4 ms ✅** |

> **p99 latency < 300 ms target achieved**

---

### 📈 Response time histogram
| % | Graph |
|-------------|----------|
|0.013 [1]   | <br>
|0.052 [497] |■■■■■■ <br>
|0.092 [3379]|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ <br>
|0.131 [1740]|■■■■■■■■■■■■■■■■■■■■■ <br>
|0.170 [587] |■■■■■■■ <br>
|0.209 [177] |■■ <br>
|0.248 [67]  |■ <br>
|0.288 [28]  | <br>
|0.327 [9]   | <br>
|0.366 [2]   | <br>
|0.405 [2]   | <br>

---
- **p99 latency = 236 ms**, meeting the **<300 ms requirement**.  
- No failed requests; consistent throughput (~216 req/sec).  
- The deployment is suitable for local and cloud scaling (Docker-ready).
