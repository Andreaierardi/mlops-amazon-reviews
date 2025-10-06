# MLOps Reviews Sentiment Prediction

How to run:
- Inference Server API
<code>
docker build -t tfidf-sentiment -f inference.dockerfile .
docker run -p 8000:8000 tfidf-sentiment   
</code>
- Model Training 
<code>
docker build -t tfidf-sentiment-train -f training.dockerfile .
docker run docker run tfidf-sentiment-train
</code>

<code>
hey -z 30s -c 20 -m POST -H "Content-Type: application/json" \
  -d '{"sentences": ["great book!", "meh", "terrible"]}' \
  http://localhost:8000/predict
<code>
## ⚡ Performance Benchmark

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
0.013 [1]   |
0.052 [497] |■■■■■■
0.092 [3379]|■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
0.131 [1740]|■■■■■■■■■■■■■■■■■■■■■
0.170 [587] |■■■■■■■
0.209 [177] |■■
0.248 [67]  |■
0.288 [28]  |
0.327 [9]   |
0.366 [2]   |
0.405 [2]   |

---

### 🔍 Network & request details

| Phase | Average | Fastest | Slowest |
|--------|----------|----------|----------|
| DNS + Dialup | 0.0000 s | 0.0133 s | 0.4051 s |
| DNS Lookup | 0.0000 s | 0.0000 s | 0.0028 s |
| Request Write | 0.0000 s | 0.0000 s | 0.0012 s |
| Response Wait | 0.0925 s | 0.0132 s | 0.4042 s |
| Response Read | 0.0001 s | 0.0000 s | 0.0076 s |

---

### ✅ Interpretation

- The model and API are stable under 20 concurrent users.  
- **p99 latency = 236 ms**, meeting the **<300 ms requirement**.  
- No failed requests; consistent throughput (~216 req/sec).  
- The deployment is suitable for local and cloud scaling (Docker-ready).