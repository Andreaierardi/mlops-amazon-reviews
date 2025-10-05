FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy artifacts (model + app)
COPY code ./code
COPY output ./output

ENV PYTHONPATH=/app/code/sentimentpredictor

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# python app.py
# % curl -X POST localhost:8002/predict -H 'Content-Type: application/json' \
#  -d '{"sentences":["I loved this", "meh", "terrible product", "product is ok, can be better"]}'

#docker build -t tfidf-sentiment -f inference.dockerfile .
#docker run -p 8000:8000 tfidf-sentiment   

#TRAIN
#docker build -t tfidf-sentiment-train -f training.dockerfile .