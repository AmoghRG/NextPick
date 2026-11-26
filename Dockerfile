FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir fastapi uvicorn[standard] qdrant-client redis lightgbm onnxruntime numpy pandas pyarrow

COPY src/ ./src/
COPY artifacts/ ./artifacts/

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "7860"]
