# One image for both the ingestion API and the stream worker; the compose
# file picks the command. Serving needs ONNX Runtime only, not PyTorch.
FROM python:3.13-slim

WORKDIR /srv
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY app ./app
COPY artifacts ./artifacts

ENV PYTHONUNBUFFERED=1
