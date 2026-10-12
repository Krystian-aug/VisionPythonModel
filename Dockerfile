
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Biblioteki systemowe wymagane przez OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libxcb1 \
       libgl1 \
       libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch w wersji CPU
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
