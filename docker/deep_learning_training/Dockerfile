FROM python:3.12-slim-bookworm

# ---- system deps ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

## 証明書設定追加(各自の証明書を./docker/certificates/certificate.crtに配置する)
COPY ./docker/certificates/certificate.crt /usr/local/share/ca-certificates/certificate.crt
RUN chmod 644 /usr/local/share/ca-certificates/certificate.crt && update-ca-certificates

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt \
    GIT_SSL_CAINFO=/etc/ssl/certs/ca-certificates.crt \
    CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
 
# ---- workdir ----
WORKDIR /app

# ---- python deps ----
COPY ./docker/deep_learning_training/requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir -r /app/requirements.txt
