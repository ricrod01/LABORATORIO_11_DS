FROM python:3.12-slim

ARG GIT_COMMIT=unknown
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_COMMIT=${GIT_COMMIT}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /app/requirements.txt

# Los datos, resultados y registros se incorporan solo mediante volumenes.
COPY app.py /app/app.py
COPY scripts /app/scripts

EXPOSE 5000 8501
