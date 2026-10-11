FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user --prefer-binary -r requirements.txt

FROM python:3.11-slim AS runner
WORKDIR /app
RUN groupadd -r mlopsuser && useradd -r -g mlopsuser mlopsuser
COPY --from=builder /root/.local /home/mlopsuser/.local
ENV PATH=/home/mlopsuser/.local/bin:$PATH
COPY config/ /app/config/
COPY src/ /app/src/
COPY models/ /app/models/
RUN chown -R mlopsuser:mlopsuser /app
USER mlopsuser
EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
