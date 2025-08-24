
### `Dockerfile`

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY run_local.sh .

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "app.main"]
