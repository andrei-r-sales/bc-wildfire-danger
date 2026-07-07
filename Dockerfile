FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    requests \
    scikit-learn==1.6.1 \
    joblib \
    matplotlib \
    boto3

COPY daily_extract.py storage.py ./

ENTRYPOINT ["python", "daily_extract.py"]
