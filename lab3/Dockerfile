# Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

WORKDIR /app

RUN python3 iris_classification.py > report.txt

CMD cat report.txt && python3 app.py

CMD ["flask", "run", "--host=0.0.0.0"]

