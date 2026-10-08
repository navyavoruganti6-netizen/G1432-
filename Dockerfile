FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --default-timeout=1000 -r requirements.txt

COPY main.py .
COPY pose_landmarker_full.task .
COPY Frames ./frames

RUN mkdir -p processed_frames

CMD ["python", "main.py"]
