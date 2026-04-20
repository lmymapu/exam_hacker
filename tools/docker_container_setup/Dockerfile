FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget curl unzip \
    python3 python3-pip python3-dev \
    libopencv-dev \
    tesseract-ocr tesseract-ocr-deu \
    libtesseract-dev \
    qt6-base-dev qt6-tools-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Python AI stack ---
RUN pip3 install --upgrade pip && \
    pip3 install \
        paddlepaddle-gpu \
        paddleocr \
        opencv-python-headless \
        numpy pillow


WORKDIR /workspace