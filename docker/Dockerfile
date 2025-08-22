FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/src/tensorrt/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgeos-dev \
    libmagic-dev \
    libexiv2-dev \
    libboost-all-dev \
    tensorrt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install --no-cache-dir \
    setuptools==69.5.1 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    tensorboard==2.19.0 \
    tensorboardx==2.6.2.2 \
    onnx==1.17.0 \
    onnxslim==0.1.48 \
    onnxruntime-gpu==1.20.1

RUN pip install --no-cache-dir \ 
    ultralytics==8.3.83

RUN pip install --no-cache-dir \ 
    supervisely[training]==6.73.421 \
    supervisely[model-benchmark]==6.73.421

LABEL python_sdk_version=6.73.421