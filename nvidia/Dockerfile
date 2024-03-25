FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    unzip \
    pkg-config \
    libssl-dev \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust and update to the latest stable version
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y \
    && /root/.cargo/bin/rustup update \
    && /root/.cargo/bin/rustup default stable

# Validate Rust installation by showing its version
RUN /root/.cargo/bin/rustc --version

# Install Protobuf Compiler
ENV PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP \
    && unzip -o $PROTOC_ZIP -d /usr/local bin/protoc \
    && unzip -o $PROTOC_ZIP -d /usr/local 'include/*' \
    && rm -f $PROTOC_ZIP

RUN ln -s /usr/bin/python3 /usr/bin/python

# Set CUDA and PATH environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$PATH:/usr/local/cuda/bin

WORKDIR /app

RUN git clone https://github.com/saad039/text-generation-inference

WORKDIR /app/text-generation-inference
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install packaging torch torchvision torchaudio
RUN nvcc --version
# Since 'auto-gptq' and 'flash-attn' were causing issues, they've been removed for now. Adjust as needed.
RUN CUDA_HOME=/usr/local/cuda pip install optimum bitsandbytes

ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV BUILD_EXTENSIONS=True
RUN make install

WORKDIR /app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set default environment variables for the model, port, and hostname
ENV MODEL_ID="brainiac-origin/jais-chat-30b-8bit"
ENV PORT=8888
ENV HOSTNAME=0.0.0.0

ENV QUANTIZE="bitsandbytes"
ENV SHARDED="false"

ENTRYPOINT ["/app/entrypoint.sh"]
