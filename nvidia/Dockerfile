FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt update && apt install -y git curl wget build-essential python3 python3-pip unzip pkg-config libssl-dev openssl
RUN pkg-config openssl
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PROTOC_ZIP=protoc-21.12-linux-x86_64.zip
RUN curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP
RUN unzip -o $PROTOC_ZIP -d /usr/local bin/protoc
RUN unzip -o $PROTOC_ZIP -d /usr/local 'include/*'
RUN rm -f $PROTOC_ZIP
RUN ln -s /usr/bin/python3 /usr/bin/python

ENV CUDA_HOME=/usr/local/cuda

WORKDIR /app

RUN git clone https://github.com/saad039/text-generation-inference

WORKDIR /app/text-generation-inference
ENV PATH="/root/.cargo/bin:${PATH}"
RUN pip install packaging torch torchvision torchaudio
RUN pip install optimum auto-gptq flash-attn

ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV BUILD_EXTENSIONS=True
RUN make install

WORKDIR /app
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Set default environment variables for the model, port, and hostname
ENV MODEL_ID="brainiac-origin/jais-chat-30b-8bit"
# ENV MODEL_REVISION="gptq-4bit-128g-actorder_True"
ENV PORT=8888
ENV HOSTNAME=0.0.0.0

ENV QUANTIZE="bitsandbytes"
ENV SHARDED="false"
#ENV MAX_INPUT_LENGTH="1024"
#ENV MAX_TOTAL_TOKENS="2048"

# Copy all files in models directory to /app/models. Do not copy the models directory itself.
# COPY model/ /root/.cache/huggingface/hub


ENTRYPOINT ["/app/entrypoint.sh"]
