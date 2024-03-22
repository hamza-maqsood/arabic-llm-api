# Use a base image with Python and CUDA support if you're utilizing GPUs
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

# Set the working directory in the Docker image
WORKDIR /app

# Install Python and pip
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Make sure python and pip commands point to python3 and pip3
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Copy the requirements.txt file into the container
COPY requirements.txt /app/requirements.txt

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . /app

# Expose the port the application runs on
EXPOSE 8888

# Set the command to run your application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8888"]
