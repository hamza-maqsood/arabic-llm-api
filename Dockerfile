# Use a base image with Python and CUDA support if you're utilizing GPUs
FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

# Set the working directory in the Docker image
WORKDIR /app

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
