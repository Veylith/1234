# Use official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (required for some AI libraries like OpenCV/audio processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a directory for weights (if needed by your app logic, or they will download on first run)
RUN mkdir -p /app/backend/weights
RUN chmod -R 777 /app/backend

# Copy the rest of the application
COPY . .

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Command to run the application (we change port to 7860 for Hugging Face)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
