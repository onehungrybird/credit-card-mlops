FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Set working directory
WORKDIR /app

# Copy dependency files
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir uvicorn

# Copy source code
COPY src ./src
COPY models ./models
COPY config ./config

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000 \
    MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
