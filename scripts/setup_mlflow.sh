#!/bin/bash
set -e

echo "=== Setting up MLflow and ML Pipeline on EC2 ==="

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-venv git curl build-essential

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Create directories
mkdir -p data/raw data/processed models/artifacts mlflow/mlruns mlflow/artifacts

# Configure firewall
sudo ufw allow 5000  # MLflow
sudo ufw allow 8000  # FastAPI

echo "Setup completed!"
echo "Next steps:"
echo "1. Upload your creditcard.csv to data/raw/"
echo "2. Run: source venv/bin/activate"
echo "3. Start MLflow: mlflow server --backend-store-uri sqlite:///mlflow/mlflow.db --default-artifact-root mlflow/artifacts --host 0.0.0.0 --port 5000"
echo "4. Train model: python -m src.models.train"
echo "5. Start API: python -m src.api.main"
