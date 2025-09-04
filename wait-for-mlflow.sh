#!/bin/sh
set -e

host="mlflow"
port="5000"
timeout=60

echo "⏳ Waiting for MLflow server at $host:$port..."

while ! nc -z "$host" "$port"; do
  timeout=$((timeout - 1))
  if [ $timeout -le 0 ]; then
    echo "MLflow server not ready after 60 seconds"
    exit 1
  fi
  echo "MLflow not ready, retrying in 1 second..."
  sleep 1
done

echo "MLflow server is ready! Starting API..."
exec "$@"