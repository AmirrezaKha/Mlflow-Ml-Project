#!/usr/bin/env bash
set -e

# Vars
IMAGE=your-dockerhub-user/mlflow-trainer:latest

echo "[1/3] Building Docker image..."
docker build -t $IMAGE .

echo "[2/3] Pushing to registry..."
docker push $IMAGE

echo "[3/3] Deploying to Kubernetes..."
kubectl apply -f k8s/mlflow-pv-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/trainer-job.yaml

echo "✅ Deployment complete"
