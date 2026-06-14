#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "Starting Minikube..."
minikube start --cpus=4 --memory=7000

echo "Building Docker image inside Minikube..."
eval $(minikube docker-env)
docker build -f serving/Dockerfile -t fraud-detector:latest .

echo "Applying manifests..."
kubectl delete deployment fraud-detector -n fraud-detection --ignore-not-found=true
kubectl apply -f serving/k8s/namespace.yaml
kubectl apply -f serving/k8s/configmap.yaml
kubectl apply -f serving/k8s/deployment.yaml
kubectl apply -f serving/k8s/service.yaml
kubectl apply -f serving/k8s/hpa.yaml

echo "Waiting for rollout..."
kubectl rollout status deployment/fraud-detector -n fraud-detection --timeout=120s

echo "Running smoke test..."
SERVICE_URL=$(minikube service fraud-detector-service -n fraud-detection --url)
curl -f "$SERVICE_URL/health"

echo ""
echo "Deployment complete. Service URL: $SERVICE_URL"
