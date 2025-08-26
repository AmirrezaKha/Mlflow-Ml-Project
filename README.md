# MLflow CI/CD with Kubernetes (kind)

This project demonstrates how to set up a **CI/CD pipeline for MLflow** using **GitHub Actions**, **Docker**, and **Kubernetes** with [kind](https://kind.sigs.k8s.io/).

---

## 🚀 Project Overview

- **MLflow** for experiment tracking and model management  
- **Docker** to build and package the MLflow app and training code  
- **kind** to create a lightweight local Kubernetes cluster in GitHub Actions  
- **Kubernetes Jobs & Deployments** to run MLflow tracking server and training jobs  
- **GitHub Actions** to automate build, deploy, and testing  

---

## 📂 Project Structure

```
.
├── k8s/
│   ├── mlflow-deployment.yaml   # Deployment for MLflow tracking server
│   ├── mlflow-service.yaml      # Service to expose MLflow
│   ├── mlflow-job.yaml          # Kubernetes Job for training
├── main.py                      # Example training script
├── Dockerfile                   # MLflow + Trainer image
├── requirements.txt             # Python dependencies
├── .github/
│   └── workflows/
│       └── mlflow-ci.yml        # GitHub Actions pipeline
└── README.md
```

---

## ⚙️ GitHub Actions Workflow

The pipeline (`.github/workflows/mlflow-ci.yml`) performs:

1. **Checkout code**
2. **Set up Docker Buildx**
3. **Install kind**
4. **Create Kubernetes cluster**
5. **Build & load MLflow image**
6. **Deploy MLflow + training job to Kubernetes**
7. **Check pod status & fetch logs**

---

## 🐳 Running Locally

### 1️⃣ Install kind & kubectl

```bash
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Install kubectl
sudo apt-get update && sudo apt-get install -y kubectl
```

### 2️⃣ Create a cluster

```bash
kind create cluster --name mlflow-cluster
```

### 3️⃣ Build and load Docker image

```bash
docker build -t mlflow-app:latest .
kind load docker-image mlflow-app:latest --name mlflow-cluster
```

### 4️⃣ Deploy resources

```bash
kubectl create namespace mlflow
kubectl apply -f k8s/mlflow-deployment.yaml -n mlflow
kubectl apply -f k8s/mlflow-service.yaml -n mlflow
kubectl apply -f k8s/mlflow-job.yaml -n mlflow
```

### 5️⃣ Get MLflow Job logs

```bash
POD=$(kubectl get pods -n mlflow -l job-name=mlflow-job -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD -n mlflow -f
```

---

## 📊 Example Training Results

- **Model**: LSTM  
- **Accuracy**: `0.96` – `1.0` depending on params  
- **Tracked in MLflow**: metrics, parameters, artifacts  

---

## 🔖 Tags (for Medium article)

- MLOps  
- Kubernetes  
- GitHub Actions  
- Machine Learning  
- DevOps  

---

## 📜 License

MIT License – free to use and modify.
