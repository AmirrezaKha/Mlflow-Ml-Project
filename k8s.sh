kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/mlflow-service.yaml
kubectl apply -f k8s/mlflow-job.yaml
kubectl port-forward svc/mlflow-service 5000:5000 -n mlflow


kubectl delete pods --all -n mlflow
kubectl delete jobs --all -n mlflow
kubectl delete all --all -n mlflow
kubectl delete job mlflow-job -n mlflow

 kubectl logs mlflow-job-7st4j -n mlflow
 kubectl port-forward svc/mlflow-service 5000:5000 -n mlflow


 
kubectl get pods -n mlflow