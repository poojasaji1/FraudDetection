.PHONY: setup services data features train promote rollback shadow serve load-test k8s-deploy k8s-status k8s-logs drift simulate-drift k8s-monitoring test clean

setup:
	python -m pip install -r requirements.txt

services:
	docker-compose up -d

data:
	python data/validate.py

features:
	python features/feature_views.py
	cd features && feast apply
	python features/materialize.py

train:
	python training/train.py

promote:
	python registry/promote.py

rollback:
	python registry/rollback.py

shadow:
	python registry/shadow.py

serve:
	MLFLOW_TRACKING_URI=file:./mlruns \
	POSTGRES_PORT=5433 POSTGRES_DB=predictions_db POSTGRES_PASSWORD=changeme \
	python serving/server.py

load-test:
	locust -f tests/load/locustfile.py --headless -u 1000 -r 100 -t 60s --host=http://localhost:8000

k8s-deploy:
	bash scripts/deploy_k8s.sh

k8s-status:
	kubectl get pods,hpa,svc -n fraud-detection

k8s-logs:
	kubectl logs -l app=fraud-detector -n fraud-detection --tail=50

drift:
	python monitoring/drift_detector.py

simulate-drift:
	python monitoring/simulate_drift.py

k8s-monitoring:
	kubectl apply -f monitoring/cronjob.yaml
	kubectl get cronjob drift-detector -n fraud-detection

test:
	POSTGRES_PORT=5433 POSTGRES_DB=predictions_db POSTGRES_PASSWORD=changeme \
	MLFLOW_TRACKING_URI=file:./mlruns \
	python -m pytest tests/integration/ -v --tb=short

clean:
	docker compose down
	minikube stop || true
