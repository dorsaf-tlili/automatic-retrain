# Makefile for YOLO Automatic Retraining System

.PHONY: help install setup clean train compare report pipeline mlflow airflow test

help: ## Show this help message
	@echo "YOLO Automatic Retraining System"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install Python dependencies
	pip install -r requirements.txt

setup: ## Run full setup including sample data
	./setup.sh

clean: ## Clean up generated files and artifacts
	rm -rf mlruns/
	rm -rf airflow/logs/
	rm -rf models/train_*/
	rm -rf logs/*.log
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Dataset management
setup-data: ## Create sample dataset structure
	python main.py setup-sample-data

check-data: ## Check available dataset versions  
	python main.py check-datasets

# Model training and management
train: ## Train YOLO model manually
	python main.py train --run-name "manual_$(shell date +%Y%m%d_%H%M%S)"

compare: ## Compare models and update champion
	python main.py compare

report: ## Generate model performance report
	python main.py report --output model_report.json
	@echo "Report saved to model_report.json"

pipeline: ## Run complete retraining pipeline
	python main.py run-pipeline

# Services
mlflow: ## Start MLflow UI
	mlflow ui --host 0.0.0.0 --port 5000

airflow-init: ## Initialize Airflow database and create admin user
	export AIRFLOW_HOME=$(PWD)/airflow && \
	airflow db init && \
	airflow users create \
		--username admin \
		--firstname Admin \
		--lastname User \
		--role Admin \
		--email admin@example.com \
		--password admin

airflow: ## Start Airflow webserver and scheduler
	@echo "Starting Airflow services..."
	@echo "Webserver: http://localhost:8080 (admin/admin)"
	export AIRFLOW_HOME=$(PWD)/airflow && \
	airflow webserver --port 8080 --daemon && \
	airflow scheduler --daemon

airflow-stop: ## Stop Airflow services
	pkill -f "airflow webserver" || true
	pkill -f "airflow scheduler" || true

# Development and testing
test: ## Run basic functionality tests
	python -c "from src.dataset_utils import DatasetVersionManager; print('✅ Dataset utils OK')"
	python -c "import sys; sys.path.append('src'); from dataset_utils import DatasetVersionManager; print('✅ Import path OK')"
	@echo "✅ All basic tests passed"

lint: ## Run code formatting and linting
	black src/ --check
	flake8 src/
	isort src/ --check-only

format: ## Format code
	black src/
	isort src/

# Docker support
docker-setup: ## Run Docker setup script
	./docker-setup.sh

docker-build: ## Build Docker images
	docker compose build

docker-build-prod: ## Build production Docker images
	docker compose -f docker-compose.prod.yml build

docker-dev: ## Start development environment with Docker
	docker compose --profile development up -d

docker-prod: ## Start production environment with Docker
	docker compose -f docker-compose.prod.yml up -d

docker-minimal: ## Start minimal Docker environment (MLflow + app)
	docker compose up -d yolo-app mlflow

docker-stop: ## Stop all Docker services
	docker compose down
	docker compose -f docker-compose.prod.yml down

docker-clean: ## Stop and remove all Docker resources
	docker compose down -v --remove-orphans
	docker compose -f docker-compose.prod.yml down -v --remove-orphans
	docker system prune -f

docker-logs: ## Show Docker container logs
	docker compose logs -f

docker-train: ## Run training in Docker container
	docker compose exec yolo-app python main.py train

docker-shell: ## Open shell in Docker container
	docker compose exec yolo-app /bin/bash

# Monitoring and logs
logs: ## Show recent logs
	tail -f logs/*.log

monitor: ## Monitor system resources during training
	watch -n 2 'nvidia-smi; echo ""; ps aux | grep python | head -5'

# Backup and restore
backup: ## Backup MLflow experiments and models
	@echo "Creating backup..."
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz mlruns/ models/champion/ config/
	@echo "Backup created: backup_$(shell date +%Y%m%d_%H%M%S).tar.gz"

# Quick start
quick-start: install setup-data ## Quick start for new users
	@echo ""
	@echo "🎉 Quick start completed!"
	@echo ""
	@echo "Next steps:"
	@echo "  make train          # Train your first model"
	@echo "  make mlflow         # Start MLflow UI"
	@echo "  make airflow        # Start Airflow (optional)"
	@echo ""