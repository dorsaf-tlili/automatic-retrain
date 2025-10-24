# Multi-stage Dockerfile for YOLO Automatic Retraining System
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/* || \
    (apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*)

# Create app user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/gold-datasets \
             /app/models \
             /app/logs \
             /app/mlruns \
             /app/airflow/logs \
             /app/airflow/plugins && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose ports
EXPOSE 5000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app/src'); from dataset_utils import DatasetVersionManager; print('Health check OK')" || exit 1

# Default command
CMD ["python", "main.py", "--help"]


# Development stage with additional tools
FROM base as development

USER root

# Install development dependencies
RUN pip install \
    jupyter \
    notebook \
    black \
    flake8 \
    isort \
    pytest \
    pytest-cov

# Install additional system tools for development
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    tree \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Expose Jupyter port
EXPOSE 8888

# Development command
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]


# Production stage
FROM base as production

# Copy only necessary files for production
COPY --from=base /app /app

# Set production environment
ENV ENVIRONMENT=production

# Production command - start services
CMD ["python", "main.py", "run-pipeline"]


# MLflow stage - dedicated MLflow server
FROM base as mlflow

# Expose MLflow port
EXPOSE 5000

# MLflow specific environment
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=/app/mlruns

# Create MLflow directories
RUN mkdir -p /app/mlflow-data

# MLflow command
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/app/mlruns"]


# Airflow stage - dedicated Airflow services
FROM base as airflow

USER root

# Install additional dependencies for Airflow
RUN apt-get update && apt-get install -y \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

USER appuser

# Airflow specific environment variables
ENV AIRFLOW_HOME=/app/airflow
ENV AIRFLOW__CORE__DAGS_FOLDER=/app/airflow/dags
ENV AIRFLOW__CORE__LOGS_FOLDER=/app/airflow/logs
ENV AIRFLOW__CORE__PLUGINS_FOLDER=/app/airflow/plugins
ENV AIRFLOW__CORE__EXECUTOR=LocalExecutor
ENV AIRFLOW__CORE__SQL_ALCHEMY_CONN=sqlite:////app/airflow/airflow.db
ENV AIRFLOW__CORE__LOAD_EXAMPLES=False
ENV AIRFLOW__WEBSERVER__SECRET_KEY=your_secret_key_change_this

# Initialize Airflow database
RUN airflow db init

# Create admin user
RUN airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Expose Airflow port
EXPOSE 8080

# Airflow command
CMD ["sh", "-c", "airflow webserver --port 8080 --daemon && airflow scheduler"]