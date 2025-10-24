#!/bin/bash

# Docker setup script for YOLO Automatic Retraining System
set -e

echo "🐳 YOLO Automatic Retraining - Docker Setup"
echo "==========================================="
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Docker installation
if ! command_exists docker; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check Docker Compose installation
if ! command_exists docker-compose; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check if Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"
echo ""

# Setup environment
if [ ! -f .env.docker ]; then
    echo "❌ .env.docker file not found. Creating from template..."
    cp .env.docker .env.docker
fi

# Get user choice for setup type
echo "Choose setup type:"
echo "1) Development (with Jupyter notebook)"
echo "2) Production (full services with PostgreSQL)"
echo "3) Minimal (just MLflow and basic app)"
echo ""
read -p "Enter your choice (1-3): " -n 1 -r
echo ""

case $REPLY in
    1)
        SETUP_TYPE="development"
        COMPOSE_FILE="docker-compose.yml"
        PROFILE="development"
        ;;
    2)
        SETUP_TYPE="production"
        COMPOSE_FILE="docker-compose.prod.yml"
        PROFILE=""
        ;;
    3)
        SETUP_TYPE="minimal"
        COMPOSE_FILE="docker-compose.yml"
        PROFILE=""
        ;;
    *)
        echo "Invalid choice. Defaulting to development setup."
        SETUP_TYPE="development"
        COMPOSE_FILE="docker-compose.yml"
        PROFILE="development"
        ;;
esac

echo "Selected: $SETUP_TYPE setup"
echo ""

# GPU support check
if command_exists nvidia-smi && nvidia-smi >/dev/null 2>&1; then
    echo "🎮 NVIDIA GPU detected. Do you want to enable GPU support?"
    read -p "Enable GPU support? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        GPU_SUPPORT="true"
        echo "GPU support will be enabled"
        
        # Check for nvidia-docker
        if ! command_exists nvidia-docker; then
            echo "⚠️  nvidia-docker not found. Installing nvidia-container-toolkit..."
            # Add installation instructions based on OS
            echo "Please install nvidia-container-toolkit:"
            echo "Ubuntu/Debian: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        fi
    else
        GPU_SUPPORT="false"
    fi
else
    GPU_SUPPORT="false"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p gold-datasets models logs airflow/logs airflow/plugins

# Create sample dataset if none exists
if [ ! -d "gold-datasets/yolo_v1.0.0" ]; then
    echo "📊 Creating sample dataset structure..."
    python -c "
import sys
sys.path.append('src')
from dataset_utils import create_sample_dataset_structure
create_sample_dataset_structure()
" 2>/dev/null || echo "⚠️  Will create sample data after containers are running"
fi

echo ""
echo "🏗️  Building Docker images..."

# Build images based on setup type
if [ "$SETUP_TYPE" = "production" ]; then
    docker-compose -f $COMPOSE_FILE build
else
    if [ "$PROFILE" != "" ]; then
        docker-compose -f $COMPOSE_FILE --profile $PROFILE build
    else
        docker-compose -f $COMPOSE_FILE build
    fi
fi

echo ""
echo "🚀 Starting services..."

# Start services
if [ "$SETUP_TYPE" = "production" ]; then
    # Production setup with initialization
    echo "Initializing production database..."
    docker-compose -f $COMPOSE_FILE up -d postgres mlflow-db
    sleep 10
    
    echo "Starting MLflow..."
    docker-compose -f $COMPOSE_FILE up -d mlflow
    sleep 5
    
    echo "Initializing Airflow..."
    docker-compose -f $COMPOSE_FILE run --rm airflow-webserver airflow db init
    docker-compose -f $COMPOSE_FILE run --rm airflow-webserver airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
    
    echo "Starting all services..."
    docker-compose -f $COMPOSE_FILE up -d
    
elif [ "$PROFILE" != "" ]; then
    docker-compose -f $COMPOSE_FILE --profile $PROFILE up -d
else
    docker-compose -f $COMPOSE_FILE up -d
fi

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 15

# Check service health
echo "🔍 Checking service status..."

# Check MLflow
if curl -f http://localhost:5000/health >/dev/null 2>&1; then
    echo "✅ MLflow is running at http://localhost:5000"
else
    echo "⚠️  MLflow might still be starting..."
fi

# Check Airflow (if running)
if [ "$SETUP_TYPE" != "minimal" ]; then
    if curl -f http://localhost:8080/health >/dev/null 2>&1; then
        echo "✅ Airflow is running at http://localhost:8080"
        echo "   Login: admin / admin"
    else
        echo "⚠️  Airflow might still be starting..."
    fi
fi

# Check Jupyter (if development)
if [ "$PROFILE" = "development" ]; then
    echo "✅ Jupyter notebook available at http://localhost:8888"
    echo "   Token will be shown in the logs: docker-compose logs jupyter"
fi

echo ""
echo "🎉 Docker setup completed!"
echo ""
echo "📋 Service URLs:"
echo "   MLflow UI:    http://localhost:5000"
if [ "$SETUP_TYPE" != "minimal" ]; then
    echo "   Airflow UI:   http://localhost:8080 (admin/admin)"
fi
if [ "$PROFILE" = "development" ]; then
    echo "   Jupyter:      http://localhost:8888"
fi
if [ "$SETUP_TYPE" = "production" ]; then
    echo "   Flower:       http://localhost:5555"
    echo "   Main site:    http://localhost:80"
fi
echo ""
echo "📖 Useful commands:"
echo "   docker-compose logs -f                 # View all logs"
echo "   docker-compose logs -f yolo-app        # View app logs"
echo "   docker-compose ps                      # Check service status"
echo "   docker-compose down                    # Stop all services"
echo "   docker-compose down -v                 # Stop and remove volumes"
echo ""
echo "🐳 To run training in Docker:"
echo "   docker-compose exec yolo-app python main.py train"
echo ""
echo "Happy training! 🤖"