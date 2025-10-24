#!/bin/bash

# YOLO Automatic Retraining Setup Script
# This script helps set up the environment and dependencies

set -e  # Exit on any error

echo "🚀 Setting up YOLO Automatic Retraining System..."
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: No virtual environment detected."
    echo "   It's recommended to use a virtual environment."
    echo ""
    echo "   Create one with:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo ""
    read -p "Continue without virtual environment? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📋 Creating .env file from template..."
    cp .env.example .env
    echo "✅ Created .env file. Please review and update it with your settings."
else
    echo "✅ .env file already exists."
fi

# Create sample dataset structure
echo "📊 Setting up sample dataset structure..."
python main.py setup-sample-data

# Initialize MLflow
echo "🔬 Initializing MLflow..."
mkdir -p mlruns
echo "✅ MLflow directory created."

# Initialize Airflow (optional)
read -p "🌪️  Initialize Apache Airflow? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Setting up Airflow..."
    export AIRFLOW_HOME=$(pwd)/airflow
    mkdir -p airflow
    
    # Initialize Airflow database
    airflow db init
    
    # Create admin user
    echo "Creating Airflow admin user..."
    airflow users create \
        --username admin \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com \
        --password admin
    
    echo "✅ Airflow initialized. Access at http://localhost:8080 (admin/admin)"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Review and update .env file with your configurations"
echo "2. Update config/training_config.yaml as needed"
echo "3. Add your actual datasets to gold-datasets/"
echo ""
echo "Quick start commands:"
echo "  python main.py check-datasets     # Check available datasets"
echo "  python main.py train             # Train a model"
echo "  python main.py run-pipeline      # Run complete pipeline"
echo ""
echo "Start MLflow UI:"
echo "  mlflow ui --host 0.0.0.0 --port 5000"
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Start Airflow:"
    echo "  airflow webserver --port 8080 &"
    echo "  airflow scheduler &"
    echo ""
fi
echo "Happy training! 🤖"