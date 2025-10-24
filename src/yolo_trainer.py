"""
YOLO model training with MLflow tracking and experiment management.
"""
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml

import mlflow
import mlflow.pytorch
from ultralytics import YOLO
import torch
from torch.utils.data import DataLoader

from dataset_utils import DatasetVersionManager

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOTrainer:
    """Handles YOLO model training with MLflow experiment tracking."""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """
        Initialize the YOLO trainer.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.dataset_manager = DatasetVersionManager(self.config.get("datasets_root", "gold-datasets"))
        
        # Set MLflow tracking URI
        mlflow_uri = self.config.get("mlflow", {}).get("tracking_uri", "mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        
        # Set experiment name
        experiment_name = self.config.get("mlflow", {}).get("experiment_name", "yolo_automatic_retrain")
        try:
            mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            pass  # Experiment already exists
        mlflow.set_experiment(experiment_name)
    
    def load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        if not self.config_path.exists():
            # Return default configuration
            return self.get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            "model": {
                "size": "yolov8n",  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
                "pretrained": True,
                "input_size": 640
            },
            "training": {
                "epochs": 100,
                "batch_size": 16,
                "patience": 50,
                "save_period": 10,
                "workers": 8,
                "device": "auto",  # auto, cpu, cuda:0
                "optimizer": "AdamW",
                "lr0": 0.01,
                "weight_decay": 0.0005,
                "momentum": 0.937
            },
            "augmentation": {
                "hsv_h": 0.015,
                "hsv_s": 0.7,
                "hsv_v": 0.4,
                "degrees": 0.0,
                "translate": 0.1,
                "scale": 0.5,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 1.0,
                "mixup": 0.0
            },
            "validation": {
                "conf": 0.001,
                "iou": 0.6,
                "max_det": 300,
                "save_txt": False,
                "save_conf": False
            },
            "datasets_root": "gold-datasets",
            "models_dir": "models",
            "mlflow": {
                "tracking_uri": "mlruns",
                "experiment_name": "yolo_automatic_retrain"
            }
        }
    
    def prepare_dataset_config(self, dataset_info: Dict[str, Any]) -> str:
        """
        Prepare dataset configuration for YOLO training.
        
        Args:
            dataset_info: Dataset information from DatasetVersionManager
            
        Returns:
            Path to the prepared dataset YAML file
        """
        dataset_config = dataset_info["config"].copy()
        
        # Update paths to be absolute
        dataset_path = Path(dataset_info["path"])
        dataset_config["train"] = str(dataset_path / "train")
        dataset_config["val"] = str(dataset_path / "validation")
        dataset_config["test"] = str(dataset_path / "test")
        
        # Save updated config
        temp_config_path = dataset_path / "training_dataset.yaml"
        with open(temp_config_path, 'w') as f:
            yaml.dump(dataset_config, f)
        
        return str(temp_config_path)
    
    def train_model(self, dataset_version: Optional[str] = None, run_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Train YOLO model with MLflow tracking.
        
        Args:
            dataset_version: Specific dataset version to use, or None for latest
            run_name: Name for the MLflow run
            
        Returns:
            Dictionary containing training results and model info
        """
        # Get dataset information
        if dataset_version:
            dataset_path = self.dataset_manager.get_dataset_path(dataset_version)
            if not dataset_path:
                raise ValueError(f"Dataset version not found: {dataset_version}")
            dataset_info = {
                "version": dataset_version,
                "path": str(dataset_path),
                "config": self.dataset_manager.load_dataset_config(dataset_path)
            }
        else:
            dataset_info = self.dataset_manager.get_latest_dataset_info()
        
        logger.info(f"Training with dataset version: {dataset_info['version']}")
        
        # Start MLflow run
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params(self.config["model"])
            mlflow.log_params(self.config["training"])
            mlflow.log_params(self.config["augmentation"])
            mlflow.log_param("dataset_version", dataset_info["version"])
            
            # Prepare dataset config
            dataset_config_path = self.prepare_dataset_config(dataset_info)
            
            # Initialize model
            model_size = self.config["model"]["size"]
            if self.config["model"]["pretrained"]:
                model = YOLO(f"{model_size}.pt")
            else:
                model = YOLO(f"{model_size}.yaml")
            
            # Prepare training arguments
            train_args = {
                "data": dataset_config_path,
                "epochs": self.config["training"]["epochs"],
                "batch": self.config["training"]["batch_size"],
                "imgsz": self.config["model"]["input_size"],
                "patience": self.config["training"]["patience"],
                "save_period": self.config["training"]["save_period"],
                "workers": self.config["training"]["workers"],
                "device": self.config["training"]["device"],
                "optimizer": self.config["training"]["optimizer"],
                "lr0": self.config["training"]["lr0"],
                "weight_decay": self.config["training"]["weight_decay"],
                "momentum": self.config["training"]["momentum"],
                "project": self.config["models_dir"],
                "name": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "exist_ok": True,
                "verbose": True
            }
            
            # Add augmentation parameters
            train_args.update(self.config["augmentation"])
            
            # Train the model
            logger.info("Starting model training...")
            start_time = time.time()
            
            results = model.train(**train_args)
            
            training_time = time.time() - start_time
            logger.info(f"Training completed in {training_time:.2f} seconds")
            
            # Get the best model path
            best_model_path = results.save_dir / "weights" / "best.pt"
            last_model_path = results.save_dir / "weights" / "last.pt"
            
            # Validate the model
            logger.info("Validating trained model...")
            val_results = model.val(
                data=dataset_config_path,
                conf=self.config["validation"]["conf"],
                iou=self.config["validation"]["iou"],
                max_det=self.config["validation"]["max_det"],
                save_txt=self.config["validation"]["save_txt"],
                save_conf=self.config["validation"]["save_conf"]
            )
            
            # Extract metrics
            metrics = self.extract_metrics(results, val_results)
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log additional info
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log model artifacts
            mlflow.log_artifact(str(best_model_path), "model")
            mlflow.log_artifact(str(last_model_path), "model")
            mlflow.log_artifact(dataset_config_path, "dataset")
            
            # Log model with MLflow
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="yolo_model",
                registered_model_name=f"yolo_{model_size}"
            )
            
            # Prepare result dictionary
            result_dict = {
                "run_id": mlflow.active_run().info.run_id,
                "model_path": str(best_model_path),
                "dataset_version": dataset_info["version"],
                "metrics": metrics,
                "training_time": training_time,
                "model_size": model_size,
                "results_dir": str(results.save_dir)
            }
            
            logger.info(f"Training completed. Run ID: {result_dict['run_id']}")
            return result_dict
    
    def extract_metrics(self, train_results, val_results) -> Dict[str, float]:
        """
        Extract relevant metrics from training and validation results.
        
        Args:
            train_results: Training results from YOLO
            val_results: Validation results from YOLO
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Training metrics
        if hasattr(train_results, 'results_dict'):
            results_dict = train_results.results_dict
            for key, value in results_dict.items():
                if isinstance(value, (int, float)):
                    metrics[f"train_{key}"] = float(value)
        
        # Validation metrics
        if hasattr(val_results, 'results_dict'):
            val_dict = val_results.results_dict
            for key, value in val_dict.items():
                if isinstance(value, (int, float)):
                    metrics[f"val_{key}"] = float(value)
        
        # Extract key metrics if available
        try:
            # mAP@0.5
            if hasattr(val_results, 'box') and hasattr(val_results.box, 'map50'):
                metrics["mAP@0.5"] = float(val_results.box.map50)
            
            # mAP@0.5:0.95
            if hasattr(val_results, 'box') and hasattr(val_results.box, 'map'):
                metrics["mAP@0.5:0.95"] = float(val_results.box.map)
            
            # Precision and Recall
            if hasattr(val_results, 'box'):
                if hasattr(val_results.box, 'mp'):
                    metrics["precision"] = float(val_results.box.mp)
                if hasattr(val_results.box, 'mr'):
                    metrics["recall"] = float(val_results.box.mr)
        
        except Exception as e:
            logger.warning(f"Could not extract some metrics: {e}")
        
        return metrics
    
    def load_model_from_run(self, run_id: str) -> YOLO:
        """
        Load a YOLO model from an MLflow run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Loaded YOLO model
        """
        model_uri = f"runs:/{run_id}/yolo_model"
        model = mlflow.pytorch.load_model(model_uri)
        return model


def train_with_latest_dataset(config_path: str = "config/training_config.yaml") -> Dict[str, Any]:
    """
    Convenience function to train with the latest dataset version.
    
    Args:
        config_path: Path to training configuration
        
    Returns:
        Training results dictionary
    """
    trainer = YOLOTrainer(config_path)
    run_name = f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return trainer.train_model(run_name=run_name)


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train YOLO model with MLflow tracking")
    parser.add_argument("--config", default="config/training_config.yaml", help="Training configuration file")
    parser.add_argument("--dataset-version", help="Specific dataset version to use")
    parser.add_argument("--run-name", help="Name for the MLflow run")
    
    args = parser.parse_args()
    
    trainer = YOLOTrainer(args.config)
    results = trainer.train_model(args.dataset_version, args.run_name)
    
    print(f"Training completed successfully!")
    print(f"Run ID: {results['run_id']}")
    print(f"Model path: {results['model_path']}")
    print(f"Key metrics: {results['metrics']}")