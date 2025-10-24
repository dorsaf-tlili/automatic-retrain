"""
Model comparison and champion selection for automatic retraining.
"""
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

import mlflow
from mlflow.tracking import MlflowClient

from yolo_trainer import YOLOTrainer
from dataset_utils import DatasetVersionManager

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelComparator:
    """Handles model comparison and champion selection logic."""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        """
        Initialize the model comparator.
        
        Args:
            config_path: Path to training configuration file
        """
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.mlflow_client = MlflowClient()
        
        # Set MLflow tracking URI
        mlflow_uri = self.config.get("mlflow", {}).get("tracking_uri", "mlruns")
        mlflow.set_tracking_uri(mlflow_uri)
        
        self.experiment_name = self.config.get("mlflow", {}).get("experiment_name", "yolo_automatic_retrain")
        self.champion_model_path = Path(self.config.get("models_dir", "models")) / "champion"
        self.champion_model_path.mkdir(parents=True, exist_ok=True)
        
        # Metric weights for model comparison (higher weight = more important)
        self.metric_weights = self.config.get("champion_selection", {}).get("metric_weights", {
            "mAP@0.5:0.95": 0.4,
            "mAP@0.5": 0.3,
            "precision": 0.15,
            "recall": 0.15
        })
        
        # Minimum improvement threshold to replace champion
        self.min_improvement_threshold = self.config.get("champion_selection", {}).get(
            "min_improvement_threshold", 0.01
        )
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            return self.get_default_config()
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for model comparison."""
        return {
            "models_dir": "models",
            "mlflow": {
                "tracking_uri": "mlruns",
                "experiment_name": "yolo_automatic_retrain"
            },
            "champion_selection": {
                "metric_weights": {
                    "mAP@0.5:0.95": 0.4,
                    "mAP@0.5": 0.3,
                    "precision": 0.15,
                    "recall": 0.15
                },
                "min_improvement_threshold": 0.01,
                "primary_metric": "mAP@0.5:0.95",
                "evaluation_dataset": "test"
            }
        }
    
    def calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate a composite score based on weighted metrics.
        
        Args:
            metrics: Dictionary of metric names to values
            
        Returns:
            Composite score (0-1 range)
        """
        total_weight = 0
        weighted_sum = 0
        
        for metric_name, weight in self.metric_weights.items():
            if metric_name in metrics:
                weighted_sum += metrics[metric_name] * weight
                total_weight += weight
            else:
                logger.warning(f"Metric '{metric_name}' not found in model metrics")
        
        if total_weight == 0:
            logger.warning("No valid metrics found for scoring")
            return 0.0
        
        return weighted_sum / total_weight
    
    def get_experiment_runs(self, limit: int = 100) -> List[mlflow.entities.Run]:
        """
        Get recent runs from the MLflow experiment.
        
        Args:
            limit: Maximum number of runs to retrieve
            
        Returns:
            List of MLflow runs
        """
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                logger.warning(f"Experiment '{self.experiment_name}' not found")
                return []
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=limit,
                output_format="list"
            )
            
            return runs
        
        except Exception as e:
            logger.error(f"Error retrieving experiment runs: {e}")
            return []
    
    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """
        Get metrics for a specific MLflow run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary of metrics
        """
        try:
            run = self.mlflow_client.get_run(run_id)
            return run.data.metrics
        except Exception as e:
            logger.error(f"Error retrieving metrics for run {run_id}: {e}")
            return {}
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Dictionary with run information
        """
        try:
            run = self.mlflow_client.get_run(run_id)
            metrics = run.data.metrics
            params = run.data.params
            
            return {
                "run_id": run_id,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000),
                "end_time": datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else None,
                "status": run.info.status,
                "metrics": metrics,
                "params": params,
                "composite_score": self.calculate_composite_score(metrics),
                "artifact_uri": run.info.artifact_uri
            }
        except Exception as e:
            logger.error(f"Error retrieving run info for {run_id}: {e}")
            return {}
    
    def find_best_model(self, exclude_run_ids: List[str] = None) -> Optional[Dict[str, Any]]:
        """
        Find the best performing model from recent experiments.
        
        Args:
            exclude_run_ids: List of run IDs to exclude from consideration
            
        Returns:
            Dictionary with information about the best model
        """
        exclude_run_ids = exclude_run_ids or []
        runs = self.get_experiment_runs()
        
        if not runs:
            logger.warning("No runs found in experiment")
            return None
        
        best_run = None
        best_score = -1
        
        for run in runs:
            if run.info.run_id in exclude_run_ids:
                continue
            
            if run.info.status != "FINISHED":
                continue
            
            run_info = self.get_run_info(run.info.run_id)
            if not run_info or not run_info.get("metrics"):
                continue
            
            score = run_info["composite_score"]
            if score > best_score:
                best_score = score
                best_run = run_info
        
        return best_run
    
    def load_champion_info(self) -> Optional[Dict[str, Any]]:
        """
        Load information about the current champion model.
        
        Returns:
            Dictionary with champion model information
        """
        champion_info_path = self.champion_model_path / "champion_info.json"
        
        if not champion_info_path.exists():
            logger.info("No champion model found")
            return None
        
        try:
            with open(champion_info_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading champion info: {e}")
            return None
    
    def save_champion_info(self, champion_info: Dict[str, Any]) -> None:
        """
        Save champion model information.
        
        Args:
            champion_info: Dictionary with champion model information
        """
        champion_info_path = self.champion_model_path / "champion_info.json"
        
        try:
            with open(champion_info_path, 'w') as f:
                json.dump(champion_info, f, indent=2, default=str)
            logger.info(f"Champion info saved to {champion_info_path}")
        except Exception as e:
            logger.error(f"Error saving champion info: {e}")
    
    def copy_model_artifacts(self, run_id: str, destination: Path) -> bool:
        """
        Copy model artifacts from MLflow run to destination.
        
        Args:
            run_id: MLflow run ID
            destination: Destination path for artifacts
            
        Returns:
            True if successful
        """
        try:
            # Download model artifacts
            artifact_path = "model"
            local_artifact_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=artifact_path
            )
            
            # Copy to destination
            if Path(local_artifact_path).exists():
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(local_artifact_path, destination)
                logger.info(f"Model artifacts copied to {destination}")
                return True
            
        except Exception as e:
            logger.error(f"Error copying model artifacts: {e}")
        
        return False
    
    def compare_and_update_champion(self, new_run_id: str) -> Dict[str, Any]:
        """
        Compare a new model with the current champion and update if better.
        
        Args:
            new_run_id: Run ID of the new model to evaluate
            
        Returns:
            Dictionary with comparison results
        """
        logger.info(f"Comparing new model (run: {new_run_id}) with champion")
        
        # Get new model info
        new_model_info = self.get_run_info(new_run_id)
        if not new_model_info:
            return {"success": False, "error": "Could not retrieve new model info"}
        
        new_score = new_model_info["composite_score"]
        
        # Load current champion info
        current_champion = self.load_champion_info()
        
        comparison_result = {
            "new_model": {
                "run_id": new_run_id,
                "score": new_score,
                "metrics": new_model_info["metrics"]
            },
            "champion_updated": False,
            "improvement": 0.0
        }
        
        if current_champion is None:
            # No existing champion, new model becomes champion
            logger.info("No existing champion found. New model becomes champion.")
            champion_updated = True
            comparison_result["improvement"] = new_score
        else:
            # Compare with existing champion
            current_score = current_champion.get("composite_score", 0)
            improvement = new_score - current_score
            
            comparison_result["current_champion"] = {
                "run_id": current_champion.get("run_id"),
                "score": current_score,
                "metrics": current_champion.get("metrics", {})
            }
            comparison_result["improvement"] = improvement
            
            # Check if new model is significantly better
            champion_updated = improvement > self.min_improvement_threshold
            
            if champion_updated:
                logger.info(f"New model is better! Improvement: {improvement:.4f}")
            else:
                logger.info(f"Current champion is still better. Improvement needed: {improvement:.4f}")
        
        if champion_updated:
            # Update champion
            champion_info = new_model_info.copy()
            champion_info["promoted_at"] = datetime.now().isoformat()
            champion_info["composite_score"] = new_score
            
            # Copy model artifacts
            model_destination = self.champion_model_path / "model"
            if self.copy_model_artifacts(new_run_id, model_destination):
                # Save champion info
                self.save_champion_info(champion_info)
                comparison_result["champion_updated"] = True
                logger.info(f"Champion model updated with run {new_run_id}")
            else:
                logger.error("Failed to copy model artifacts")
                comparison_result["success"] = False
                comparison_result["error"] = "Failed to copy model artifacts"
        
        return comparison_result
    
    def evaluate_latest_training(self) -> Dict[str, Any]:
        """
        Evaluate the latest training run and compare with champion.
        
        Returns:
            Dictionary with evaluation results
        """
        runs = self.get_experiment_runs(limit=1)
        
        if not runs:
            return {"success": False, "error": "No training runs found"}
        
        latest_run = runs[0]
        return self.compare_and_update_champion(latest_run.info.run_id)
    
    def get_champion_model_path(self) -> Optional[Path]:
        """
        Get the path to the current champion model.
        
        Returns:
            Path to champion model or None if no champion exists
        """
        champion_info = self.load_champion_info()
        if champion_info is None:
            return None
        
        model_path = self.champion_model_path / "model"
        if model_path.exists():
            # Look for .pt file in the model directory
            pt_files = list(model_path.glob("*.pt"))
            if pt_files:
                return pt_files[0]  # Return first .pt file found
        
        return None
    
    def generate_model_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive report about models and champion selection.
        
        Returns:
            Dictionary with model report
        """
        champion_info = self.load_champion_info()
        recent_runs = self.get_experiment_runs(limit=10)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "champion_model": champion_info,
            "recent_runs": []
        }
        
        for run in recent_runs:
            run_info = self.get_run_info(run.info.run_id)
            if run_info:
                report["recent_runs"].append({
                    "run_id": run_info["run_id"],
                    "start_time": run_info["start_time"].isoformat(),
                    "composite_score": run_info["composite_score"],
                    "key_metrics": {
                        metric: run_info["metrics"].get(metric, 0)
                        for metric in self.metric_weights.keys()
                        if metric in run_info["metrics"]
                    }
                })
        
        return report


def run_champion_selection(config_path: str = "config/training_config.yaml") -> Dict[str, Any]:
    """
    Convenience function to run champion selection on the latest training.
    
    Args:
        config_path: Path to training configuration
        
    Returns:
        Champion selection results
    """
    comparator = ModelComparator(config_path)
    return comparator.evaluate_latest_training()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare models and select champion")
    parser.add_argument("--config", default="config/training_config.yaml", help="Configuration file")
    parser.add_argument("--run-id", help="Specific run ID to compare")
    parser.add_argument("--report", action="store_true", help="Generate model report")
    
    args = parser.parse_args()
    
    comparator = ModelComparator(args.config)
    
    if args.report:
        report = comparator.generate_model_report()
        print(json.dumps(report, indent=2, default=str))
    elif args.run_id:
        result = comparator.compare_and_update_champion(args.run_id)
        print(json.dumps(result, indent=2, default=str))
    else:
        result = comparator.evaluate_latest_training()
        print(json.dumps(result, indent=2, default=str))