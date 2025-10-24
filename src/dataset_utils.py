"""
Dataset utilities for finding and managing YOLO dataset versions.
"""
import os
import re
from typing import Optional, List, Tuple
from pathlib import Path
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetVersionManager:
    """Manages YOLO dataset versions and finds the latest available version."""
    
    def __init__(self, datasets_root: str = "gold-datasets"):
        """
        Initialize the DatasetVersionManager.
        
        Args:
            datasets_root: Root directory containing versioned datasets
        """
        self.datasets_root = Path(datasets_root)
        self.version_pattern = re.compile(r'yolo_v(\d+)\.(\d+)\.(\d+)')
    
    def parse_version(self, version_str: str) -> Tuple[int, int, int]:
        """
        Parse version string into tuple of integers.
        
        Args:
            version_str: Version string like 'yolo_v1.2.3'
            
        Returns:
            Tuple of (major, minor, patch) version numbers
        """
        match = self.version_pattern.match(version_str)
        if not match:
            raise ValueError(f"Invalid version format: {version_str}")
        
        return tuple(map(int, match.groups()))
    
    def find_dataset_versions(self) -> List[str]:
        """
        Find all available dataset versions in the datasets root directory.
        
        Returns:
            List of dataset version directory names
        """
        if not self.datasets_root.exists():
            logger.warning(f"Datasets root directory not found: {self.datasets_root}")
            return []
        
        versions = []
        for item in self.datasets_root.iterdir():
            if item.is_dir() and self.version_pattern.match(item.name):
                versions.append(item.name)
        
        return versions
    
    def get_latest_version(self) -> Optional[str]:
        """
        Get the latest dataset version based on semantic versioning.
        
        Returns:
            Latest version directory name or None if no versions found
        """
        versions = self.find_dataset_versions()
        if not versions:
            return None
        
        # Sort versions by semantic version
        versions.sort(key=self.parse_version, reverse=True)
        return versions[0]
    
    def get_dataset_path(self, version: Optional[str] = None) -> Optional[Path]:
        """
        Get the path to a specific dataset version or the latest version.
        
        Args:
            version: Specific version to get, or None for latest
            
        Returns:
            Path to dataset directory or None if not found
        """
        if version is None:
            version = self.get_latest_version()
        
        if version is None:
            return None
        
        dataset_path = self.datasets_root / version
        if dataset_path.exists():
            return dataset_path
        
        return None
    
    def validate_dataset_structure(self, dataset_path: Path) -> bool:
        """
        Validate that a dataset has the required YOLO structure.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            True if dataset structure is valid
        """
        required_files = [
            "dataset.yaml",
            "train/labels",
            "validation/labels",
            "test/labels"
        ]
        
        for required_file in required_files:
            file_path = dataset_path / required_file
            if not file_path.exists():
                logger.error(f"Missing required path: {file_path}")
                return False
        
        return True
    
    def load_dataset_config(self, dataset_path: Path) -> dict:
        """
        Load dataset configuration from dataset.yaml.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Dataset configuration dictionary
        """
        config_path = dataset_path / "dataset.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_latest_dataset_info(self) -> dict:
        """
        Get information about the latest dataset version.
        
        Returns:
            Dictionary with dataset path, version, and config
        """
        latest_version = self.get_latest_version()
        if not latest_version:
            raise RuntimeError("No dataset versions found")
        
        dataset_path = self.get_dataset_path(latest_version)
        if not dataset_path:
            raise RuntimeError(f"Dataset path not found for version: {latest_version}")
        
        if not self.validate_dataset_structure(dataset_path):
            raise RuntimeError(f"Invalid dataset structure for version: {latest_version}")
        
        config = self.load_dataset_config(dataset_path)
        
        return {
            "version": latest_version,
            "path": str(dataset_path),
            "config": config,
            "train_path": str(dataset_path / "train"),
            "validation_path": str(dataset_path / "validation"),
            "test_path": str(dataset_path / "test")
        }


def create_sample_dataset_structure(datasets_root: str = "gold-datasets"):
    """
    Create a sample dataset structure for testing.
    
    Args:
        datasets_root: Root directory for datasets
    """
    versions = ["yolo_v1.0.0", "yolo_v1.1.0", "yolo_v2.0.0"]
    
    for version in versions:
        version_path = Path(datasets_root) / version
        version_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        for split in ["train", "validation", "test"]:
            labels_dir = version_path / split / "labels"
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a sample label file
            sample_label = labels_dir / "sample.txt"
            sample_label.write_text("0 0.5 0.5 0.2 0.2\n")
        
        # Create dataset.yaml
        dataset_config = {
            "train": f"{version}/train",
            "val": f"{version}/validation", 
            "test": f"{version}/test",
            "nc": 80,  # number of classes
            "names": ["person", "bicycle", "car"]  # class names
        }
        
        config_path = version_path / "dataset.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(dataset_config, f)
    
    logger.info(f"Created sample dataset structure in {datasets_root}")


if __name__ == "__main__":
    # Example usage
    manager = DatasetVersionManager()
    
    # Create sample structure if no datasets exist
    if not manager.find_dataset_versions():
        print("No datasets found. Creating sample structure...")
        create_sample_dataset_structure()
    
    # Find and display latest dataset
    try:
        dataset_info = manager.get_latest_dataset_info()
        print(f"Latest dataset version: {dataset_info['version']}")
        print(f"Dataset path: {dataset_info['path']}")
        print(f"Config: {dataset_info['config']}")
    except RuntimeError as e:
        print(f"Error: {e}")