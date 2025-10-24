"""
Main entry point for the YOLO automatic retraining system.
Provides a command-line interface for running individual components.
"""

import argparse
import sys
from pathlib import Path
import logging
import json

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from dataset_utils import DatasetVersionManager, create_sample_dataset_structure
from yolo_trainer import YOLOTrainer, train_with_latest_dataset
from model_comparator import ModelComparator, run_champion_selection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_sample_data(args):
    """Create sample dataset structure for testing."""
    logger.info("Creating sample dataset structure...")
    create_sample_dataset_structure(args.datasets_root)
    logger.info("Sample dataset structure created successfully!")


def check_datasets(args):
    """Check available dataset versions."""
    manager = DatasetVersionManager(args.datasets_root)
    versions = manager.find_dataset_versions()
    
    if not versions:
        print("No dataset versions found.")
        return
    
    print(f"Found {len(versions)} dataset versions:")
    for version in sorted(versions, key=manager.parse_version, reverse=True):
        print(f"  - {version}")
    
    latest = manager.get_latest_version()
    if latest:
        print(f"\nLatest version: {latest}")
        
        try:
            dataset_info = manager.get_latest_dataset_info()
            print(f"Dataset path: {dataset_info['path']}")
            print(f"Configuration: {json.dumps(dataset_info['config'], indent=2)}")
        except Exception as e:
            print(f"Error loading dataset info: {e}")


def train_model(args):
    """Train a YOLO model."""
    logger.info("Starting YOLO model training...")
    
    trainer = YOLOTrainer(args.config)
    results = trainer.train_model(
        dataset_version=args.dataset_version,
        run_name=args.run_name
    )
    
    print(f"\nTraining completed successfully!")
    print(f"Run ID: {results['run_id']}")
    print(f"Model path: {results['model_path']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Dataset version: {results['dataset_version']}")
    
    if results['metrics']:
        print("\nKey metrics:")
        for metric, value in results['metrics'].items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")


def compare_models(args):
    """Compare models and update champion."""
    logger.info("Running model comparison...")
    
    comparator = ModelComparator(args.config)
    
    if args.run_id:
        result = comparator.compare_and_update_champion(args.run_id)
    else:
        result = comparator.evaluate_latest_training()
    
    print(json.dumps(result, indent=2, default=str))
    
    if result.get('champion_updated'):
        print("\n🏆 New champion model selected!")
    else:
        print("\n✅ Current champion remains the best.")


def generate_report(args):
    """Generate model performance report."""
    logger.info("Generating model report...")
    
    comparator = ModelComparator(args.config)
    report = comparator.generate_model_report()
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {args.output}")
    else:
        print(json.dumps(report, indent=2, default=str))


def run_full_pipeline(args):
    """Run the complete retraining pipeline."""
    logger.info("Starting full retraining pipeline...")
    
    # Step 1: Check datasets
    logger.info("Step 1: Checking dataset availability...")
    manager = DatasetVersionManager(args.datasets_root)
    
    try:
        dataset_info = manager.get_latest_dataset_info()
        logger.info(f"Using dataset version: {dataset_info['version']}")
    except Exception as e:
        logger.error(f"No valid dataset found: {e}")
        return
    
    # Step 2: Train model
    logger.info("Step 2: Training model...")
    results = train_with_latest_dataset(args.config)
    logger.info(f"Training completed. Run ID: {results['run_id']}")
    
    # Step 3: Compare with champion
    logger.info("Step 3: Comparing with champion...")
    comparison_result = run_champion_selection(args.config)
    
    if comparison_result.get('champion_updated'):
        logger.info("🏆 New champion model selected!")
    else:
        logger.info("Current champion remains the best.")
    
    # Step 4: Generate report
    logger.info("Step 4: Generating final report...")
    comparator = ModelComparator(args.config)
    report = comparator.generate_model_report()
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*50)
    print(f"Training Run ID: {results['run_id']}")
    print(f"Champion Updated: {comparison_result.get('champion_updated', False)}")
    print(f"Improvement: {comparison_result.get('improvement', 0):.4f}")
    print("="*50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YOLO Automatic Retraining System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py setup-sample-data
  python main.py check-datasets
  python main.py train --run-name "manual_training"
  python main.py compare --run-id "abc123"
  python main.py report --output report.json
  python main.py run-pipeline
        """
    )
    
    parser.add_argument(
        '--config', 
        default='config/training_config.yaml',
        help='Path to training configuration file'
    )
    
    parser.add_argument(
        '--datasets-root',
        default='gold-datasets',
        help='Root directory for datasets'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup sample data command
    setup_parser = subparsers.add_parser('setup-sample-data', help='Create sample dataset structure')
    setup_parser.set_defaults(func=setup_sample_data)
    
    # Check datasets command
    check_parser = subparsers.add_parser('check-datasets', help='Check available dataset versions')
    check_parser.set_defaults(func=check_datasets)
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument('--dataset-version', help='Specific dataset version to use')
    train_parser.add_argument('--run-name', help='Name for the MLflow run')
    train_parser.set_defaults(func=train_model)
    
    # Compare models command
    compare_parser = subparsers.add_parser('compare', help='Compare models and update champion')
    compare_parser.add_argument('--run-id', help='Specific run ID to compare')
    compare_parser.set_defaults(func=compare_models)
    
    # Generate report command
    report_parser = subparsers.add_parser('report', help='Generate model performance report')
    report_parser.add_argument('--output', help='Output file for the report')
    report_parser.set_defaults(func=generate_report)
    
    # Run full pipeline command
    pipeline_parser = subparsers.add_parser('run-pipeline', help='Run complete retraining pipeline')
    pipeline_parser.set_defaults(func=run_full_pipeline)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()