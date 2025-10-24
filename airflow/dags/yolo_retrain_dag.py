"""
Apache Airflow DAG for automatic YOLO model retraining.

This DAG orchestrates the monthly automatic retraining workflow:
1. Check for new dataset versions
2. Train YOLO model with latest data
3. Compare with champion model
4. Update champion if new model performs better
5. Send notifications about results
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule

# Add src directory to Python path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from dataset_utils import DatasetVersionManager
from yolo_trainer import YOLOTrainer
from model_comparator import ModelComparator


# DAG configuration
DEFAULT_ARGS = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=6),  # Maximum 6 hours for training
}

# DAG definition
dag = DAG(
    'yolo_automatic_retrain',
    default_args=DEFAULT_ARGS,
    description='Automatic YOLO model retraining pipeline',
    schedule_interval='0 2 1 * *',  # Monthly on 1st at 2 AM
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'yolo', 'computer-vision', 'retraining']
)


def check_dataset_availability(**context):
    """Check if new dataset versions are available."""
    try:
        # Get configuration path from Airflow Variable or use default
        config_path = Variable.get('yolo_config_path', 'config/training_config.yaml')
        datasets_root = Variable.get('datasets_root', 'gold-datasets')
        
        manager = DatasetVersionManager(datasets_root)
        latest_version = manager.get_latest_version()
        
        if not latest_version:
            raise ValueError("No dataset versions found")
        
        # Get dataset info
        dataset_info = manager.get_latest_dataset_info()
        
        # Store dataset info in XCom for downstream tasks
        context['task_instance'].xcom_push(key='dataset_version', value=latest_version)
        context['task_instance'].xcom_push(key='dataset_path', value=dataset_info['path'])
        
        print(f"Latest dataset version found: {latest_version}")
        print(f"Dataset path: {dataset_info['path']}")
        
        return {
            'version': latest_version,
            'path': dataset_info['path'],
            'config': dataset_info['config']
        }
    
    except Exception as e:
        print(f"Error checking dataset availability: {str(e)}")
        raise


def train_yolo_model(**context):
    """Train YOLO model with the latest dataset."""
    try:
        # Get dataset version from previous task
        dataset_version = context['task_instance'].xcom_pull(
            task_ids='check_dataset', key='dataset_version'
        )
        
        if not dataset_version:
            raise ValueError("No dataset version available for training")
        
        # Get configuration
        config_path = Variable.get('yolo_config_path', 'config/training_config.yaml')
        
        # Initialize trainer
        trainer = YOLOTrainer(config_path)
        
        # Create run name with timestamp
        run_name = f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"Starting training with dataset version: {dataset_version}")
        
        # Train model
        results = trainer.train_model(dataset_version=dataset_version, run_name=run_name)
        
        # Store training results in XCom
        context['task_instance'].xcom_push(key='run_id', value=results['run_id'])
        context['task_instance'].xcom_push(key='model_path', value=results['model_path'])
        context['task_instance'].xcom_push(key='metrics', value=results['metrics'])
        context['task_instance'].xcom_push(key='training_time', value=results['training_time'])
        
        print(f"Training completed successfully!")
        print(f"Run ID: {results['run_id']}")
        print(f"Training time: {results['training_time']:.2f} seconds")
        print(f"Key metrics: {results['metrics']}")
        
        return results
    
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        raise


def compare_with_champion(**context):
    """Compare newly trained model with current champion."""
    try:
        # Get run ID from training task
        run_id = context['task_instance'].xcom_pull(task_ids='train_model', key='run_id')
        
        if not run_id:
            raise ValueError("No training run ID available for comparison")
        
        # Get configuration
        config_path = Variable.get('yolo_config_path', 'config/training_config.yaml')
        
        # Initialize comparator
        comparator = ModelComparator(config_path)
        
        print(f"Comparing model from run {run_id} with current champion")
        
        # Perform comparison
        comparison_result = comparator.compare_and_update_champion(run_id)
        
        # Store comparison results in XCom
        context['task_instance'].xcom_push(key='comparison_result', value=comparison_result)
        context['task_instance'].xcom_push(
            key='champion_updated', 
            value=comparison_result.get('champion_updated', False)
        )
        
        if comparison_result.get('champion_updated', False):
            print("🏆 New champion model selected!")
            print(f"Improvement: {comparison_result.get('improvement', 0):.4f}")
        else:
            print("Current champion model remains the best")
        
        return comparison_result
    
    except Exception as e:
        print(f"Error during model comparison: {str(e)}")
        raise


def generate_training_report(**context):
    """Generate comprehensive training report."""
    try:
        # Get data from previous tasks
        run_id = context['task_instance'].xcom_pull(task_ids='train_model', key='run_id')
        metrics = context['task_instance'].xcom_pull(task_ids='train_model', key='metrics')
        training_time = context['task_instance'].xcom_pull(task_ids='train_model', key='training_time')
        comparison_result = context['task_instance'].xcom_pull(
            task_ids='compare_models', key='comparison_result'
        )
        dataset_version = context['task_instance'].xcom_pull(
            task_ids='check_dataset', key='dataset_version'
        )
        
        # Get configuration
        config_path = Variable.get('yolo_config_path', 'config/training_config.yaml')
        comparator = ModelComparator(config_path)
        
        # Generate full report
        model_report = comparator.generate_model_report()
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_run_id': run_id,
            'dataset_version': dataset_version,
            'training_time_seconds': training_time,
            'metrics': metrics,
            'comparison_result': comparison_result,
            'champion_updated': comparison_result.get('champion_updated', False),
            'improvement': comparison_result.get('improvement', 0),
            'full_model_report': model_report
        }
        
        # Store report in XCom
        context['task_instance'].xcom_push(key='training_report', value=report)
        
        print("Training report generated successfully")
        return report
    
    except Exception as e:
        print(f"Error generating training report: {str(e)}")
        raise


def cleanup_old_artifacts(**context):
    """Clean up old training artifacts to save disk space."""
    try:
        import shutil
        from pathlib import Path
        
        models_dir = Path(Variable.get('models_dir', 'models'))
        
        # Keep only the last 5 training runs
        if models_dir.exists():
            training_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('train_')]
            training_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old directories (keep latest 5)
            for old_dir in training_dirs[5:]:
                try:
                    shutil.rmtree(old_dir)
                    print(f"Cleaned up old training directory: {old_dir}")
                except Exception as e:
                    print(f"Failed to remove {old_dir}: {e}")
        
        print("Cleanup completed")
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        # Don't fail the DAG for cleanup errors
        pass


def send_success_notification(**context):
    """Send notification about successful retraining."""
    try:
        # Get report data
        report = context['task_instance'].xcom_pull(
            task_ids='generate_report', key='training_report'
        )
        
        champion_updated = report.get('champion_updated', False)
        improvement = report.get('improvement', 0)
        
        if champion_updated:
            subject = "🏆 YOLO Model - New Champion Selected!"
            message = f"""
            Great news! A new champion YOLO model has been selected.
            
            Training Details:
            - Run ID: {report.get('training_run_id')}
            - Dataset Version: {report.get('dataset_version')}
            - Training Time: {report.get('training_time_seconds', 0):.1f} seconds
            - Improvement: {improvement:.4f}
            
            Key Metrics:
            """
            
            for metric, value in report.get('metrics', {}).items():
                if isinstance(value, float):
                    message += f"            - {metric}: {value:.4f}\n"
        else:
            subject = "✅ YOLO Model - Training Completed (No Champion Update)"
            message = f"""
            YOLO model training completed successfully.
            
            Training Details:
            - Run ID: {report.get('training_run_id')}
            - Dataset Version: {report.get('dataset_version')}
            - Training Time: {report.get('training_time_seconds', 0):.1f} seconds
            
            The current champion model remains the best performer.
            Improvement needed: {improvement:.4f}
            
            Key Metrics:
            """
            
            for metric, value in report.get('metrics', {}).items():
                if isinstance(value, float):
                    message += f"            - {metric}: {value:.4f}\n"
        
        message += f"""
        
        Timestamp: {report.get('timestamp')}
        
        Check MLflow for detailed experiment tracking and model artifacts.
        """
        
        # Store notification content for email operator
        context['task_instance'].xcom_push(key='email_subject', value=subject)
        context['task_instance'].xcom_push(key='email_message', value=message)
        
        print(f"Notification prepared: {subject}")
        
    except Exception as e:
        print(f"Error preparing notification: {str(e)}")
        # Don't fail for notification errors


# Task definitions
check_dataset_task = PythonOperator(
    task_id='check_dataset',
    python_callable=check_dataset_availability,
    dag=dag,
    doc_md="Check for available dataset versions and validate structure"
)

train_model_task = PythonOperator(
    task_id='train_model',
    python_callable=train_yolo_model,
    dag=dag,
    doc_md="Train YOLO model with the latest dataset version"
)

compare_models_task = PythonOperator(
    task_id='compare_models',
    python_callable=compare_with_champion,
    dag=dag,
    doc_md="Compare newly trained model with current champion"
)

generate_report_task = PythonOperator(
    task_id='generate_report',
    python_callable=generate_training_report,
    dag=dag,
    doc_md="Generate comprehensive training and comparison report"
)

cleanup_task = PythonOperator(
    task_id='cleanup_artifacts',
    python_callable=cleanup_old_artifacts,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE,  # Run even if previous tasks fail
    doc_md="Clean up old training artifacts to save disk space"
)

prepare_notification_task = PythonOperator(
    task_id='prepare_notification',
    python_callable=send_success_notification,
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE,  # Run even if previous tasks fail
    doc_md="Prepare success notification content"
)

# Email notification task (optional - requires email configuration)
email_notification_task = EmailOperator(
    task_id='send_email_notification',
    to=Variable.get('notification_emails', '').split(',') if Variable.get('notification_emails', '') else [],
    subject="YOLO Automatic Retraining - {{ ti.xcom_pull(task_ids='prepare_notification', key='email_subject') }}",
    html_content="{{ ti.xcom_pull(task_ids='prepare_notification', key='email_message') }}",
    dag=dag,
    trigger_rule=TriggerRule.ALL_DONE
)

# Task dependencies
check_dataset_task >> train_model_task >> compare_models_task >> generate_report_task
generate_report_task >> [cleanup_task, prepare_notification_task]
prepare_notification_task >> email_notification_task

# Documentation
dag.doc_md = """
# YOLO Automatic Retraining DAG

This DAG orchestrates the automatic retraining of YOLO models on a monthly schedule.

## Workflow:
1. **Check Dataset**: Verify latest dataset version availability
2. **Train Model**: Train YOLO model with MLflow tracking
3. **Compare Models**: Compare with current champion model
4. **Generate Report**: Create comprehensive training report
5. **Cleanup**: Remove old training artifacts
6. **Notify**: Send notification about training results

## Configuration:
Set these Airflow Variables:
- `yolo_config_path`: Path to training configuration (default: config/training_config.yaml)
- `datasets_root`: Root directory for datasets (default: gold-datasets)
- `models_dir`: Directory for model outputs (default: models)
- `notification_emails`: Comma-separated email addresses for notifications

## Schedule:
Runs monthly on the 1st at 2:00 AM UTC
"""