# Logs directory for YOLO automatic retraining system

This directory contains application logs and training outputs.

## Log Files

- `training.log` - Training process logs
- `comparison.log` - Model comparison logs  
- `airflow.log` - Airflow DAG execution logs
- `mlflow.log` - MLflow tracking logs

## Log Levels

- INFO: General information about system operation
- WARNING: Non-critical issues that should be noted
- ERROR: Critical errors that require attention
- DEBUG: Detailed debugging information (when enabled)

## Log Rotation

Logs are automatically rotated to prevent disk space issues. Old logs are compressed and archived.

## Viewing Logs

```bash
# View recent training logs
tail -f logs/training.log

# Search for errors
grep ERROR logs/*.log

# View logs by date
ls -la logs/
```