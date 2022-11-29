#!/bin/bash
mkdir -p "/s3_storage/mlbucket"
MINIO_ROOT_USER=11111111 MINIO_ROOT_PASSWORD=22222222 /minio server --address 0.0.0.0:9000 /storage &
MLFLOW_TRACKING_URI=http://localhost:5000 AWS_ACCESS_KEY_ID=11111111 AWS_SECRET_ACCESS_KEY=22222222 MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 mlflow server -h 0.0.0.0 -p 5000 --default-artifact-root mlflow-artifacts:/ --serve-artifacts --backend-store-uri sqlite:///mlflow.db
