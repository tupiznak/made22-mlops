#!/bin/bash
chmod +x /minio
mkdir /s3_storage/mlbucket
MINIO_ACCESS_KEY=11111111 MINIO_SECRET_KEY=22222222 /minio server /s3_storage &
mlflow server -h 0.0.0.0 -p 5000 --default-artifact-root s3://mlbucket --backend-store-uri sqlite:///mlflow.db &
cd ml_project
dvc exp run
cd ../online_inference
run_server
