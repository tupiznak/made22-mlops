import os
from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.dates import days_ago
from docker.types import Mount

PROJECT_PATH = os.environ.get('PROJECT_PATH', None)
if PROJECT_PATH is None:
    raise ModuleNotFoundError('need set PROJECT_PATH env')

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    waiting = PythonSensor(
        task_id='check-data',
        python_callable=lambda p: Path(p).exists(),
        op_args=['/opt/airflow/data/models/{{ ds }}/model.pkl'],
        poke_interval=10
    )
    predict = DockerOperator(
        image="made22-mlops-hw3-predict:1.0",
        command='python /src/predict.py '
                '--features-file /data/raw/{{ ds }}/data.csv '
                '--targets-file /data/predictions/{{ ds }}/predictions.csv '
                '--model-path /data/models/{{ ds }}/model.pkl ',
        network_mode="host",
        task_id="prediction",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{PROJECT_PATH}/data", target="/data", type='bind'),
            Mount(source=f"{PROJECT_PATH}/docker/predict", target="/src", type='bind'),
        ]
    )

    waiting >> predict
