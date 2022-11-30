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
    "owner": "qqq",
    "email": [os.environ["AIRFLOW__SMTP__SMTP_MAIL_FROM"]],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "train",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(1),
) as dag:
    waiting = PythonSensor(
        task_id='check-data',
        python_callable=lambda p: Path(p).exists(),
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        poke_interval=10
    )
    preprocess = DockerOperator(
        image="made22-mlops-hw3-train:1.0",
        command='python /src/preprocess.py '
                '--features-file-raw /data/raw/{{ ds }}/data.csv '
                '--targets-file-raw /data/raw/{{ ds }}/target.csv '
                '--features-file /data/preprocess/{{ ds }}/data.csv '
                '--targets-file /data/preprocess/{{ ds }}/target.csv',
        network_mode="bridge",
        task_id="preprocess",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{PROJECT_PATH}/data", target="/data", type='bind'),
            Mount(source=f"{PROJECT_PATH}/docker/train", target="/src", type='bind'),
        ]
    )
    split = DockerOperator(
        image="made22-mlops-hw3-train:1.0",
        command='python /src/split.py '
                '--features-file-preprocess /data/preprocess/{{ ds }}/data.csv '
                '--targets-file-preprocess /data/preprocess/{{ ds }}/target.csv '
                '--features-file-train /data/split/{{ ds }}/data_train.csv '
                '--targets-file-train /data/split/{{ ds }}/target_train.csv '
                '--features-file-test /data/split/{{ ds }}/data_test.csv '
                '--targets-file-test /data/split/{{ ds }}/target_test.csv ',
        network_mode="bridge",
        task_id="split",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{PROJECT_PATH}/data", target="/data", type='bind'),
            Mount(source=f"{PROJECT_PATH}/docker/train", target="/src", type='bind'),
        ]
    )
    train = DockerOperator(
        image="made22-mlops-hw3-train:1.0",
        command='python /src/train.py '
                '--features-file-train /data/split/{{ ds }}/data_train.csv '
                '--targets-file-train /data/split/{{ ds }}/target_train.csv '
                '--features-file-test /data/split/{{ ds }}/data_test.csv '
                '--targets-file-test /data/split/{{ ds }}/target_test.csv '
                '--file-metrics-validate /data/models/{{ ds }}/metric.json '
                '--model-path /data/models/{{ ds }}/model.pkl ',
        network_mode="host",
        task_id="train",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{PROJECT_PATH}/data", target="/data", type='bind'),
            Mount(source=f"{PROJECT_PATH}/docker/train", target="/src", type='bind'),
        ]
    )
    waiting >> preprocess >> split >> train
