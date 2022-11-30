import os
from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
from docker.types import Mount

PROJECT_PATH = os.environ.get('PROJECT_PATH', None)
if PROJECT_PATH is None:
    raise ModuleNotFoundError('need set PROJECT_PATH env')

default_args = {
    "owner": "airflow",
    "email": [os.environ["AIRFLOW__SMTP__SMTP_MAIL_FROM"]],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        "generate",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(1),
) as dag:
    DockerOperator(
        image="made22-mlops-hw3-generate:1.0",
        command='python /src/generate.py '
                '--features-file /data/raw/{{ ds }}/data.csv '
                '--targets-file /data/raw/{{ ds }}/target.csv',
        network_mode="bridge",
        task_id="generate",
        do_xcom_push=False,
        mount_tmp_dir=False,
        mounts=[
            Mount(source=f"{PROJECT_PATH}/data", target="/data", type='bind'),
            Mount(source=f"{PROJECT_PATH}/docker/generate", target="/src", type='bind'),
        ]
    )
