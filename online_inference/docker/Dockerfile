FROM python:3.10

RUN wget https://dl.minio.io/server/minio/release/linux-amd64/minio

COPY ./ml_project/requirements.txt /requirements1.txt
COPY ./online_inference/requirements.txt /requirements2.txt
RUN pip install --no-cache-dir --upgrade -r requirements1.txt -r requirements2.txt &&  \
    yes | pip uninstall tangled-up-in-unicode

WORKDIR /usr/src/app
COPY ./ml_project ./ml_project

RUN cd ml_project && pip install --no-cache-dir --upgrade -e .

ENV PROJECT_PATH=/usr/src/app/ml_project/
ENV MLFLOW_TRACKING_URI=http://127.0.0.1:5000
#RUN apt update \
#    && apt install dvc sqlite3 \
#    && rm -rf /var/cache/apt/archives /var/lib/apt/lists/*
RUN pip install --no-cache-dir boto3
VOLUME ["/s3_storage"]
#RUN --mount=type=secret,id=env . /run/secrets/env \
##    && chmod +x /minio && useradd -ms /bin/bash minio && chown minio:minio /minio  \
#    && chmod +x /minio && MINIO_ACCESS_KEY=11111111 MINIO_SECRET_KEY=22222222 /minio server /s3_storage \
##    && mkdir /s3_storage && chown -R minio:minio /s3_storage && su minio -c "MINIO_ACCESS_KEY=11111111 MINIO_SECRET_KEY=22222222 /minio server /s3_storage" \
#    & . /run/secrets/env \
#    && mlflow server -h 0.0.0.0 -p 5000 --default-artifact-root s3://mlbucket --backend-store-uri sqlite:///mlflow.db
#    & . /run/secrets/env \
RUN git init \
    && git config --global user.email "you@example.com" \
    && git config --global user.name "Your Name"  \
    && git add --all  \
    && git commit -am '-'  \
    && dvc init
#    && cd ml_project \
#    && KAGGLE_USERNAME=${KAGGLE_USERNAME} KAGGLE_KEY=${KAGGLE_KEY} dvc exp run
#    && MINIO_ACCESS_KEY=11111111 MINIO_SECRET_KEY=22222222 /minio server /minio_dir
#COPY ./online_inference/docker/entrypoint.sh /entrypoint.sh

COPY ./online_inference ./online_inference
RUN cd online_inference && pip install --no-cache-dir --upgrade -e .
RUN chmod +x ./online_inference/docker/entrypoint.sh
ENTRYPOINT ["./online_inference/docker/entrypoint.sh"]
