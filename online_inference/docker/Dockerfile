FROM python:3.10

RUN wget -q https://dl.minio.io/server/minio/release/linux-amd64/minio  \
    && chmod +x /minio  \
    && mkdir -p /s3_storage/mlbucket

COPY ./ml_project/requirements.txt /requirements1.txt
COPY ./online_inference/requirements.txt /requirements2.txt
RUN pip install --no-cache-dir --upgrade -r requirements1.txt -r requirements2.txt &&  \
    yes | pip uninstall tangled-up-in-unicode

WORKDIR /usr/src/app
COPY ./ml_project ./ml_project

RUN cd ml_project && pip install --no-cache-dir --upgrade -e .

ENV PROJECT_PATH=/usr/src/app/ml_project/
VOLUME ["/s3_storage"]

RUN git init \
    && git config --global user.email "you@example.com" \
    && git config --global user.name "Your Name"  \
    && git add --all  \
    && git commit -am '-' \
    && dvc init

COPY ./online_inference ./online_inference
RUN cd online_inference && pip install --no-cache-dir --upgrade -e .

RUN chmod +x ./online_inference/docker/entrypoint.sh
ENTRYPOINT ["./online_inference/docker/entrypoint.sh"]
