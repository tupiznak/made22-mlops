FROM python:3.10

COPY ./ml_project/requirements.txt /requirements1.txt
COPY ./online_inference/requirements.txt /requirements2.txt
RUN pip install --no-cache-dir --upgrade -r requirements1.txt -r requirements2.txt

WORKDIR /usr/src/app
COPY ./ml_project ./ml_project
COPY ./online_inference ./online_inference

RUN cd ml_project && pip install --no-cache-dir --upgrade -e .
RUN cd online_inference && pip install --no-cache-dir --upgrade -e .

ENV PROJECT_PATH=/usr/src/app/ml_project/

RUN --mount=type=secret,id=env . /run/secrets/env \
    && git init \
    && git config --global user.email "you@example.com" \
    && git config --global user.name "Your Name"  \
    && git add --all  \
    && git commit -am '-'  \
    && dvc init  \
    && cd ml_project \
    && ls -al .  \
    && KAGGLE_USERNAME=${KAGGLE_USERNAME} KAGGLE_KEY=${KAGGLE_KEY} dvc exp run

CMD ["run_server"]
