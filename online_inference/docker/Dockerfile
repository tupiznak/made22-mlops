FROM python:3.10

WORKDIR /usr/src/app

COPY ../.. .

RUN pip install --no-cache-dir --upgrade -r online_interface/requirements.txt

CMD ["run_server"]
