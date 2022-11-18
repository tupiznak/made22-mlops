FROM python:3.10

WORKDIR /usr/src/app

COPY . .
RUN cd ml_project && pip install --no-cache-dir --upgrade -r requirements.txt
RUN cd online_inference && pip install --no-cache-dir --upgrade -r requirements.txt

CMD ["run_server"]
