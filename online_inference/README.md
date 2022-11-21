
Команды для докера:
Запустить скрипт run.sh из текущей папки
(docker run --name made22-mlops -p34222:34222 -eKAGGLE_USERNAME="<name>" -eKAGGLE_KEY="<key>" -v"<storage>":/s3_storage --rm ajdioawd21e/mlops:1.0)

Поменял структуру типа:

COPY ./ml_project ./ml_project
COPY ./online_inference ./online_inference
RUN cd ml_project && pip install --no-cache-dir --upgrade -r requirements.txt
RUN cd online_inference && pip install --no-cache-dir --upgrade -r requirements.txt

на структуру:

COPY ./ml_project/requirements.txt /requirements1.txt
COPY ./online_inference/requirements.txt /requirements2.txt
RUN pip install --no-cache-dir --upgrade -r requirements1.txt -r requirements2.txt 

Тем самым все зависимости ставятся первыми слоями, что позволяет не перебилдивать долгий процесс каждый раз.

Также просмотрел ставящиеся пакеты - удалил tangled-up-in-unicode, который весит 1.5ГБ - непонятно, для кого он был нужен:
RUN pip install --no-cache-dir --upgrade -r requirements1.txt -r requirements2.txt &&  \
    yes | pip uninstall tangled-up-in-unicode

Все зависимости ставятся с ключом --no-cache-dir, что позволяет уменьшить размер за счёт удаления исходных архивов пакетов

Странный костыль с dvc нужен для корректного запуска dvc для последующего обучения модели
RUN git init \
    && git config --global user.email "you@example.com" \
    && git config --global user.name "Your Name"  \
    && git add --all  \
    && git commit -am '-' \
    && dvc init

