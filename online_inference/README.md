Поменял структуру типа 

COPY ./ml_project ./ml_project
COPY ./online_inference ./online_inference
RUN cd ml_project && pip install --no-cache-dir --upgrade -r requirements.txt
RUN cd online_inference && pip install --no-cache-dir --upgrade -r requirements.txt

на
