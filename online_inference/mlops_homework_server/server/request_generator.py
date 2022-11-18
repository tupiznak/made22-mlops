import logging
from random import randint
from time import sleep

import click
import requests

SERV_URL = 'http://127.0.0.1:34222'

logging.basicConfig()
logger = logging.getLogger('requests generator')


def gen_random_valid_request():
    valid_body = dict(
        age=randint(20, 80),
        sex=randint(0, 1),
        cp=randint(0, 3),
        trestbps=randint(120, 200),
        chol=randint(120, 200),
        fbs=randint(0, 1),
        restecg=randint(0, 2),
        thalach=randint(100, 200),
        exang=randint(0, 1),
        oldpeak=randint(1, 6),
        slope=randint(0, 2),
        ca=randint(0, 3),
        thal=randint(0, 2),
    )
    return valid_body


@click.command
@click.option('--count', default=100, help='count of queries')
def main(count: int):
    logger.setLevel(level=logging.DEBUG)
    while True:
        if requests.get(f'{SERV_URL}/health').status_code == 200:
            break
        logger.warning('model not ready. waiting...')
        sleep(1)
    logger.info('model ready.')

    for _ in range(count):
        req = gen_random_valid_request()
        res = requests.post(f'{SERV_URL}/predict', json=req)
        logger.debug(f'response: {res.content}. request: {req}.')


if __name__ == '__main__':
    main()
