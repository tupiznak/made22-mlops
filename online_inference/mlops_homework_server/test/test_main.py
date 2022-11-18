import json
from asyncio import sleep

import pytest
from fastapi.testclient import TestClient

from mlops_homework_server.server.main import app


@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client


def test_smoke(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Ready"}


@pytest.mark.anyio
async def test_health(client):
    response = client.get("/health")
    assert response.status_code == 503
    await sleep(.5)
    response = client.get("/health")
    assert response.status_code == 200


@pytest.mark.anyio
async def test_predict_not_valid(client):
    valid_body = dict(
        age=20,
        sex=1,
        cp=2,
        trestbps=150,
        chol=130,
        fbs=0,
        restecg=2,
        thalach=100,
        exang=0,
        oldpeak=3.3,
        slope=1,
        ca=1,
        thal=0,
    )

    response = client.post("/predict", data=json.dumps(valid_body))
    assert response.status_code == 200, response.content
    assert json.loads(response.content) == {'message': {'result': 0}}

    def check_is_invalid_field(key: str, value):
        invalid_body = valid_body.copy()
        invalid_body[key] = value
        res = client.post("/predict", data=json.dumps(invalid_body))
        assert res.status_code == 400

    check_is_invalid_field('age', 0)
    check_is_invalid_field('age', 22.2)
    check_is_invalid_field('age', 1000)

    check_is_invalid_field('sex', -1)
    check_is_invalid_field('sex', 2)

    check_is_invalid_field('cp', -1)
    check_is_invalid_field('cp', 5)

    check_is_invalid_field('trestbps', -1)
    check_is_invalid_field('trestbps', 22.2)
    check_is_invalid_field('trestbps', 300)

    check_is_invalid_field('chol', 90)
    check_is_invalid_field('chol', 122.2)
    check_is_invalid_field('chol', 700)

    check_is_invalid_field('fbs', -1)
    check_is_invalid_field('fbs', 2)

    check_is_invalid_field('restecg', -1)
    check_is_invalid_field('restecg', 3)

    check_is_invalid_field('thalach', 60)
    check_is_invalid_field('thalach', 122.2)
    check_is_invalid_field('thalach', 300)

    check_is_invalid_field('exang', -1)
    check_is_invalid_field('exang', 2)

    check_is_invalid_field('oldpeak', -1)
    check_is_invalid_field('oldpeak', 8)

    check_is_invalid_field('exang', -1)
    check_is_invalid_field('exang', 2)

    check_is_invalid_field('slope', -1)
    check_is_invalid_field('slope', 3)

    check_is_invalid_field('ca', -1)
    check_is_invalid_field('ca', 4)

    check_is_invalid_field('thal', -1)
    check_is_invalid_field('thal', 3)
