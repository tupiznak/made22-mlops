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
