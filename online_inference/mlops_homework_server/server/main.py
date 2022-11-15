import asyncio
import json
import logging
from asyncio import sleep
from fastapi import FastAPI
from fastapi.exceptions import HTTPException, RequestValidationError
from starlette.responses import PlainTextResponse

from mlops_homework_server.server.model_data import ModelData, ModelFields

model = ModelData()
app = FastAPI()
logger = logging.getLogger('server')


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_, exc: RequestValidationError):
    return PlainTextResponse(exc.json(indent=None), status_code=400)


async def load_model():
    logger.info('loading model...')
    await sleep(.1)
    model.status = 'ready'
    logger.info('model loaded')


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(load_model())
    logger.info('server started')


@app.get("/")
async def root():
    return {"message": "Ready"}


@app.get("/health")
async def check_health():
    if model.status != 'ready':
        raise HTTPException(503, detail='model not loaded')


@app.post("/predict")
async def predict(fields: ModelFields):
    pass
