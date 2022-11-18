import asyncio
import logging
from asyncio import sleep
from io import StringIO

import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import HTTPException, RequestValidationError
from hydra import initialize_config_dir, compose
from starlette.responses import PlainTextResponse

from mlops_homework.conf.config import PROJECT_PATH, Config
from mlops_homework.models.baseline.predict_baseline_model import predict as predict_model
from mlops_homework_server.server.model_data import ModelData, ModelFields

initialize_config_dir(version_base=None, config_dir=f'{PROJECT_PATH}mlops_homework/conf')
config: Config = compose(config_name="config")

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
    features = StringIO()
    targets = StringIO()
    df = pd.DataFrame.from_dict({k: [v] for k, v in jsonable_encoder(fields).items()})
    df.to_csv(features, index=False)
    features.seek(0)
    predict_model(features_file=features, targets_file=targets,
                  encoder_path=PROJECT_PATH + config.relative_path_to_model_encoder,
                  model_path=PROJECT_PATH + config.relative_path_to_model)
    return {'message': {'result': int(targets.getvalue().strip())}}
