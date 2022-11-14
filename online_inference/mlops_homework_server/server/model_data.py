from typing import Literal

from pydantic import BaseModel

ModelStatus = Literal['loading', 'ready']


class ModelData(BaseModel):
    model: None = None
    status: ModelStatus = 'loading'
