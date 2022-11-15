from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, validator, conint, confloat
from pydantic.fields import ModelField

ModelStatus = Literal['loading', 'ready']


@dataclass
class Sex:
    Male: int = 1
    Female: int = 0


@dataclass
class ChestPainType:
    TypicalAngina: int = 0
    AtypicalAngina: int = 1
    NonAnginalPain: int = 2
    Asymptomatic: int = 3


@dataclass
class RestingElectrocardiographic:
    Normal: int = 0
    HavingSTTWaveAbnormality: int = 1
    ProbableOrDefiniteLeftVentricularHypertrophy: int = 2


@dataclass
class ExerciseInducedAngina:
    Yes: int = 1
    No: int = 0


@dataclass
class Slope:
    Upsloping: int = 0
    Flat: int = 1
    Downsloping: int = 2


@dataclass
class Thal:
    Normal: int = 0
    FixedDefect: int = 1
    ReversableDefect: int = 2


@dataclass
class Condition:
    NoDisease: int = 0
    Disease: int = 1


class ModelFields(BaseModel):
    age: conint(ge=10, le=100, strict=True)
    sex: int
    cp: int
    trestbps: conint(ge=90, le=220, strict=True)
    chol: conint(ge=100, le=600, strict=True)
    fbs: conint(ge=0, le=1, strict=True)
    restecg: int
    thalach: conint(ge=70, le=220, strict=True)
    exang: int
    oldpeak: confloat(ge=0, le=7)
    slope: int
    ca: conint(ge=0, le=3, strict=True)
    thal: int

    @validator('sex', 'cp', 'restecg', 'exang', 'thal', 'slope')
    def in_category(cls, v, values, **kwargs):
        def check_in_cat(obj_type, val: int):
            if val not in obj_type().__dict__.values():
                raise ValueError(f'{obj_type} not in exist category')

        field: ModelField = kwargs['field']
        match field:
            case ModelField(name='sex'):
                check_in_cat(Sex, v)
            case ModelField(name='cp'):
                check_in_cat(ChestPainType, v)
            case ModelField(name='restecg'):
                check_in_cat(RestingElectrocardiographic, v)
            case ModelField(name='exang'):
                check_in_cat(ExerciseInducedAngina, v)
            case ModelField(name='thal'):
                check_in_cat(Thal, v)
            case ModelField(name='slope'):
                check_in_cat(Slope, v)


class ModelData(BaseModel):
    model: None = None
    status: ModelStatus = 'loading'
