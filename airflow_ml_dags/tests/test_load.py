import pytest

from airflow.models import DagBag


@pytest.fixture()
def dagbag():
    return DagBag()


def test_generate_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="generate")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 1


def test_train_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="train")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 4


def test_predict_loaded(dagbag):
    dag = dagbag.get_dag(dag_id="predict")
    assert dagbag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 2
