import pytest

from dspy.datasets.dataloader import DataLoader

pytestmark = pytest.mark.extra


def test_from_pandas_preserves_integer_values_with_float_columns():
    import pandas as pd

    identifiers = [2**53 + 1, 2**63 - 1]
    df = pd.DataFrame({"id": identifiers, "score": [0.5, 1.5]}, index=[7, 7])

    examples = DataLoader().from_pandas(df, input_keys=("id",))

    assert [int(example.id) for example in examples] == identifiers
    assert all(isinstance(example.id, int) for example in examples)
    assert [example.inputs().toDict() for example in examples] == [{"id": identifier} for identifier in identifiers]
    assert [example.labels().toDict() for example in examples] == [{"score": 0.5}, {"score": 1.5}]


def test_from_pandas_preserves_selected_fields_and_column_names():
    import pandas as pd

    df = pd.DataFrame({"sample id": [2**53 + 1], "_score": [0.5], "unused": [1.5]})

    examples = DataLoader().from_pandas(df, fields=["_score", "sample id"])

    assert list(examples[0].keys()) == ["_score", "sample id"]
    assert int(examples[0]["sample id"]) == 2**53 + 1
    assert isinstance(examples[0]["sample id"], int)
    assert examples[0]["_score"] == 0.5


@pytest.mark.parametrize("fields", [None, []])
def test_from_pandas_preserves_rows_without_columns(fields):
    import pandas as pd

    df = pd.DataFrame(index=[3, 5])

    examples = DataLoader().from_pandas(df, fields=fields)

    assert [example.toDict() for example in examples] == [{}, {}]


def test_from_pandas_empty_fields_preserves_row_count():
    import pandas as pd

    df = pd.DataFrame({"id": [1, 2]})

    examples = DataLoader().from_pandas(df, fields=[])

    assert [example.toDict() for example in examples] == [{}, {}]


def test_from_pandas_empty_dataframe():
    import pandas as pd

    assert DataLoader().from_pandas(pd.DataFrame(columns=["id", "score"])) == []
