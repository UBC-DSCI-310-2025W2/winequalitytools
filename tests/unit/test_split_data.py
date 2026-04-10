import pandas as pd
from wine_quality_tools import stratified_split


def test_split_size():
    df = pd.DataFrame({
        "feature": range(100),
        "quality": [0]*50 + [1]*50
    })

    train, test = stratified_split(df, "quality", test_size=0.3)

    assert len(train) == 70
    assert len(test) == 30


def test_stratification():
    df = pd.DataFrame({
        "feature": range(100),
        "quality": [0]*50 + [1]*50
    })

    train, test = stratified_split(df, "quality", test_size=0.2)

    assert train["quality"].mean() == 0.5
    assert test["quality"].mean() == 0.5


def test_invalid_column():
    df = pd.DataFrame({
        "a": [1, 2, 3]
    })

    try:
        stratified_split(df, "quality")
        assert False
    except KeyError:
        assert True