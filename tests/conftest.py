import pandas as pd
import pytest

from churn.data import load_clean


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    return load_clean()


@pytest.fixture(scope="session")
def cliente_base(df) -> dict:
    """Um cliente real da base, usado como ponto de partida nos testes."""
    return df.drop(columns=["Churn"]).iloc[0].to_dict()
