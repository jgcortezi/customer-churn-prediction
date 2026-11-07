import pandas as pd
import pytest

from churn.data import clean, load_raw


def test_base_tem_o_tamanho_esperado(df):
    assert df.shape == (7043, 21)


def test_total_charges_vira_numero_sem_nulos(df):
    assert pd.api.types.is_float_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0


def test_total_charges_em_branco_vira_zero_para_cliente_novo():
    raw = load_raw()
    em_branco = raw["TotalCharges"].str.strip() == ""
    assert em_branco.sum() == 11
    assert (raw.loc[em_branco, "tenure"] == 0).all()
    assert (clean(raw).loc[em_branco.to_numpy(), "TotalCharges"] == 0).all()


def test_churn_binario_e_taxa_conhecida(df):
    assert set(df["Churn"].unique()) == {0, 1}
    assert df["Churn"].mean() == pytest.approx(0.2654, abs=1e-3)


def test_senior_citizen_vira_categoria(df):
    assert set(df["SeniorCitizen"].unique()) == {"No", "Yes"}


def test_churn_invalido_gera_erro():
    raw = load_raw().head(3).copy()
    raw.loc[0, "Churn"] = "Talvez"
    with pytest.raises(ValueError):
        clean(raw)
