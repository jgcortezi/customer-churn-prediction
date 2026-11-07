import json

import pytest
from sklearn.model_selection import train_test_split

from churn.config import RANDOM_STATE, REPORTS_DIR
from churn.data import split_xy
from churn.features import CATEGORICAL, NUMERIC, build_preprocessor
from churn.models import candidates
from churn.train import choose_model, evaluate


@pytest.fixture(scope="module")
def split(df):
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def test_preprocessador_ignora_categoria_nova(df):
    X, _ = split_xy(df)
    prep = build_preprocessor().fit(X)
    novo = X.head(1).copy()
    novo["PaymentMethod"] = "Pix"
    assert prep.transform(novo).shape[1] == len(prep.get_feature_names_out())


def test_features_cobrem_todas_as_colunas(df):
    X, _ = split_xy(df)
    assert sorted(NUMERIC + CATEGORICAL) == sorted(X.columns)


def test_regressao_logistica_supera_o_baseline(split):
    X_train, X_test, y_train, y_test = split
    pipes = candidates()
    lr = pipes["Regressão Logística"].fit(X_train, y_train)
    base = pipes["Baseline (classe mais comum)"].fit(X_train, y_train)
    auc_lr = evaluate(y_test, lr.predict_proba(X_test)[:, 1])["roc_auc"]
    auc_base = evaluate(y_test, base.predict_proba(X_test)[:, 1])["roc_auc"]
    assert auc_base == pytest.approx(0.5)
    assert auc_lr > 0.82


def test_escolha_prefere_o_mais_simples_no_empate():
    import pandas as pd

    cv = pd.DataFrame(
        {
            "modelo": ["Gradient Boosting", "Regressão Logística", "Random Forest"],
            "roc_auc": [0.850, 0.847, 0.80],
        }
    )
    assert choose_model(cv) == "Regressão Logística"
    cv.loc[0, "roc_auc"] = 0.90
    assert choose_model(cv) == "Gradient Boosting"


def test_metricas_salvas_batem_com_o_readme():
    """Os números do README e da página vêm deste arquivo."""
    m = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert m["modelo_escolhido"] == "Regressão Logística"
    assert round(m["teste"]["roc_auc"], 2) == 0.84
    assert round(m["teste"]["recall"], 2) == 0.78
