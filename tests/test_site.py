import json
import re

from churn.config import SITE_HTML


def _dados():
    html = SITE_HTML.read_text(encoding="utf-8")
    m = re.search(r"const DADOS = (\{.*?\});\n", html, flags=re.S)
    assert m, "bloco de dados gerados não encontrado na página"
    return json.loads(m.group(1))


def test_pagina_usa_os_numeros_do_treino():
    d = _dados()
    assert d["clientes"] == 7043
    assert d["escolhido"] == "Regressão Logística"
    assert round(d["teste"]["auc"], 2) == 0.84


def test_simulador_tem_todas_as_categorias():
    coef = _dados()["simulador"]["coef"]
    for chave in [
        "Contract_Month-to-month",
        "Contract_Two year",
        "InternetService_No",
        "PaymentMethod_Electronic check",
        "TechSupport_Yes",
        "TechSupport_No internet service",
    ]:
        assert chave in coef
