import pytest

from churn.predict import predict_proba


def test_probabilidade_entre_zero_e_um(cliente_base):
    (p,) = predict_proba([cliente_base])
    assert 0 <= p <= 1


def test_contrato_de_dois_anos_reduz_o_risco(cliente_base):
    mensal = {**cliente_base, "Contract": "Month-to-month"}
    dois_anos = {**cliente_base, "Contract": "Two year"}
    p_mensal, p_dois = predict_proba([mensal, dois_anos])
    assert p_dois < p_mensal


def test_cliente_antigo_tem_menos_risco_que_novo(cliente_base):
    novo = {**cliente_base, "tenure": 1, "TotalCharges": cliente_base["MonthlyCharges"]}
    antigo = {**cliente_base, "tenure": 60, "TotalCharges": cliente_base["MonthlyCharges"] * 60}
    p_novo, p_antigo = predict_proba([novo, antigo])
    assert p_antigo < p_novo


def test_campo_faltando_gera_erro_claro(cliente_base):
    incompleto = {k: v for k, v in cliente_base.items() if k != "Contract"}
    with pytest.raises(ValueError, match="Contract"):
        predict_proba([incompleto])
