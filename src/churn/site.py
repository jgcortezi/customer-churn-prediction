"""Atualiza os números da página (index.html) com o resultado do último treino.

Tudo o que aparece na página (taxas, métricas, coeficientes do simulador) é
escrito aqui, entre os marcadores <dados-gerados>. Assim a página nunca fica
com um número diferente do que o código produziu.
"""

import json
import re

import pandas as pd

from churn.config import SITE_HTML
from churn.plots import PT_CONTRACT, PT_PAYMENT

START, END = "// <dados-gerados>", "// </dados-gerados>"
AUTOMATIC = ["Bank transfer (automatic)", "Credit card (automatic)"]


def build_data(rates: dict, cv_results: pd.DataFrame, test_scores: dict, chosen: str, sim: dict) -> dict:
    contrato = rates["contrato"]
    pagamento = rates["pagamento"]
    melhor_cartao = pagamento["Credit card (automatic)"]
    modelos = [
        {
            "m": row["modelo"],
            "auc": round(row["roc_auc"], 3),
            "rec": round(row["recall"], 3),
            "f1": round(row["f1"], 3),
            "best": bool(row["roc_auc"] == cv_results["roc_auc"].max()),
            "pick": row["modelo"] == chosen,
        }
        for _, row in cv_results.iterrows()
        if not row["modelo"].startswith("Baseline")
    ]
    teste = test_scores[chosen]
    reais = cv_results.loc[~cv_results["modelo"].str.startswith("Baseline"), "roc_auc"]
    diferenca = reais.max() - reais.min()
    nota = (
        f"Os três modelos ficaram a menos de {diferenca:.3f} de ROC-AUC entre si".replace(".", ",")
        + f", diferença menor que a variação entre as dobras. Fiquei com a <b>{chosen}</b>: "
        f"empata com os mais complexos e cada coeficiente pode ser explicado. No conjunto de teste, "
        f"que o modelo nunca viu, ela encontra <b>{teste['recall']:.0%}</b> dos cancelamentos."
    )
    return {
        "clientes": rates["clientes"],
        "taxa_base": round(rates["base"], 4),
        "contrato": [{"n": PT_CONTRACT[k], "v": round(v, 3)} for k, v in contrato.items()],
        "pagamento": sorted(
            [{"n": PT_PAYMENT[k], "v": round(v, 3)} for k, v in pagamento.items()],
            key=lambda d: -d["v"],
        ),
        "tempo": [{"n": str(k), "v": round(v, 3)} for k, v in rates["tempo_de_casa"].items()],
        "razao_contrato": contrato["Month-to-month"] / contrato["Two year"],
        "razao_pagamento": pagamento["Electronic check"] / melhor_cartao,
        "modelos": modelos,
        "escolhido": chosen,
        "teste": {"auc": teste["roc_auc"], "recall": teste["recall"], "precisao": teste["precision"]},
        "nota_modelos": nota,
        "simulador": {
            "auc": sim["roc_auc_teste"],
            "intercept": sim["intercept"],
            "coef": sim["coef"],
            "mean": sim["mean"],
            "std": sim["std"],
        },
    }


def render_block(data: dict) -> str:
    js = json.dumps(data, ensure_ascii=False, indent=1)
    return (
        f"{START} (gerado por `python -m churn.train`; não edite à mão)\n"
        f"const DADOS = {js};\n"
        "const CONTRACT = DADOS.contrato, PAYMENT = DADOS.pagamento, TENURE = DADOS.tempo;\n"
        "const MODELS = DADOS.modelos, MODEL = DADOS.simulador;\n"
        f"{END}"
    )


def update(rates, cv_results, test_scores, chosen, sim, path=SITE_HTML) -> None:
    html = path.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), flags=re.S)
    if not pattern.search(html):
        raise ValueError("Marcadores <dados-gerados> não encontrados no index.html")
    block = render_block(build_data(rates, cv_results, test_scores, chosen, sim))
    path.write_text(pattern.sub(lambda _: block, html, count=1), encoding="utf-8")
