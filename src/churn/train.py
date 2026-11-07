"""Treina, compara e avalia os modelos. Rode com: python -m churn.train

Etapas:
1. Separa 20% da base para teste (estratificado), que só é usado no final.
2. Compara os modelos com validação cruzada de 5 dobras na parte de treino.
3. Escolhe o de maior ROC-AUC médio; em empate técnico, fica o mais simples.
4. Treina o escolhido no treino inteiro e mede uma única vez no teste.
5. Salva o modelo, as métricas, os gráficos e atualiza a página do projeto.
"""

import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from churn import plots, site
from churn.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE
from churn.data import load_clean, split_xy
from churn.models import candidates
from churn.simulator import fit_simulator

# Diferença de ROC-AUC abaixo disso é considerada empate (é menor que o desvio entre dobras).
TIE_TOLERANCE = 0.005
# Ordem de preferência em caso de empate: do mais simples para o mais complexo.
SIMPLICITY = ["Regressão Logística", "Random Forest", "Gradient Boosting"]
THRESHOLD = 0.5


def compare_models(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall": "recall",
        # zero_division=0: o baseline não prevê nenhum churn e a precisão fica indefinida.
        "precision": make_scorer(precision_score, zero_division=0),
        "f1": "f1",
    }
    rows = []
    for name, pipe in candidates().items():
        res = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
        row = {"modelo": name}
        for key in scoring:
            row[key] = res[f"test_{key}"].mean()
            row[f"{key}_std"] = res[f"test_{key}"].std()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)


def choose_model(cv_results: pd.DataFrame) -> str:
    """Maior ROC-AUC; entre os empatados com o melhor, o mais simples."""
    best = cv_results["roc_auc"].max()
    tied = cv_results.loc[cv_results["roc_auc"] >= best - TIE_TOLERANCE, "modelo"].tolist()
    return min(tied, key=lambda m: SIMPLICITY.index(m) if m in SIMPLICITY else len(SIMPLICITY))


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float = THRESHOLD) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    # Se o time de retenção ligar só para os 20% de maior risco, quantos churns ele alcança?
    top = np.argsort(-proba)[: int(len(proba) * 0.2)]
    captured_top20 = y_true.to_numpy()[top].sum() / y_true.sum()
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "recall": recall_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "churn_capturado_top20pct": float(captured_top20),
    }


def churn_rates(df: pd.DataFrame) -> dict:
    """Taxas de churn usadas no README e na página."""
    bins = [-1, 6, 12, 24, 48, 72]
    labels = ["0–6m", "7–12m", "13–24m", "25–48m", "49–72m"]
    tenure_band = pd.cut(df["tenure"], bins=bins, labels=labels)
    return {
        "base": df["Churn"].mean(),
        "clientes": len(df),
        "contrato": df.groupby("Contract")["Churn"].mean().to_dict(),
        "pagamento": df.groupby("PaymentMethod")["Churn"].mean().to_dict(),
        "tempo_de_casa": df.groupby(tenure_band, observed=True)["Churn"].mean().to_dict(),
    }


def main() -> dict:
    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    print("1/4 Comparando modelos com validação cruzada (5 dobras)...")
    cv_results = compare_models(X_train, y_train)
    print(cv_results[["modelo", "roc_auc", "roc_auc_std", "pr_auc", "recall", "f1"]].round(3))

    chosen = choose_model(cv_results)
    print(f"2/4 Modelo escolhido: {chosen}")
    all_models = candidates()
    test_scores = {}
    fitted = {}
    for name, pipe in all_models.items():
        if name.startswith("Baseline"):
            continue
        fitted[name] = clone(pipe).fit(X_train, y_train)
        test_scores[name] = evaluate(y_test, fitted[name].predict_proba(X_test)[:, 1])
    final = fitted[chosen]
    metrics = test_scores[chosen]
    print(
        f"3/4 Teste: ROC-AUC {metrics['roc_auc']:.3f} · recall {metrics['recall']:.1%}"
        f" · precisão {metrics['precision']:.1%}"
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, MODELS_DIR / "churn_model.joblib")

    sim = fit_simulator(X_train, y_train, X_test, y_test)
    rates = churn_rates(df)
    report = {
        "gerado_em": date.today().isoformat(),
        "dados": {"clientes": len(df), "treino": len(X_train), "teste": len(X_test)},
        "modelo_escolhido": chosen,
        "validacao_cruzada": cv_results.to_dict(orient="records"),
        "teste": metrics,
        "teste_todos_os_modelos": test_scores,
        "simulador": {"roc_auc_teste": sim["roc_auc_teste"], "variaveis": sim["variaveis"]},
        "taxas_de_churn": rates,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )

    print("4/4 Gerando gráficos e atualizando a página...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plots.save_all(df, final, X_test, y_test, metrics, FIGURES_DIR)
    site.update(rates, cv_results, test_scores, chosen, sim)
    print("Pronto: models/churn_model.joblib, reports/metrics.json, reports/figures/, index.html")
    return report


if __name__ == "__main__":
    main()
