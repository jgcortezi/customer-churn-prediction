"""Gráficos do relatório (reports/figures). Mesma identidade visual da página."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

INK, INK_2, MUTED, LINE = "#15130f", "#57534c", "#8a877e", "#e5e3db"
BLUE, ORANGE, TEAL = "#2a78d6", "#e6602f", "#159a6d"
GROUND = "#ffffff"

PT_CONTRACT = {"Month-to-month": "Mensal", "One year": "1 ano", "Two year": "2 anos"}
PT_PAYMENT = {
    "Electronic check": "Cheque eletrônico",
    "Mailed check": "Cheque pelo correio",
    "Bank transfer (automatic)": "Transferência automática",
    "Credit card (automatic)": "Cartão automático",
}


def _style(ax, title: str, subtitle: str | None = None) -> None:
    ax.set_facecolor(GROUND)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=LINE, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=MUTED)


def _bar_rates(ax, rates: pd.Series, title: str, subtitle: str) -> None:
    x = np.arange(len(rates))
    ax.bar(x, rates.values * 100, color=ORANGE, width=0.55, edgecolor=GROUND, linewidth=2)
    for xi, v in zip(x, rates.values, strict=True):
        ax.text(xi, v * 100 + 1.2, f"{v:.0%}", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(x, rates.index)
    ax.set_ylabel("Taxa de churn (%)", color=INK_2, fontsize=9)
    ax.set_ylim(0, max(rates.values) * 100 * 1.25)
    _style(ax, title, subtitle)


def save_all(df: pd.DataFrame, model, X_test, y_test, metrics: dict, out: Path) -> list[Path]:
    paths = []

    def save(fig, name):
        path = out / name
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=GROUND)
        plt.close(fig)
        paths.append(path)

    # 1. Churn por tipo de contrato
    fig, ax = plt.subplots(figsize=(6, 3.6))
    rates = df.groupby("Contract")["Churn"].mean().rename(index=PT_CONTRACT)
    _bar_rates(ax, rates, "Churn por tipo de contrato", "Quanto menor o compromisso, maior o cancelamento")
    save(fig, "churn_por_contrato.png")

    # 2. Churn por forma de pagamento
    fig, ax = plt.subplots(figsize=(7, 3.6))
    rates = df.groupby("PaymentMethod")["Churn"].mean().rename(index=PT_PAYMENT).sort_values(ascending=False)
    _bar_rates(ax, rates, "Churn por forma de pagamento", "Pagamento automático segura o cliente")
    save(fig, "churn_por_pagamento.png")

    # 3. Churn por tempo de casa
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    bands = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0–6", "7–12", "13–24", "25–48", "49–72"])
    rates = df.groupby(bands, observed=True)["Churn"].mean()
    ax.plot(
        range(len(rates)),
        rates.values * 100,
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=8,
        markerfacecolor=GROUND,
        markeredgewidth=2,
    )
    for i, v in enumerate(rates.values):
        ax.text(i, v * 100 + 3, f"{v:.0%}", ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.set_xticks(range(len(rates)), [f"{b} meses" for b in rates.index])
    ax.set_ylim(0, 65)
    ax.set_ylabel("Taxa de churn (%)", color=INK_2, fontsize=9)
    _style(ax, "Churn por tempo de casa", "Quem cancela, cancela cedo")
    save(fig, "churn_por_tempo_de_casa.png")

    proba = model.predict_proba(X_test)[:, 1]

    # 4. Curva ROC
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    fpr, tpr, _ = roc_curve(y_test, proba)
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1, label="Chute (AUC 0,50)")
    ax.plot(
        fpr, tpr, color=BLUE, linewidth=2, label=f"Modelo (AUC {metrics['roc_auc']:.2f})".replace(".", ",")
    )
    ax.set_xlabel("Falsos positivos (taxa)", color=INK_2, fontsize=9)
    ax.set_ylabel("Churns encontrados (recall)", color=INK_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax, "Curva ROC no teste")
    ax.grid(axis="x", color=LINE, linewidth=0.8)
    save(fig, "curva_roc.png")

    # 5. Precisão x recall
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    prec, rec, _ = precision_recall_curve(y_test, proba)
    ax.axhline(y_test.mean(), color=MUTED, linestyle="--", linewidth=1, label="Taxa da base")
    ax.plot(
        rec, prec, color=BLUE, linewidth=2, label=f"Modelo (PR-AUC {metrics['pr_auc']:.2f})".replace(".", ",")
    )
    ax.set_xlabel("Recall", color=INK_2, fontsize=9)
    ax.set_ylabel("Precisão", color=INK_2, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax, "Precisão × recall no teste")
    save(fig, "precisao_recall.png")

    # 6. Matriz de confusão
    cm = metrics["confusion_matrix"]
    grid = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max() * 2.4)
    labels = [
        ["Ficou e o modelo\nacertou", "Ficou, mas o modelo\navisou (alarme falso)"],
        ["Cancelou e o modelo\nnão viu", "Cancelou e o modelo\navisou"],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i - 0.08,
                f"{grid[i, j]}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=INK,
            )
            ax.text(j, i + 0.22, labels[i][j], ha="center", va="center", fontsize=7.5, color=INK_2)
    ax.set_xticks([0, 1], ["Previsto: fica", "Previsto: cancela"])
    ax.set_yticks([0, 1], ["Real: ficou", "Real: cancelou"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.set_title(
        f"Matriz de confusão (limiar {metrics['threshold']:.1f})".replace(".", ","),
        loc="left",
        fontsize=12,
        fontweight="bold",
        color=INK,
    )
    save(fig, "matriz_de_confusao.png")

    # 7. Variáveis que mais pesam (coeficientes da Regressão Logística)
    clf = model.named_steps["model"]
    if hasattr(clf, "coef_"):
        names = model.named_steps["prep"].get_feature_names_out()
        coefs = pd.Series(clf.coef_[0], index=names)
        top = pd.concat([coefs.nlargest(6), coefs.nsmallest(6)]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4.6))
        colors = [ORANGE if v > 0 else TEAL for v in top.values]
        ax.barh(range(len(top)), top.values, color=colors, height=0.6, edgecolor=GROUND, linewidth=2)
        ax.set_yticks(range(len(top)), [n.replace("_", " = ") for n in top.index])
        ax.axvline(0, color=INK_2, linewidth=1)
        ax.set_xlabel("Coeficiente (positivo aumenta o risco, negativo diminui)", color=INK_2, fontsize=9)
        _style(
            ax, "O que mais pesa na previsão", "Coeficientes da Regressão Logística (variáveis padronizadas)"
        )
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=LINE, linewidth=0.8)
        save(fig, "coeficientes.png")

    return paths
