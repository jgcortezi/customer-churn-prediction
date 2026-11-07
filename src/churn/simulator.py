"""Modelo compacto que roda no simulador da página (index.html).

O modelo principal usa todas as 19 variáveis. Na página, o usuário só mexe em 6,
então treinamos uma Regressão Logística só com essas 6. Assim cada coeficiente do
simulador é real, em vez de "fingir" que as outras variáveis não existem.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_STATE
from churn.features import build_preprocessor

SIM_NUMERIC = ["tenure", "MonthlyCharges"]
SIM_CATEGORICAL = ["Contract", "InternetService", "PaymentMethod", "TechSupport"]


def fit_simulator(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    cols = SIM_NUMERIC + SIM_CATEGORICAL
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(SIM_NUMERIC, SIM_CATEGORICAL)),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            ),
        ]
    ).fit(X_train[cols], y_train)

    prep = pipe.named_steps["prep"]
    model = pipe.named_steps["model"]
    scaler = prep.named_transformers_["num"].named_steps["scaler"]
    names = prep.get_feature_names_out()
    return {
        "pipeline": pipe,
        "variaveis": cols,
        "roc_auc_teste": roc_auc_score(y_test, pipe.predict_proba(X_test[cols])[:, 1]),
        "intercept": float(model.intercept_[0]),
        "coef": {str(n): float(c) for n, c in zip(names, model.coef_[0], strict=True)},
        "mean": dict(zip(SIM_NUMERIC, map(float, scaler.mean_), strict=True)),
        "std": dict(zip(SIM_NUMERIC, map(float, scaler.scale_), strict=True)),
    }
