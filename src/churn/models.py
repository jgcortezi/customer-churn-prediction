"""Modelos candidatos. Todos usam o mesmo pré-processamento."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.config import RANDOM_STATE
from churn.features import build_preprocessor


def candidates() -> dict[str, Pipeline]:
    """class_weight="balanced" compensa a base desbalanceada (26,5% de churn)."""
    models = {
        "Baseline (classe mais comum)": DummyClassifier(strategy="most_frequent"),
        "Regressão Logística": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("model", m)]) for name, m in models.items()}
