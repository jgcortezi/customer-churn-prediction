"""Previsão com o modelo treinado.

Uso pelo terminal:
    python -m churn.predict exemplos/cliente.json
"""

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from churn.config import MODELS_DIR
from churn.features import CATEGORICAL, NUMERIC

MODEL_PATH = MODELS_DIR / "churn_model.joblib"
FEATURES = NUMERIC + CATEGORICAL


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError("Modelo não encontrado. Rode antes: python -m churn.train")
    return joblib.load(path)


def predict_proba(clientes: list[dict], model=None) -> list[float]:
    """Risco de churn (0 a 1) para cada cliente. Serve para ordenar; não é calibrado."""
    faltando = [f for f in FEATURES if any(f not in c for c in clientes)]
    if faltando:
        raise ValueError(f"Campos obrigatórios ausentes: {sorted(set(faltando))}")
    model = model or load_model()
    df = pd.DataFrame(clientes)[FEATURES]
    return model.predict_proba(df)[:, 1].round(4).tolist()


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python -m churn.predict caminho/para/cliente.json")
        raise SystemExit(1)
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    clientes = data if isinstance(data, list) else [data]
    for c, p in zip(clientes, predict_proba(clientes), strict=True):
        print(f"{c.get('customerID', 'cliente')}: risco estimado de {p:.0%}")


if __name__ == "__main__":
    main()
