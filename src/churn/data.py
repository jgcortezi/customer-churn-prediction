"""Download, leitura e limpeza da base Telco Customer Churn (IBM)."""

import urllib.request
from pathlib import Path

import pandas as pd

from churn.config import DATA_RAW, DATA_URL, ID_COL, TARGET

EXPECTED_COLUMNS = {
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
}


def download(path: Path = DATA_RAW, url: str = DATA_URL) -> Path:
    """Baixa o CSV público, se ainda não estiver no disco."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_raw(path: Path = DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(download(path))
    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos e problemas conhecidos da base.

    - TotalCharges vem como texto e tem 11 valores em branco: são clientes com
      tenure = 0, que ainda não pagaram nenhuma fatura. O valor correto é 0.
    - SeniorCitizen vem como 0/1; vira "No"/"Yes" para ser tratada como categoria.
    - Churn vira 0/1.
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"].astype(str).str.strip(), errors="coerce")
    out.loc[out["TotalCharges"].isna() & (out["tenure"] == 0), "TotalCharges"] = 0.0
    out["SeniorCitizen"] = out["SeniorCitizen"].map({0: "No", 1: "Yes"})
    out[TARGET] = out[TARGET].map({"No": 0, "Yes": 1})
    if out[TARGET].isna().any():
        raise ValueError("A coluna Churn tem valores diferentes de Yes/No.")
    out[TARGET] = out[TARGET].astype(int)
    return out.drop_duplicates(subset=ID_COL).reset_index(drop=True)


def load_clean(path: Path = DATA_RAW) -> pd.DataFrame:
    return clean(load_raw(path))


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[TARGET, ID_COL]), df[TARGET]
