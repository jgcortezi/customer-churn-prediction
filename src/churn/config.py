"""Caminhos e constantes do projeto."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SITE_HTML = ROOT / "index.html"

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Churn"
ID_COL = "customerID"
