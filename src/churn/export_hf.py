"""Prepara a pasta huggingface/ para publicar o modelo no Hugging Face Hub.

Rode depois do treino:  python -m churn.export_hf
Gera huggingface/model.skops (formato seguro, sem pickle) e huggingface/README.md.
"""

import json

import joblib
from skops.io import dump

from churn.config import MODELS_DIR, REPORTS_DIR, ROOT
from churn.features import CATEGORICAL, NUMERIC

OUT = ROOT / "huggingface"


def main() -> None:
    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    m = json.loads((REPORTS_DIR / "metrics.json").read_text(encoding="utf-8"))
    t = m["teste"]
    OUT.mkdir(exist_ok=True)
    dump(model, OUT / "model.skops")
    exemplo = json.loads((ROOT / "exemplos" / "cliente.json").read_text(encoding="utf-8"))[0]
    exemplo.pop("customerID")
    card = (ROOT / "huggingface" / "card_template.md").read_text(encoding="utf-8").format(
        roc_auc=f"{t['roc_auc']:.3f}",
        pr_auc=f"{t['pr_auc']:.3f}",
        recall=f"{t['recall']:.3f}",
        precision=f"{t['precision']:.3f}",
        f1=f"{t['f1']:.3f}",
        top20=f"{t['churn_capturado_top20pct']:.0%}",
        n_train=m["dados"]["treino"],
        n_test=m["dados"]["teste"],
        numeric=", ".join(f"`{c}`" for c in NUMERIC),
        categorical=", ".join(f"`{c}`" for c in CATEGORICAL),
        exemplo=json.dumps(exemplo, ensure_ascii=False, indent=4),
    )
    (OUT / "README.md").write_text(card, encoding="utf-8")
    print(f"Pronto: {OUT / 'model.skops'} e {OUT / 'README.md'}")


if __name__ == "__main__":
    main()
