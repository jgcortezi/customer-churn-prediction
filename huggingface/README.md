---
license: mit
library_name: sklearn
pipeline_tag: tabular-classification
language:
- pt
tags:
- tabular-classification
- churn
- scikit-learn
- logistic-regression
datasets:
- telco-customer-churn
metrics:
- roc_auc
- recall
- precision
- f1
model-index:
- name: churn-radar-logreg
  results:
  - task:
      type: tabular-classification
      name: Previsão de churn
    dataset:
      name: Telco Customer Churn (IBM)
      type: telco-customer-churn
    metrics:
    - type: roc_auc
      value: 0.842
    - type: recall
      value: 0.783
    - type: precision
      value: 0.504
    - type: f1
      value: 0.614
---

# Churn Radar · Regressão Logística para previsão de churn

Modelo que ordena clientes de uma operadora de telecom pelo risco de cancelar o contrato. É um `Pipeline` do scikit-learn com pré-processamento (padronização + one-hot) e Regressão Logística com `class_weight="balanced"`.

Código, notebooks e testes: [github.com/jgcortezi/customer-churn-prediction](https://github.com/jgcortezi/customer-churn-prediction)

## Resultados no conjunto de teste (1409 clientes, nunca vistos no treino)

| Métrica | Valor |
|---|---|
| ROC-AUC | 0.842 |
| PR-AUC | 0.633 (taxa de churn da base: 0,265) |
| Recall (limiar 0,5) | 0.783 |
| Precisão (limiar 0,5) | 0.504 |
| F1 | 0.614 |
| Churns entre os 20% de maior risco | 50% do total |

Na validação cruzada de 5 dobras, empatou com Random Forest e Gradient Boosting (diferença < 0,005 de ROC-AUC). Foi escolhida por ser a mais simples e explicável.

## Como usar

```python
import pandas as pd
from huggingface_hub import hf_hub_download
from skops.io import get_untrusted_types, load

path = hf_hub_download("jgcortezi/churn-radar-logreg", "model.skops")
print(get_untrusted_types(file=path))  # confira os tipos antes de carregar
model = load(path, trusted=get_untrusted_types(file=path))

cliente = {
    "gender": "Female",
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 95.5,
    "TotalCharges": 286.5
}
risco = model.predict_proba(pd.DataFrame([cliente]))[0, 1]
print(f"Risco estimado: {risco:.0%}")
```

**Entradas** (as 19 colunas da base original, menos `customerID` e `Churn`):
- numéricas: `tenure`, `MonthlyCharges`, `TotalCharges`
- categóricas: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`

`SeniorCitizen` deve vir como `"Yes"`/`"No"`.

## Dados de treino

[Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d), base pública de exemplo da IBM: 7.043 clientes fictícios, 26,5% de churn. Divisão estratificada: 5634 para treino e 1409 para teste.

## Limitações e uso adequado

- **Serve para ordenar clientes por risco**, não como probabilidade calibrada: o balanceamento de classes empurra os valores para cima.
- Treinado em uma base fictícia de um único período. Não use para decisões sobre clientes reais sem retreinar com dados próprios e monitorar.
- Indica correlação, não causa: mostra quem priorizar, não por que o cliente vai sair.

Feito por João Gabriel Cortezi.
