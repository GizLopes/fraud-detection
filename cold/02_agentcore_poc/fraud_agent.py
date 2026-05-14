"""
fraud_agent.py
POC de Fraud Detection usando Amazon Bedrock AgentCore (Runtime).

Fluxo:
  1. Carrega transactions.json + merchants.json gerados por generate_data.py
  2. Para cada transação, calcula um score de risco via regras de negócio
  3. Envia ao AgentCore (claude-3-sonnet via bedrock) com o score + contexto
  4. Coleta a classificação retornada (legitimate / suspicious / fraud)
  5. Compara com o ground-truth e exibe a Matriz de Confusão

Pré-requisitos
──────────────
  pip install boto3 scikit-learn matplotlib seaborn

Configuração AWS
────────────────
  Execute `aws configure` no terminal ou defina as variáveis de ambiente:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
  A região padrão usada aqui é us-east-1 (Bedrock disponível).
"""

import json
import time
import boto3
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
DATA_DIR    = Path("data")
AWS_REGION  = "us-east-1"          # altere se necessário
MODEL_ID    = "global.anthropic.claude-sonnet-4-6"
DELAY_SEC   = 0.5                  # pausa entre chamadas (throttle)

# ──────────────────────────────────────────────
# Regras de negócio para o score de risco
# ──────────────────────────────────────────────
# Pesos somam até 100. O agente recebe o score e os fatores acionados.
# ─────────────────────────────────────────────
WEIGHTS = {
    "high_value_5x":       35,   # Valor > 5× avg_ticket do merchant
    "high_risk_merchant":  20,   # Merchant marcado como high_risk
    "high_value_2x":       15,   # Valor entre 2× e 5× avg_ticket
    "night_transaction":   15,   # Horário 01h–04h
    "late_night":          10,   # Horário 23h / 04h–06h
    "high_velocity":       20,   # > 3 txns do mesmo cartão em 1 h
    "high_amount_night":   25,   # Valor > 500 + horário noturno (01-04h)
}

# Limiares de decisão
THRESHOLD_FRAUD      = 50
THRESHOLD_SUSPICIOUS = 25


def compute_risk_score(txn: dict, merchant: dict) -> tuple[int, list[str]]:
    """Calcula o risk score e retorna os fatores acionados."""
    score   = 0
    factors = []
    amount      = txn["amount"]
    avg_ticket  = merchant["avg_ticket"]
    hour        = txn["hour"]
    velocity    = txn["velocity_1h"]
    high_risk   = merchant["high_risk"]

    if amount > 5 * avg_ticket:
        score += WEIGHTS["high_value_5x"]
        factors.append(f"Valor {amount:.2f} > 5× avg_ticket ({avg_ticket:.2f})")

    elif amount > 2 * avg_ticket:
        score += WEIGHTS["high_value_2x"]
        factors.append(f"Valor {amount:.2f} entre 2× e 5× avg_ticket ({avg_ticket:.2f})")

    if high_risk:
        score += WEIGHTS["high_risk_merchant"]
        factors.append("Merchant classificado como high_risk")

    if 1 <= hour <= 4:
        score += WEIGHTS["night_transaction"]
        factors.append(f"Horário suspeito: {hour:02d}h (madrugada)")
        if amount > 500:
            score += WEIGHTS["high_amount_night"]
            factors.append(f"Valor alto ({amount:.2f}) em horário de madrugada")

    elif hour in (23, 4, 5):
        score += WEIGHTS["late_night"]
        factors.append(f"Horário tardio/early: {hour:02d}h")

    if velocity > 3:
        score += WEIGHTS["high_velocity"]
        factors.append(f"Alta frequência: {velocity} transações no último 1h")

    return min(score, 100), factors


def build_prompt(txn: dict, merchant: dict, score: int, factors: list[str]) -> str:
    """Monta o prompt enviado ao Bedrock."""
    factors_str = "\n".join(f"  - {f}" for f in factors) if factors else "  - Nenhum fator de risco detectado"
    label_hint = (
        "fraud"      if score >= THRESHOLD_FRAUD      else
        "suspicious" if score >= THRESHOLD_SUSPICIOUS else
        "legitimate"
    )
    return f"""Você é um sistema especialista em detecção de fraudes em transações financeiras.

## Dados da Transação
- ID: {txn['transaction_id']}
- Cartão: {txn['card_id']}
- Valor: R$ {txn['amount']:.2f}
- Horário: {txn['timestamp']}
- Merchant: {merchant['name']} (categoria: {merchant['category']}, país: {merchant['country']})
- Ticket médio do merchant: R$ {merchant['avg_ticket']:.2f}
- Transações do cartão na última 1h: {txn['velocity_1h']}

## Score de Risco Calculado
Score: {score}/100 (limiar fraude: {THRESHOLD_FRAUD}, suspeita: {THRESHOLD_SUSPICIOUS})

## Fatores de Risco Identificados
{factors_str}

## Instrução
Com base nos dados e no score acima, classifique esta transação.
Responda APENAS com um JSON no formato exato abaixo (sem markdown, sem texto extra):
{{"classification": "<legitimate|suspicious|fraud>", "reasoning": "<explicação em 1 frase>"}}

Dica interna: o score sugere "{label_hint}"."""


def classify_transaction(client, txn: dict, merchant: dict) -> dict:
    """Chama o Bedrock e retorna o dict com classification + reasoning."""
    score, factors = compute_risk_score(txn, merchant)
    prompt         = build_prompt(txn, merchant, score, factors)

    response = client.invoke_model(
        modelId     = MODEL_ID,
        contentType = "application/json",
        accept      = "application/json",
        body        = json.dumps({
            "anthropic_version": "bedrock-2023-05-31", #Bedrock API
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
        }),
    )

    body = json.loads(response["body"].read())
    raw  = body["content"][0]["text"].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # fallback: usa o label_hint calculado localmente
        result = {
            "classification": (
                "fraud"      if score >= THRESHOLD_FRAUD      else
                "suspicious" if score >= THRESHOLD_SUSPICIOUS else
                "legitimate"
            ),
            "reasoning": "(fallback – resposta inválida do modelo)",
        }

    result["risk_score"] = score
    result["factors"]    = factors
    return result


def plot_confusion_matrix(y_true: list, y_pred: list) -> None:
    """Plota e salva a matriz de confusão."""
    labels  = ["legitimate", "suspicious", "fraud"]
    cm      = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Fraud Detection POC – Matriz de Confusão", fontsize=14, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".0%"],
        ["Contagem absoluta", "Proporção por classe real"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=labels, yticklabels=labels,
            linewidths=0.5, ax=ax,
        )
        ax.set_xlabel("Predito pelo Agente", fontsize=11)
        ax.set_ylabel("Ground-Truth",        fontsize=11)
        ax.set_title(title,                  fontsize=11)

    plt.tight_layout()
    out = "confusion_matrix.png"
    plt.savefig(out, dpi=150)
    print(f"\n📊  Matriz de confusão salva em: {out}")
    plt.show()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    # 1. Carrega dados
    transactions = json.loads((DATA_DIR / "transactions.json").read_text())
    merchants    = {m["merchant_id"]: m for m in json.loads((DATA_DIR / "merchants.json").read_text())}

    print(f"🔎  {len(transactions)} transações carregadas.")

    # 2. Cliente Bedrock
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)

    # 3. Classifica cada transação
    results  = []
    y_true   = []
    y_pred   = []

    for idx, txn in enumerate(transactions, 1):
        merchant = merchants[txn["merchant_id"]]
        try:
            res = classify_transaction(bedrock, txn, merchant)
        except Exception as e:
            print(f"  ⚠️  Erro na txn {txn['transaction_id']}: {e}")
            res = {
                "classification": "legitimate",
                "reasoning":      f"Erro na API: {e}",
                "risk_score":     0,
                "factors":        [],
            }

        pred  = res["classification"]
        truth = txn["label"]

        y_true.append(truth)
        y_pred.append(pred)
        results.append({**txn, **res, "ground_truth": truth})

        status = "✅" if pred == truth else "❌"
        print(f"  [{idx:3d}/{len(transactions)}] {txn['transaction_id']} "
              f"| score={res['risk_score']:3d} "
              f"| truth={truth:10s} pred={pred:10s} {status}")

        time.sleep(DELAY_SEC)

    # 4. Relatório
    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                 target_names=["legitimate", "suspicious", "fraud"],
                                 zero_division=0))

    # 5. Matriz de confusão
    plot_confusion_matrix(y_true, y_pred)

    # 6. Salva resultados detalhados
    out_file = "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"💾  Resultados detalhados salvos em: {out_file}")


if __name__ == "__main__":
    main()
