"""
generate_data.py
Gera dados sintéticos de transações e merchants para a POC de Fraud Detection.
Salva os arquivos transactions.json e merchants.json na pasta data/.
"""

import json
import random
import os
from datetime import datetime, timedelta

random.seed(42)

# ──────────────────────────────────────────────
# Configurações
# ──────────────────────────────────────────────
NUM_MERCHANTS   = 30
NUM_TRANSACTIONS = 100
OUTPUT_DIR       = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Merchants
# ──────────────────────────────────────────────
MERCHANT_CATEGORIES = [
    "grocery", "electronics", "restaurant", "travel",
    "clothing", "pharmacy", "gas_station", "online_retail"
]

# Merchants marcados como high_risk representam estabelecimentos com histórico
# de fraude ou operação em países de alto risco.
merchants = []
for i in range(1, NUM_MERCHANTS + 1):
    high_risk = random.random() < 0.2          # 20% são high-risk
    country   = random.choice(
        ["BR", "BR", "BR", "US", "US", "NG", "RU", "CN"]
        if high_risk else
        ["BR", "BR", "BR", "US", "US", "DE", "FR", "JP"]
    )
    merchants.append({
        "merchant_id":   f"M{i:03d}",
        "name":          f"Merchant_{i:03d}",
        "category":      random.choice(MERCHANT_CATEGORIES),
        "country":       country,
        "high_risk":     high_risk,
        "avg_ticket":    round(random.uniform(20, 500), 2),
    })

with open(f"{OUTPUT_DIR}/merchants.json", "w") as f:
    json.dump(merchants, f, indent=2)
print(f"✅  {len(merchants)} merchants gerados → {OUTPUT_DIR}/merchants.json")

# ──────────────────────────────────────────────
# Transações
# ──────────────────────────────────────────────
# Regras de negócio para ground-truth label
# ─────────────────────────────────────────────
# FRAUDE  → qualquer uma das condições abaixo:
#   1. Valor > 5× o avg_ticket do merchant
#   2. Merchant high_risk E valor > 2× o avg_ticket
#   3. Horário entre 01h–04h E valor > 500
#
# SUSPEITA → qualquer uma abaixo (sem ser fraude):
#   1. Merchant high_risk
#   2. Horário entre 23h–01h ou 04h–06h
#   3. Valor entre 2× e 5× o avg_ticket
#   4. Mais de 3 transações do mesmo cartão nas últimas 1 h
#
# LEGÍTIMA → todo o resto
# ──────────────────────────────────────────────

merchant_map = {m["merchant_id"]: m for m in merchants}
BASE_DATE    = datetime(2024, 6, 1)
card_pool    = [f"CARD_{i:04d}" for i in range(1, 60)]

# rastreia histórico por cartão para checar velocity
card_history: dict[str, list[datetime]] = {}

transactions = []
for i in range(1, NUM_TRANSACTIONS + 1):
    merchant   = random.choice(merchants)
    card_id    = random.choice(card_pool)
    ts         = BASE_DATE + timedelta(
        days=random.randint(0, 29),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    amount     = round(random.uniform(5, merchant["avg_ticket"] * 6), 2)
    avg_ticket = merchant["avg_ticket"]
    hour       = ts.hour

    # ── velocity: qtd de txns do mesmo cartão na última 1h ──
    card_history.setdefault(card_id, [])
    window_start = ts - timedelta(hours=1)
    recent = [t for t in card_history[card_id] if t >= window_start]
    velocity = len(recent)
    card_history[card_id].append(ts)

    # ── calcula label ──
    is_fraud = (
        amount > 5 * avg_ticket
        or (merchant["high_risk"] and amount > 2 * avg_ticket)
        or (1 <= hour <= 4 and amount > 500)
    )
    is_suspicious = not is_fraud and (
        merchant["high_risk"]
        or hour in (23, 0, 4, 5)
        or (2 * avg_ticket < amount <= 5 * avg_ticket)
        or velocity > 3
    )

    label = "fraud" if is_fraud else ("suspicious" if is_suspicious else "legitimate")

    transactions.append({
        "transaction_id": f"T{i:04d}",
        "card_id":        card_id,
        "merchant_id":    merchant["merchant_id"],
        "amount":         amount,
        "timestamp":      ts.isoformat(),
        "hour":           hour,
        "velocity_1h":    velocity,
        "label":          label,        # ground-truth para a matriz de confusão
    })

with open(f"{OUTPUT_DIR}/transactions.json", "w") as f:
    json.dump(transactions, f, indent=2, default=str)

# Sumário
from collections import Counter
counts = Counter(t["label"] for t in transactions)
print(f"✅  {len(transactions)} transações geradas → {OUTPUT_DIR}/transactions.json")
print(f"   legitimate : {counts['legitimate']}")
print(f"   suspicious : {counts['suspicious']}")
print(f"   fraud      : {counts['fraud']}")
