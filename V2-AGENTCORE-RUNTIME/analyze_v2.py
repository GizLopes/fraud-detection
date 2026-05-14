import boto3
import pandas as pd
import itertools
import sys

# =========================================================
# AWS
# =========================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "global.anthropic.claude-sonnet-4-6"

# =========================================================
# RULE ENGINE
# =========================================================

WEIGHTS = {
    "high_amount": 25,
    "high_velocity": 30,
    "device_changed": 15,
    "country_mismatch": 20,
    "high_risk_merchant": 25,
    "new_customer": 20,
    "card_not_present": 10
}

THRESHOLD_FRAUD = 60
THRESHOLD_SUSPICIOUS = 30

# =========================================================
# RISK SCORE
# =========================================================

def compute_risk_score(txn):

    score = 0
    factors = []

    if txn["amount"] > 3000:
        score += WEIGHTS["high_amount"]
        factors.append(
            "High transaction amount"
        )

    if txn["velocity_24h"] > 8:
        score += WEIGHTS["high_velocity"]
        factors.append(
            "High transaction velocity"
        )

    if txn["device_changed"]:
        score += WEIGHTS["device_changed"]
        factors.append(
            "Device recently changed"
        )

    if txn["merchant_country"] != txn["ip_country"]:
        score += WEIGHTS["country_mismatch"]
        factors.append(
            "Country mismatch detected"
        )

    if txn["merchant_risk"] == "HIGH":
        score += WEIGHTS["high_risk_merchant"]
        factors.append(
            "High-risk merchant"
        )

    if txn["customer_age_days"] < 10:
        score += WEIGHTS["new_customer"]
        factors.append(
            "New customer account"
        )

    if not txn["card_present"]:
        score += WEIGHTS["card_not_present"]
        factors.append(
            "Card not present"
        )

    return min(score, 100), factors

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(
    "../shared/transactions.csv"
)

analysis_results = []

# =========================================================
# SPINNER
# =========================================================

spinner = itertools.cycle(
    ["|", "/", "-", "\\"]
)

# =========================================================
# AGENTCORE ANALYSIS
# =========================================================

for _, row in df.iterrows():

    score, factors = compute_risk_score(
        row
    )

    factors_text = "\n".join(
        [f"- {f}" for f in factors]
    )

    prompt = f"""
You are an operational fraud detection agent.

Analyze the transaction using:

1. Risk score
2. Operational risk factors
3. Transaction context

Transaction:

amount: {row['amount']}
merchant_risk: {row['merchant_risk']}
merchant_category: {row['merchant_category']}
velocity_24h: {row['velocity_24h']}
merchant_country: {row['merchant_country']}
ip_country: {row['ip_country']}
device_changed: {row['device_changed']}
card_present: {row['card_present']}
customer_age_days: {row['customer_age_days']}

Operational Risk Score:
{score}/100

Operational Risk Factors:
{factors_text}

Classification Rules:
- score >= 60 → FRAUD
- score >= 30 and < 60 → SUSPICIOUS
- score < 30 → LEGIT

Return ONLY one label:

LEGIT
SUSPICIOUS
FRAUD

Do not explain.
Do not use markdown.
"""

    # =====================================================
    # BEDROCK INFERENCE
    # =====================================================

    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        inferenceConfig={
            "temperature": 0,
            "maxTokens": 5
        }
    )

    # =====================================================
    # CLASSIFICATION
    # =====================================================

    classification = (
        response["output"]["message"]["content"][0]["text"]
        .strip()
        .replace("*", "")
        .upper()
    )

    # =====================================================
    # TOKEN OBSERVABILITY
    # =====================================================

    usage = response.get(
        "usage",
        {}
    )

    input_tokens = usage.get(
        "inputTokens",
        0
    )

    output_tokens = usage.get(
        "outputTokens",
        0
    )

    total_tokens = usage.get(
        "totalTokens",
        0
    )

    # =====================================================
    # SAVE RESULTS
    # =====================================================

    analysis_results.append({

        "transaction_id": row["transaction_id"],

        "risk_score": score,

        "classification": classification,

        "risk_factors": " | ".join(
            factors
        ),

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    })

    # =====================================================
    # TERMINAL SPINNER
    # =====================================================

    sys.stdout.write(
        f"\rProcessing operational fraud analysis and token observability... "
        f"{next(spinner)}"
    )

    sys.stdout.flush()

# =========================================================
# SAVE RESULTS
# =========================================================

analysis_df = pd.DataFrame(
    analysis_results
)

analysis_df.to_csv(
    "analysis_v2.csv",
    index=False
)

# =========================================================
# FINISHED
# =========================================================

print("\nAnalysis completed.")
print("File generated: analysis_v2.csv")