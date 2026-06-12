import boto3
import pandas as pd
import itertools
import sys

# BEDROCK CLIENT
bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "global.anthropic.claude-sonnet-4-6"

# RULE ENGINE
WEIGHTS = {
    "high_amount_deviation": 30,
    "country_deviation": 25,
    "category_deviation": 20,
    "velocity_deviation": 20,
    "device_risk": 15,
    "high_risk_merchant": 20
}

THRESHOLD_FRAUD = 60
THRESHOLD_SUSPICIOUS = 30

# LOAD DATA
transactions_df = pd.read_csv(
    "../shared/transactions.csv"
)

memory_df = pd.read_csv(
    "customer_memory.csv"
)

memory_map = {
    row["customer_id"]: row
    for _, row in memory_df.iterrows()
}

analysis_results = []

# SPINNER
spinner = itertools.cycle(
    ["|", "/", "-", "\\"]
)

# ANALYSIS
for _, row in transactions_df.iterrows():

    memory = memory_map[
        row["customer_id"]
    ]

    score = 0

    factors = []

    # MEMORY COMPARISON
    if row["amount"] > (
        memory["avg_amount"] * 3
    ):

        score += WEIGHTS[
            "high_amount_deviation"
        ]

        factors.append(
            "Amount deviates from customer behavior"
        )

    if row["merchant_country"] != (
        memory["common_country"]
    ):

        score += WEIGHTS[
            "country_deviation"
        ]

        factors.append(
            "Country differs from customer pattern"
        )

    if row["merchant_category"] != (
        memory["preferred_category"]
    ):

        score += WEIGHTS[
            "category_deviation"
        ]

        factors.append(
            "Merchant category differs from customer pattern"
        )

    if row["velocity_24h"] > (
        memory["avg_velocity"] * 2
    ):

        score += WEIGHTS[
            "velocity_deviation"
        ]

        factors.append(
            "Transaction velocity abnormal"
        )

    if row["device_changed"]:

        score += WEIGHTS[
            "device_risk"
        ]

        factors.append(
            "Device recently changed"
        )

    if row["merchant_risk"] == "HIGH":

        score += WEIGHTS[
            "high_risk_merchant"
        ]

        factors.append(
            "High-risk merchant"
        )

    score = min(score, 100)

    # PROMPT
    prompt = f"""
You are an advanced fraud detection agent.

Use:
- operational risk analysis
- behavioral memory
- customer historical context

Current Transaction:

amount: {row['amount']}
merchant_country: {row['merchant_country']}
merchant_category: {row['merchant_category']}
velocity_24h: {row['velocity_24h']}
merchant_risk: {row['merchant_risk']}
device_changed: {row['device_changed']}

Behavioral Memory:

avg_amount: {memory['avg_amount']}
common_country: {memory['common_country']}
preferred_category: {memory['preferred_category']}
avg_velocity: {memory['avg_velocity']}
usual_card_present: {memory['usual_card_present']}

Behavioral Risk Score:
{score}/100

Behavioral Risk Factors:
{chr(10).join(f"- {f}" for f in factors)}

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

    # BEDROCK INFERENCE
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

    # CLASSIFICATION
    classification = (
        response["output"]["message"]["content"][0]["text"]
        .strip()
        .replace("*", "")
        .upper()
    )

    # TOKEN OBSERVABILITY
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

    # SAVE RESULTS
    analysis_results.append({

        "transaction_id": row["transaction_id"],

        "customer_id": row["customer_id"],

        "behavioral_score": score,

        "classification": classification,

        "risk_factors": " | ".join(factors),

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens
    })


    # TERMINAL SPINNER
    sys.stdout.write(
        f"\rProcessing behavioral fraud analysis and token observability... "
        f"{next(spinner)}"
    )

    sys.stdout.flush()

# SAVE
analysis_df = pd.DataFrame(
    analysis_results
)

analysis_df.to_csv(
    "analysis_v3.csv",
    index=False
)

# FINISHED
print("\nV3 behavioral analysis completed.")
print("File generated: analysis_v3.csv")