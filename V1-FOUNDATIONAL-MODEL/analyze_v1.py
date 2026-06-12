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

# LOAD DATA
df = pd.read_csv(
    "../shared/transactions.csv"
)

analysis_results = []

# SPINNER - Frill for terminal output while processing transactions
spinner = itertools.cycle(
    ["|", "/", "-", "\\"]
)

# MAIN LOOP
for _, row in df.iterrows():

    prompt = f"""
You are a fraud analyst.

Analyze this transaction and classify it as:

- LEGIT
- SUSPICIOUS
- FRAUD

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

Return ONLY one label:

LEGIT
SUSPICIOUS
FRAUD

Do not use markdown.
Do not explain.
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

    # TOKEN USAGE
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

    # Claude Sonnet 4.6 → 200k context window
    context_usage_pct = round(
        (total_tokens / 200000) * 100,
        6
    )


    # SAVE RESULTS
    analysis_results.append({

        "transaction_id": row["transaction_id"],

        "classification": classification,

        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,

        "context_usage_pct": context_usage_pct
    })

    # TERMINAL SPINNER
    sys.stdout.write(
        f"\rProcessing transactions and checking tokens... "
        f"{next(spinner)}"
    )

    sys.stdout.flush()

# FINAL DATAFRAME
analysis_df = pd.DataFrame(
    analysis_results
)

analysis_df.to_csv(
    "analysis_v1.csv",
    index=False
)

# FINISHED
print("\nAnalysis completed.")
print("File generated: analysis_v1.csv")