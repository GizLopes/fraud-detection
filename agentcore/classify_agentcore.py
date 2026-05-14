import boto3
import pandas as pd

bedrock = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)

MODEL_ID = "global.anthropic.claude-sonnet-4-6"

df = pd.read_csv("transactions.csv")

analysis_results = []

for _, row in df.iterrows():

    prompt = f"""
    You are a fraud operations analyst.

    Analyze this transaction using operational fraud reasoning.

    Transaction:

    amount: {row['amount']}
    merchant_risk: {row['merchant_risk']}
    velocity_24h: {row['velocity_24h']}
    merchant_country: {row['merchant_country']}
    ip_country: {row['ip_country']}
    device_changed: {row['device_changed']}
    card_present: {row['card_present']}
    customer_age_days: {row['customer_age_days']}

    Return ONLY one of these values:

    LEGIT
    SUSPICIOUS
    FRAUD

    Do not use markdown.
    Do not use asterisks.
    Return plain text only.
    """

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
            "maxTokens": 20
        }
    )

    classification = response["output"]["message"]["content"][0]["text"]

    analysis_results.append({
        "transaction_id": row["transaction_id"],
        "classification": classification.strip()
    })

analysis_df = pd.DataFrame(analysis_results)

analysis_df.to_csv("analysis.csv", index=False)

print("Analysis completed.")