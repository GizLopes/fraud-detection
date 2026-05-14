import random
import pandas as pd

N = 100

countries = ["BR", "US", "MX", "DR"]

transactions = []
ground_truth = []

for i in range(1, N + 1):

    transaction = {
        "transaction_id": i,
        "amount": round(random.uniform(10, 8000), 2),
        "merchant_risk": random.choice(["LOW", "MEDIUM", "HIGH"]),
        "velocity_24h": random.randint(1, 15),
        "merchant_country": random.choice(countries),
        "ip_country": random.choice(countries),
        "device_changed": random.choice([True, False]),
        "card_present": random.choice([True, False]),
        "customer_age_days": random.randint(1, 1000)
    }

    # Hidden evaluation logic
    score = 0

    if transaction["amount"] > 3000:
        score += 25

    if transaction["velocity_24h"] > 8:
        score += 30

    if transaction["device_changed"]:
        score += 15

    if transaction["merchant_country"] != transaction["ip_country"]:
        score += 20

    if transaction["merchant_risk"] == "HIGH":
        score += 25

    if transaction["customer_age_days"] < 10:
        score += 20

    if not transaction["card_present"]:
        score += 10

    if score < 30:
        label = "LEGIT"
    elif score < 60:
        label = "SUSPICIOUS"
    else:
        label = "FRAUD"

    transactions.append(transaction)

    ground_truth.append({
        "transaction_id": i,
        "true_label": label
    })

pd.DataFrame(transactions).to_csv("transactions.csv", index=False)

pd.DataFrame(ground_truth).to_csv("hidden_ground_truth.csv", index=False)

print("Files generated.")