import random
import pandas as pd

N = 100

countries = ["BR", "US", "MX", "AR"]
merchant_categories = ["FOOD", "TECH", "TRAVEL", "GAMING"]

transactions = []
ground_truth = []

for i in range(1, N + 1):

    amount = round(random.uniform(10, 8000), 2)

    merchant_risk = random.choice([
        "LOW",
        "MEDIUM",
        "HIGH"
    ])

    velocity_24h = random.randint(1, 15)

    merchant_country = random.choice(countries)

    ip_country = random.choice(countries)

    device_changed = random.choice([True, False])

    card_present = random.choice([True, False])

    customer_age_days = random.randint(1, 1000)

    transaction = {
        "transaction_id": i,
        "amount": amount,
        "merchant_risk": merchant_risk,
        "merchant_category": random.choice(merchant_categories),
        "velocity_24h": velocity_24h,
        "merchant_country": merchant_country,
        "ip_country": ip_country,
        "device_changed": device_changed,
        "card_present": card_present,
        "customer_age_days": customer_age_days
    }

    # Hidden operational baseline
    score = 0

    if amount > 3000:
        score += 25

    if velocity_24h > 8:
        score += 30

    if device_changed:
        score += 15

    if merchant_country != ip_country:
        score += 20

    if merchant_risk == "HIGH":
        score += 25

    if customer_age_days < 10:
        score += 20

    if not card_present:
        score += 10

    if score >= 60:
        label = "FRAUD"

    elif score >= 30:
        label = "SUSPICIOUS"

    else:
        label = "LEGIT"

    transactions.append(transaction)

    ground_truth.append({
        "transaction_id": i,
        "true_label": label
    })

transactions_df = pd.DataFrame(transactions)

truth_df = pd.DataFrame(ground_truth)

transactions_df.to_csv(
    "../shared/transactions.csv",
    index=False
)

truth_df.to_csv(
    "../shared/hidden_ground_truth.csv",
    index=False
)

print("Synthetic transactions generated.")