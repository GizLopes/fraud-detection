import random
import pandas as pd

df = pd.read_csv(
    "../shared/transactions.csv"
)

profiles = []

countries = ["BR", "US", "MX", "AR"]

categories = [
    "FOOD",
    "TECH",
    "TRAVEL",
    "GAMING"
]

for customer_id in range(1, 51):

    profile = {
        "customer_id": customer_id,

        "avg_amount": round(
            random.uniform(100, 3000),
            2
        ),

        "common_country": random.choice(
            countries
        ),

        "preferred_category": random.choice(
            categories
        ),

        "avg_velocity": random.randint(
            1,
            5
        ),

        "usual_card_present": random.choice(
            [True, False]
        )
    }

    profiles.append(profile)

profiles_df = pd.DataFrame(
    profiles
)

profiles_df.to_csv(
    "customer_memory.csv",
    index=False
)

print("Behavioral memory profiles created.")