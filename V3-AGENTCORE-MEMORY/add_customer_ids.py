import random
import pandas as pd

df = pd.read_csv(
    "../shared/transactions.csv"
)

# Fixed Seed to stablishment of consistent customer_id assignment across runs
random.seed(42)

df["customer_id"] = [
    random.randint(1, 50)
    for _ in range(len(df))
]

df.to_csv(
    "../shared/transactions.csv",
    index=False
)

print("customer_id added successfully.")