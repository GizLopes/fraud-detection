import pandas as pd

truth_df = pd.read_csv(
    "../shared/hidden_ground_truth.csv"
)

analysis_df = pd.read_csv(
    "analysis_v2.csv"
)

df = truth_df.merge(
    analysis_df,
    on="transaction_id"
)

POSITIVE_CLASSES = [
    "FRAUD",
    "SUSPICIOUS"
]

results = []

tp = tn = fp = fn = 0

for _, row in df.iterrows():

    truth_positive = (
        row["true_label"]
        .upper()
        in POSITIVE_CLASSES
    )

    pred_positive = (
        row["classification"]
        .upper()
        in POSITIVE_CLASSES
    )

    if truth_positive and pred_positive:

        outcome = "TP"
        tp += 1

    elif not truth_positive and not pred_positive:

        outcome = "TN"
        tn += 1

    elif not truth_positive and pred_positive:

        outcome = "FP"
        fp += 1

    else:

        outcome = "FN"
        fn += 1

    results.append({
        "transaction_id": row["transaction_id"],
        "truth": row["true_label"],
        "agentcore analysis": row["classification"],
        "risk_score": row["risk_score"],
        "outcome": outcome
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    "operational_results_v2.csv",
    index=False
)

print("\nOperational Results — V2")
print("=" * 40)

print(f"True Positives : {tp}")
print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

print("\nDetailed operational results saved.")