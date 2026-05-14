import pandas as pd

truth_df    = pd.read_csv("hidden_ground_truth.csv")
analysis_df = pd.read_csv("analysis.csv")

df = truth_df.merge(analysis_df, on="transaction_id")

# Classes consideradas positivas
POSITIVE_CLASSES = ["fraud", "suspicious"]

results = []

tp = tn = fp = fn = 0

for _, row in df.iterrows():

    truth_positive = row["true_label"].lower() in POSITIVE_CLASSES
    pred_positive  = row["classification"].lower() in POSITIVE_CLASSES

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
        "prediction": row["classification"],
        "outcome": outcome
    })

results_df = pd.DataFrame(results)

results_df.to_csv("operational_results.csv", index=False)

print("\nOperational Fraud Analysis")
print("=" * 40)

print(f"True Positives : {tp}")
print(f"True Negatives : {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")

print("\nDetailed results saved to operational_results.csv")