import pandas as pd

# LOAD FILES
truth_df = pd.read_csv(
    "../shared/hidden_ground_truth.csv"
)

v1_df = pd.read_csv(
    "../V1-FOUNDATIONAL-MODEL/analysis_v1.csv"
)

v2_df = pd.read_csv(
    "../V2-AGENTCORE-RUNTIME/analysis_v2.csv"
)

v3_df = pd.read_csv(
    "../V3-AGENTCORE-MEMORY/analysis_v3.csv"
)

# STANDARDIZE COLUMN NAMES
v1_df = v1_df.rename(columns={
    "classification": "v1_fm"
})

v2_df = v2_df.rename(columns={
    "classification": "v2_agentcore rt"
})

v3_df = v3_df.rename(columns={
    "classification": "v3_agentcore rt + memo"
})

# KEEP ONLY NECESSARY COLUMNS
v1_df = v1_df[[
    "transaction_id",
    "v1_fm"
]]

v2_df = v2_df[[
    "transaction_id",
    "v2_agentcore rt"
]]

v3_df = v3_df[[
    "transaction_id",
    "v3_agentcore rt + memo",
    "behavioral_score",
    "risk_factors"
]]


# MERGE DATASETS
df = truth_df.merge(
    v1_df,
    on="transaction_id",
    how="left"
)

df = df.merge(
    v2_df,
    on="transaction_id",
    how="left"
)

df = df.merge(
    v3_df,
    on="transaction_id",
    how="left"
)


# OPERATIONAL OUTCOME FUNCTION
POSITIVE_CLASSES = [
    "FRAUD",
    "SUSPICIOUS"
]

def compute_outcome(
    truth,
    prediction
):

    truth_positive = (
        str(truth).upper()
        in POSITIVE_CLASSES
    )

    pred_positive = (
        str(prediction).upper()
        in POSITIVE_CLASSES
    )

    if truth_positive and pred_positive:
        return "TP"

    elif not truth_positive and not pred_positive:
        return "TN"

    elif not truth_positive and pred_positive:
        return "FP"

    else:
        return "FN"

# COMPUTE OUTCOMES
df["v1_outcome"] = df.apply(
    lambda row: compute_outcome(
        row["true_label"],
        row["v1_fm"]
    ),
    axis=1
)

df["v2_outcome"] = df.apply(
    lambda row: compute_outcome(
        row["true_label"],
        row["v2_agentcore rt"]
    ),
    axis=1
)

df["v3_outcome"] = df.apply(
    lambda row: compute_outcome(
        row["true_label"],
        row["v3_agentcore rt + memo"]
    ),
    axis=1
)


# COLUMN ORDER
final_columns = [

    "transaction_id",

    "true_label",

    "v1_fm",
    "v1_outcome",

    "v2_agentcore rt",
    "v2_outcome",

    "v3_agentcore rt + memo",
    "v3_outcome",

    "behavioral_score",
    "risk_factors"
]

df = df[final_columns]

# SAVE FINAL TABLE
OUTPUT_FILE = "analytics_confusion_matrix.csv"

df.to_csv(
    OUTPUT_FILE,
    index=False
)

# SUMMARY
print("\nComparison table generated successfully.")

print(f"\nOutput file:")
print(f"{OUTPUT_FILE}")

print("\nColumns:")
for col in df.columns:
    print(f" - {col}")

print(f"\nTotal rows: {len(df)}")