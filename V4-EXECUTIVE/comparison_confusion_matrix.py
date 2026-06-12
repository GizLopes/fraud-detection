import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


# LOAD DATA
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


# MERGE
v1 = truth_df.merge(
    v1_df,
    on="transaction_id"
)

v2 = truth_df.merge(
    v2_df,
    on="transaction_id"
)

v3 = truth_df.merge(
    v3_df,
    on="transaction_id"
)

# BINARY CONVERSION
def to_binary(label):

    if label == "FRAUD":
        return 1

    return 0

# CONFUSION MATRIX FUNCTION
def build_confusion(df):

    y_true = df["true_label"].apply(
    lambda x: 0 if x == "LEGIT" else 1
    )

    y_pred = df["classification"].apply(
    lambda x: 0 if x == "LEGIT" else 1
    )

    return confusion_matrix(
        y_true,
        y_pred
    )

# MATRICES
cm_v1 = build_confusion(v1)
cm_v2 = build_confusion(v2)
cm_v3 = build_confusion(v3)

# PLOT
fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

versions = [
    ("V1 - FM", cm_v1),
    ("V2 - Runtime + Rules", cm_v2),
    ("V3 - Runtime + Memory", cm_v3)
]

for ax, (title, cm) in zip(
    axes,
    versions
):

    # TITLES
    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold"
    )

    # AXES
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels([
        "LLM NEGATIVE",
        "LLM POSITIVE"
    ])

    ax.set_yticklabels([
        "ACTUAL NEGATIVE",
        "ACTUAL POSITIVE"
    ])

    ax.set_xlabel(
        "LLM ANALYZED",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "GROUND TRUTH",
        fontsize=12,
        fontweight="bold"
    )

    # DRAW CELLS
    for i in range(2):
        for j in range(2):

            value = cm[i, j]

            # LABELS
            if i == 0 and j == 0:
                label = "TN"

            elif i == 0 and j == 1:
                label = "FP"

            elif i == 1 and j == 0:
                label = "FN"

            else:
                label = "TP"

            # COLORS
            if label in ["TP", "TN"]:

                cell_color = "#0B1F3A"
                text_color = "white"

            else:

                cell_color = "#B7D7F7"
                text_color = "black"

            rect = plt.Rectangle( 
                (j - 0.5, i - 0.5),
                1,
                1,
                facecolor=cell_color,
                edgecolor="white",
                linewidth=3
            )

            ax.add_patch(rect)

            ax.text(
                j,
                i,
                f"{label}\n{value}",
                ha="center",
                va="center",
                fontsize=18,
                fontweight="bold",
                color=text_color
            )

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)

# SAVE
plt.tight_layout()

plt.savefig(
    "comparison_confusion_matrix.png",
    dpi=300
)

print(
    "\nConfusion matrix generated:"
)

print(
    "comparison_confusion_matrix.png"
)

plt.show(block=False)
plt.pause(3)
plt.close("all")