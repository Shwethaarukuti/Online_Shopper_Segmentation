import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Store model performance
# -------------------------------------------------

results = {
    "Model": [
        "Logistic Regression",
        "K-Means + Logistic Regression"
    ],
    "Accuracy": [
        85.00,
        84.87
    ],
    "Precision": [
        51.07,
        50.80
    ],
    "Recall": [
        74.87,
        74.87
    ],
    "F1 Score": [
        60.72,
        60.53
    ]
}

results_df = pd.DataFrame(results)

# -------------------------------------------------
# 2. Display comparison table
# -------------------------------------------------

print("========== MODEL PERFORMANCE COMPARISON ==========\n")

print(results_df.to_string(index=False))

# -------------------------------------------------
# 3. Save comparison table
# -------------------------------------------------

results_df.to_csv(
    "model_performance_comparison.csv",
    index=False
)

print("\nComparison table saved as:")
print("model_performance_comparison.csv")

# -------------------------------------------------
# 4. Prepare chart
# -------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

logistic_values = [
    85.00,
    51.07,
    74.87,
    60.72
]

combined_values = [
    84.87,
    50.80,
    74.87,
    60.53
]

x = range(len(metrics))
width = 0.35

# -------------------------------------------------
# 5. Create comparison chart
# -------------------------------------------------

plt.figure(figsize=(10, 6))

plt.bar(
    [i - width / 2 for i in x],
    logistic_values,
    width=width,
    label="Logistic Regression"
)

plt.bar(
    [i + width / 2 for i in x],
    combined_values,
    width=width,
    label="K-Means + Logistic Regression"
)

plt.xticks(
    list(x),
    metrics
)

plt.ylabel("Score (%)")
plt.xlabel("Evaluation Metric")

plt.title("Model Performance Comparison")

plt.legend()

plt.tight_layout()

# -------------------------------------------------
# 6. Save chart
# -------------------------------------------------

plt.savefig(
    "model_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nComparison chart saved as:")
print("model_performance_comparison.png")

# -------------------------------------------------
# 7. Completed
# -------------------------------------------------

print("\n========== STEP 10 COMPLETED ==========")