import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# -------------------------------------------------
# 1. Load clustered dataset
# -------------------------------------------------

df = pd.read_csv("online_shoppers_clustered.csv")

print("Clustered dataset loaded successfully.")
print("Dataset shape:", df.shape)

# -------------------------------------------------
# 2. Convert boolean columns to integers
# -------------------------------------------------

df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# -------------------------------------------------
# 3. Prepare features and target
# -------------------------------------------------

# Revenue is the target.
# Cluster is included as an additional feature.

X = df.drop(columns=["Revenue"])
y = df["Revenue"]

print("\nFeatures including Cluster:", X.shape)
print("Target:", y.shape)

# -------------------------------------------------
# 4. Split data into training and testing sets
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# -------------------------------------------------
# 5. Scale the features
# -------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# -------------------------------------------------
# 6. Create Logistic Regression model
# -------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# -------------------------------------------------
# 7. Train the combined model
# -------------------------------------------------

model.fit(X_train_scaled, y_train)

print("\nCombined K-Means + Logistic Regression model trained successfully.")

# -------------------------------------------------
# 8. Make predictions
# -------------------------------------------------

y_pred = model.predict(X_test_scaled)

# -------------------------------------------------
# 9. Calculate performance metrics
# -------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== COMBINED MODEL PERFORMANCE ==========")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

# -------------------------------------------------
# 10. Confusion Matrix
# -------------------------------------------------

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)

# -------------------------------------------------
# 11. Classification Report
# -------------------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["No Purchase", "Purchase"]
    )
)

# -------------------------------------------------
# 12. Completed
# -------------------------------------------------

print("\n========== COMBINED MODEL COMPLETED ==========")