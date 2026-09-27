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
# 1. Load dataset
# -------------------------------------------------

df = pd.read_csv("online_shoppers_intention.csv")

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)

# -------------------------------------------------
# 2. Convert boolean columns to integers
# -------------------------------------------------

df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# -------------------------------------------------
# 3. Encode categorical columns
# -------------------------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=["Month", "VisitorType"],
    drop_first=True,
    dtype=int
)

print("\nEncoded dataset shape:", df_encoded.shape)

# -------------------------------------------------
# 4. Separate features and target
# -------------------------------------------------

X = df_encoded.drop(columns=["Revenue"])
y = df_encoded["Revenue"]

print("\nFeatures:", X.shape)
print("Target:", y.shape)

# -------------------------------------------------
# 5. Split data into training and testing sets
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
# 6. Scale the features
# -------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")

# -------------------------------------------------
# 7. Create Logistic Regression model
# -------------------------------------------------

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

# -------------------------------------------------
# 8. Train the model
# -------------------------------------------------

model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully.")

# -------------------------------------------------
# 9. Make predictions
# -------------------------------------------------

y_pred = model.predict(X_test_scaled)

# -------------------------------------------------
# 10. Evaluate the model
# -------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

# -------------------------------------------------
# 11. Confusion Matrix
# -------------------------------------------------

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)

# -------------------------------------------------
# 12. Classification Report
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
# 13. Completed
# -------------------------------------------------

print("\n========== LOGISTIC REGRESSION COMPLETED ==========")