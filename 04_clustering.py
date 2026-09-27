import pandas as pd
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# 1. Load the original dataset
# -------------------------------------------------

df = pd.read_csv("online_shoppers_intention.csv")

print("Original dataset shape:", df.shape)

# -------------------------------------------------
# 2. Convert Boolean columns to integers
# -------------------------------------------------

df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# -------------------------------------------------
# 3. One-hot encode categorical columns
# -------------------------------------------------

df = pd.get_dummies(
    df,
    columns=["Month", "VisitorType"],
    drop_first=True,
    dtype=int
)

# -------------------------------------------------
# 4. Separate clustering features
# -------------------------------------------------

X_cluster = df.drop(columns=["Revenue"])

print("\nClustering data shape:", X_cluster.shape)

# -------------------------------------------------
# 5. Scale the features
# -------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_cluster)

print("\nScaled data shape:", X_scaled.shape)

print("\nFirst 5 scaled rows:")
print(X_scaled[:5])