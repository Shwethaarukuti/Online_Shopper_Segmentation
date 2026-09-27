import pandas as pd

# Load dataset
df = pd.read_csv("online_shoppers_intention.csv")

print("Original dataset shape:", df.shape)

# -------------------------------------------------
# 1. Convert Boolean columns to integers
# -------------------------------------------------

df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

# -------------------------------------------------
# 2. Convert categorical columns using One-Hot Encoding
# -------------------------------------------------

df = pd.get_dummies(
    df,
    columns=["Month", "VisitorType"],
    drop_first=True,
    dtype=int
)

# -------------------------------------------------
# 3. Display the processed dataset
# -------------------------------------------------

print("\nProcessed dataset shape:", df.shape)

print("\nProcessed column names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())