import pandas as pd

# Load dataset
df = pd.read_csv("online_shoppers_intention.csv")

print("========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicates:", df.duplicated().sum())

print("\n========== REVENUE DISTRIBUTION ==========")
print(df["Revenue"].value_counts())

print("\n========== REVENUE PERCENTAGE ==========")
print(df["Revenue"].value_counts(normalize=True) * 100)

print("\n========== VISITOR TYPE ==========")
print(df["VisitorType"].value_counts())

print("\n========== MONTH ==========")
print(df["Month"].value_counts())

print("\n========== DESCRIPTIVE STATISTICS ==========")
print(df.describe())