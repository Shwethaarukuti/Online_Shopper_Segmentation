import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------------------------
# 1. Load the dataset
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

df_encoded = pd.get_dummies(
    df,
    columns=["Month", "VisitorType"],
    drop_first=True,
    dtype=int
)

# -------------------------------------------------
# 4. Remove Revenue from clustering features
# -------------------------------------------------

X_cluster = df_encoded.drop(columns=["Revenue"])

print("Clustering data shape:", X_cluster.shape)

# -------------------------------------------------
# 5. Scale the features
# -------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_cluster)

print("Data scaling completed.")

# -------------------------------------------------
# 6. Apply K-Means clustering
# -------------------------------------------------

K = 10

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

# -------------------------------------------------
# 7. Add cluster labels to the dataset
# -------------------------------------------------

df_encoded["Cluster"] = clusters

print("\n========== CLUSTERING COMPLETED ==========")

# -------------------------------------------------
# 8. Display cluster sizes
# -------------------------------------------------

print("\nCluster sizes:")

print(
    df_encoded["Cluster"]
    .value_counts()
    .sort_index()
)

# -------------------------------------------------
# 9. Cluster profile
# -------------------------------------------------

print("\n========== CLUSTER PROFILE ==========")

cluster_profile = df_encoded.groupby("Cluster").agg({
    "Administrative": "mean",
    "Informational": "mean",
    "ProductRelated": "mean",
    "ProductRelated_Duration": "mean",
    "BounceRates": "mean",
    "ExitRates": "mean",
    "PageValues": "mean",
    "Revenue": "mean"
}).round(3)

print(cluster_profile)

# -------------------------------------------------
# 10. Save clustered dataset
# -------------------------------------------------

df_encoded.to_csv(
    "online_shoppers_clustered.csv",
    index=False
)

print("\nClustered dataset saved as:")
print("online_shoppers_clustered.csv")

print("\n========== COMPLETED ==========")