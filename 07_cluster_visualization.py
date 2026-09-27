import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------
# 1. Load clustered dataset
# -------------------------------------------------

df = pd.read_csv("online_shoppers_clustered.csv")

print("Dataset loaded successfully.")
print("Shape:", df.shape)

# -------------------------------------------------
# 2. Cluster Size Visualization
# -------------------------------------------------

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 5))

plt.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

plt.xlabel("Cluster")
plt.ylabel("Number of Shoppers")
plt.title("Number of Shoppers in Each Cluster")

plt.tight_layout()

plt.savefig(
    "cluster_sizes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# -------------------------------------------------
# 3. Revenue Rate by Cluster
# -------------------------------------------------

revenue_rate = (
    df.groupby("Cluster")["Revenue"]
    .mean() * 100
)

plt.figure(figsize=(9, 5))

plt.bar(
    revenue_rate.index.astype(str),
    revenue_rate.values
)

plt.xlabel("Cluster")
plt.ylabel("Revenue Rate (%)")
plt.title("Revenue Rate by Shopper Cluster")

plt.tight_layout()

plt.savefig(
    "revenue_by_cluster.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# -------------------------------------------------
# 4. Prepare data for PCA
# -------------------------------------------------

features_to_use = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Weekend"
]

X = df[features_to_use]

# -------------------------------------------------
# 5. Standardize data
# -------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------------------------
# 6. PCA to 2 dimensions
# -------------------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print("\nPCA completed.")

print(
    "Explained variance:",
    round(pca.explained_variance_ratio_.sum() * 100, 2),
    "%"
)

# -------------------------------------------------
# 7. PCA Cluster Visualization
# -------------------------------------------------

plt.figure(figsize=(9, 6))

plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=df["Cluster"],
    alpha=0.5
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Online Shopper Segments using PCA")

plt.colorbar(label="Cluster")

plt.tight_layout()

plt.savefig(
    "cluster_pca.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

plt.close()

# -------------------------------------------------
# 8. Completed
# -------------------------------------------------

print("\n========== VISUALIZATION COMPLETED ==========")

print("Created files:")
print("1. cluster_sizes.png")
print("2. revenue_by_cluster.png")
print("3. cluster_pca.png")

print("\n========== COMPLETED ==========")