import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------

df = pd.read_csv("online_shoppers_intention.csv")

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
# 4. Remove Revenue for clustering
# -------------------------------------------------

X_cluster = df.drop(columns=["Revenue"])

# -------------------------------------------------
# 5. Scale the data
# -------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

print("Data prepared successfully.")
print("Shape:", X_scaled.shape)

# -------------------------------------------------
# 6. Elbow Method
# -------------------------------------------------

inertia = []

print("\n========== ELBOW METHOD ==========")

for k in range(2, 11):

    print("Calculating K =", k)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)

    print("K =", k, "Inertia =", round(kmeans.inertia_, 2))

# -------------------------------------------------
# 7. Display Elbow Method graph
# -------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 11),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.xticks(range(2, 11))
plt.grid(True)

plt.savefig("elbow_method.png", dpi=300, bbox_inches="tight")
print("Elbow graph saved as elbow_method.png")
# -------------------------------------------------
# 8. Silhouette Score
# -------------------------------------------------

print("\n========== SILHOUETTE SCORES ==========")

# Use a sample to make silhouette calculation faster
sample_size = min(3000, len(X_scaled))

sample_indices = pd.Series(range(len(X_scaled))).sample(
    n=sample_size,
    random_state=42
).values

X_sample = X_scaled[sample_indices]

for k in range(2, 11):

    print("Calculating Silhouette Score for K =", k)

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_sample)

    score = silhouette_score(X_sample, labels)

    print(
        "K =", k,
        "Silhouette Score =", round(score, 4)
    )

print("\n========== COMPLETED ==========")