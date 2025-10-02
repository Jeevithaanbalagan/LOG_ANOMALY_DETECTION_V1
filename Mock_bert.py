import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

# Example logs
logs = [
    "Failed password for admin",
    "Failed password for root",
    "Database connection error",
    "Database timeout occurred",
    "Firewall DROP TCP connection",
    "VPN login success",
    "Unexpected kernel panic",
    "SSH login from suspicious IP"
]

# Generate mock BERT-like embeddings (8 logs × 10 dims for simplicity)
np.random.seed(42)
embeddings = np.random.rand(len(logs), 10)

print("Embeddings shape:", embeddings.shape)
print("\nAll embeddings:\n", embeddings)

# Normalize embeddings (good for cosine similarity)
embeddings_norm = normalize(embeddings)

# DBSCAN clustering
dbscan = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
labels = dbscan.fit_predict(embeddings_norm)

print("\nDBSCAN Output:")
for log, label in zip(logs, labels):
    print(f"{log} → Cluster {label}")
