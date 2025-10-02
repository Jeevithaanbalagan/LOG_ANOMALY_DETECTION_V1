import numpy as np
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

# --------------------------
# STEP 1: Load/Generate embeddings
# --------------------------
# In real project -> use BERT model from HuggingFace
# Here -> simulate embeddings for 8 logs, each 768-dim
np.random.seed(42)
embeddings = np.random.rand(100, 768)   # 100 logs × 768 dims

# Optional: labels (1 = anomaly, 0 = normal), only for evaluation
# In reality, these labels come from SOC feedback
true_labels = np.random.choice([0, 1], size=100, p=[0.85, 0.15])

# --------------------------
# STEP 2: Preprocessing
# --------------------------
# Normalize embeddings (important for cosine distance)
emb_norm = normalize(embeddings, norm='l2')

# Reduce dimensions for stability (optional but recommended)
pca = PCA(n_components=50, random_state=42)
emb_pca = pca.fit_transform(emb_norm)

# --------------------------
# STEP 3: Train DBSCAN
# --------------------------
dbscan = DBSCAN(eps=0.18, min_samples=5, metric='cosine')
cluster_labels = dbscan.fit_predict(emb_pca)

# DBSCAN outputs: 0,1,2,... for clusters and -1 for anomalies
print("DBSCAN cluster assignments (first 20):", cluster_labels[:20])

# --------------------------
# STEP 4: Evaluation (if labels exist)
# --------------------------
# Treat -1 as anomaly prediction
pred_anomalies = (cluster_labels == -1).astype(int)

precision = precision_score(true_labels, pred_anomalies, zero_division=0)
recall = recall_score(true_labels, pred_anomalies, zero_division=0)
f1 = f1_score(true_labels, pred_anomalies, zero_division=0)

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}")

# --------------------------
# STEP 5: Save model
# --------------------------
model_artifact = {
    "dbscan": dbscan,
    "pca": pca,
}
joblib.dump(model_artifact, "dbscan_model.joblib")
print("Model saved as dbscan_model.joblib")

# --------------------------
# STEP 6: Load model and test on new data
# --------------------------
loaded = joblib.load("dbscan_model.joblib")
db = loaded["dbscan"]
pca_model = loaded["pca"]

# Example new log embeddings (simulated)
new_embeddings = np.random.rand(5, 768)

# Normalize + PCA transform
new_emb_norm = normalize(new_embeddings, norm='l2')
new_emb_pca = pca_model.transform(new_emb_norm)

# Predict clusters (-1 = anomaly)
new_preds = db.fit_predict(new_emb_pca)   # NOTE: DBSCAN is not incremental
print("New logs cluster/anomaly labels:", new_preds)
