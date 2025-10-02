import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from datetime import datetime
from collections import Counter
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots
import seaborn as sns

# -------------------------
# Feature Engineering
# -------------------------
def feature_engineering(logs: pd.DataFrame, window="5T") -> pd.DataFrame:
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors="coerce")
    logs = logs.dropna(subset=['timestamp'])
    logs = logs.set_index('timestamp')

    grouped = logs.resample(window)

    # Count failed logins from message column
    failed_counts = grouped['message'].apply(
        lambda x: x.str.contains("authentication failure", case=False, na=False).sum()
    )

    # IP entropy
    def ip_entropy(series):
        ips = series.dropna().tolist()
        if not ips:
            return 0.0
        counts = Counter(ips)
        total = len(ips)
        return -sum((c/total) * math.log(c/total, 2) for c in counts.values())

    features = pd.DataFrame({
        "event_count": grouped.size(),
        "unique_ips_src": grouped['ip_src'].nunique(),
        "unique_ips_dst": grouped['ip_dst'].nunique(),
        "failed_logins": failed_counts,
        "ip_src_entropy": grouped['ip_src'].apply(ip_entropy),
    }).fillna(0)

    # Burstiness
    mean_event = features["event_count"].mean() + 1
    features["burstiness"] = features["event_count"] / mean_event

    # Time-of-day encoding
    features["hour"] = features.index.hour
    features["hour_sin"] = np.sin(2 * np.pi * features["hour"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour"] / 24)
    features.drop(columns=["hour"], inplace=True)

    return features

# -------------------------
# Anomaly Detection
# -------------------------
def anomaly_detection(features: pd.DataFrame) -> pd.DataFrame:
    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(features)
    scores = model.decision_function(features)
    results = features.copy()
    results["iso_label"] = preds
    results["iso_score"] = scores
    return results

# -------------------------
# DBSCAN Clustering
# -------------------------
def cluster_analysis(features: pd.DataFrame) -> pd.Series:
    clustering = DBSCAN(eps=1.5, min_samples=3).fit(features)
    return pd.Series(clustering.labels_, index=features.index, name="db_cluster")

# -------------------------
# Visualization Functions
# -------------------------
def plot_timeline(results):
    plt.figure(figsize=(12,6))
    normal = results[results["iso_label"] == 1]
    anomaly = results[results["iso_label"] == -1]

    plt.plot(normal.index, normal["event_count"], 'o', label="Normal", alpha=0.6)
    plt.plot(anomaly.index, anomaly["event_count"], 'rx', label="Anomaly", markersize=10)

    plt.title("Event Timeline with Anomalies")
    plt.xlabel("Time")
    plt.ylabel("Event Count")
    plt.legend()
    plt.show()

def plot_correlation_heatmap(features):
    plt.figure(figsize=(10,8))
    sns.heatmap(features.corr(), annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

def plot_3d_scatter(results):
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')

    normal = results[results["iso_label"] == 1]
    anomaly = results[results["iso_label"] == -1]

    ax.scatter(normal["failed_logins"], normal["event_count"], normal["ip_src_entropy"],
               c='blue', label='Normal', alpha=0.5)
    ax.scatter(anomaly["failed_logins"], anomaly["event_count"], anomaly["ip_src_entropy"],
               c='red', label='Anomaly', marker='x')

    ax.set_xlabel("Failed Logins")
    ax.set_ylabel("Event Count")
    ax.set_zlabel("IP Src Entropy")
    ax.set_title("3D Scatter: Failed Logins vs Event Count vs Entropy")
    ax.legend()
    plt.show()

def plot_clusters(results):
    plt.figure(figsize=(10,6))
    scatter = plt.scatter(results["event_count"], results["failed_logins"],
                          c=results["db_cluster"], cmap="tab10", alpha=0.7)
    plt.colorbar(scatter, label="Cluster ID")
    plt.title("DBSCAN Clusters (Event Count vs Failed Logins)")
    plt.xlabel("Event Count")
    plt.ylabel("Failed Logins")
    plt.show()

# -------------------------
# Example Run
# -------------------------
if __name__ == "__main__":
    logs = pd.read_csv("log.csv")

    # Feature extraction
    features = feature_engineering(logs)

    # Anomaly detection
    results = anomaly_detection(features)
    results["db_cluster"] = cluster_analysis(features)

    # === Visualization Pages ===
    plot_timeline(results)
    plot_correlation_heatmap(features)
    plot_3d_scatter(results)
    plot_clusters(results)
