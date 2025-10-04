import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from collections import Counter
import math

# -------------------------
# Feature Engineering
# -------------------------
def feature_engineering(logs: pd.DataFrame, window="5T") -> pd.DataFrame:
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors="coerce")
    logs = logs.dropna(subset=['timestamp'])
    logs = logs.set_index('timestamp')

    grouped = logs.resample(window)

    # Failed login counts
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
# Metrics Validation
# -------------------------
def validate_metrics(features: pd.DataFrame, anomalies: pd.Series):
    return {
        "avg_event_rate": features["event_count"].mean(),
        "anomaly_fraction": (anomalies == -1).mean(),
        "avg_failed_logins": features["failed_logins"].mean(),
    }

# -------------------------
# Example Run
# -------------------------
if __name__ == "__main__":
    # Load logs
    logs = pd.read_csv("log.csv")
    print(f"Loaded {len(logs)} log entries.\n")

    # Layer 1: Features
    features = feature_engineering(logs)

    # Layer 2: Anomalies
    results = anomaly_detection(features)
    results["db_cluster"] = cluster_analysis(features)

    # Layer 3: Security metrics
    log_metrics = validate_metrics(features, results["iso_label"])
    nonlog_metrics = {"patch_compliance": 0.85, "MFA_coverage": 0.9, "backup_success_rate": 0.95}
    combined_metrics = {**log_metrics, **nonlog_metrics}

    # Layer 4: Business outcomes (placeholder correlation)
    business_outcomes = {"uptime": 0.997, "revenue_growth": 0.05}
    correlations = {}
    for k, v in combined_metrics.items():
        for b, bv in business_outcomes.items():
            correlations[(k, b)] = 0  # placeholder

    # -------------------------
    # Save outputs
    # -------------------------
    features.to_csv("engineered_features.csv")
    results.to_csv("anomaly_results.csv")
    pd.DataFrame([combined_metrics]).to_csv("combined_metrics.csv", index=False)
    pd.DataFrame(list(correlations.items()), columns=["Metric↔Business", "Correlation"]).to_csv("business_correlation.csv", index=False)

    # -------------------------
    # Print schema-style outputs
    # -------------------------
    print("=== Engineered Features ===")
    print(features.head())

    print("\n=== Anomaly Summary ===")
    print(results[["iso_label", "iso_score", "db_cluster"]].value_counts())

    print("\n=== Combined Metrics ===")
    for k, v in combined_metrics.items():
        print(f"  - {k}: {v}")

    print("\n=== Business Correlations ===")
    for (metric, outcome), corr in correlations.items():
        print(f"  {metric} ↔ {outcome}: {corr}")
