import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from collections import Counter
import math
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Log Anomaly Detection", layout="wide")

# -------------------------
# Feature Engineering
# -------------------------
def feature_engineering(logs: pd.DataFrame, window="5T") -> pd.DataFrame:
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors="coerce")
    logs = logs.dropna(subset=['timestamp'])
    logs = logs.set_index('timestamp')

    grouped = logs.resample(window)

    failed_counts = grouped['message'].apply(
        lambda x: x.str.contains("authentication failure", case=False, na=False).sum()
    )

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

    mean_event = features["event_count"].mean() + 1
    features["burstiness"] = features["event_count"] / mean_event

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

def cluster_analysis(features: pd.DataFrame) -> pd.Series:
    clustering = DBSCAN(eps=1.5, min_samples=3).fit(features)
    return pd.Series(clustering.labels_, index=features.index, name="db_cluster")

# -------------------------
# Visualization (Plotly)
# -------------------------
def plot_timeline(results):
    fig = px.scatter(
        results.reset_index(),
        x="timestamp", y="event_count",
        color=results["iso_label"].map({1: "Normal", -1: "Anomaly"}),
        title="Event Timeline with Anomalies",
        labels={"event_count": "Event Count", "timestamp": "Time"},
        color_discrete_map={"Normal": "blue", "Anomaly": "red"}
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap(features):
    corr = features.corr()
    fig = px.imshow(
        corr, text_auto=True, aspect="auto", 
        color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_3d(results):
    fig = px.scatter_3d(
        results.reset_index(),
        x="failed_logins", y="event_count", z="ip_src_entropy",
        color=results["iso_label"].map({1: "Normal", -1: "Anomaly"}),
        title="3D Scatter: Failed Logins vs Event Count vs Entropy",
        labels={"failed_logins": "Failed Logins", "event_count": "Event Count", "ip_src_entropy": "IP Src Entropy"},
        color_discrete_map={"Normal": "blue", "Anomaly": "red"},
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_clusters(results):
    fig = px.scatter(
        results.reset_index(),
        x="event_count", y="failed_logins",
        color=results["db_cluster"].astype(str),
        title="DBSCAN Clusters (Event Count vs Failed Logins)",
        labels={"event_count": "Event Count", "failed_logins": "Failed Logins", "db_cluster": "Cluster ID"},
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🔍 Log Anomaly Detection & Visualization (Plotly)")

uploaded_file = st.file_uploader("Upload your log CSV", type=["csv"])

if uploaded_file:
    logs = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    with st.spinner("Extracting features..."):
        features = feature_engineering(logs)

    with st.spinner("Running anomaly detection..."):
        results = anomaly_detection(features)
        results["db_cluster"] = cluster_analysis(features)

    st.subheader("📊 Engineered Features (Preview)")
    st.dataframe(features.head())

    # Tabs for visualization
    tab1, tab2, tab3, tab4 = st.tabs(["Timeline", "Heatmap", "3D Scatter", "Clusters"])

    with tab1:
        plot_timeline(results)

    with tab2:
        plot_heatmap(features)

    with tab3:
        plot_3d(results)

    with tab4:
        plot_clusters(results)
