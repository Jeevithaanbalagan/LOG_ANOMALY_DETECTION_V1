import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import plotly.express as px
import hashlib
import time

# ----------------- Streamlit Config -----------------
st.set_page_config(page_title="SOC Dashboard", layout="wide")

st.title("🛡️ Security Operations Center (SOC) Dashboard")
st.markdown("Monitor logs, detect anomalies, and correlate with business impact in real-time.")

# ----------------- File Upload -----------------
uploaded_file = st.file_uploader("📂 Upload your log.csv", type=["csv"])

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh every 10s", value=False)
refresh_rate = 10

if uploaded_file:
    logs = pd.read_csv(uploaded_file)

    # ----------------- Window Selector -----------------
    window_choice = st.sidebar.selectbox(
        "Select Time Window for Analysis",
        options=["1min", "5min", "15min", "30min", "1H"],
        index=1
    )
    st.sidebar.write(f"Using **{window_choice}** window for feature grouping.")

    # ----------------- Preprocessing -----------------
    logs['timestamp'] = pd.to_datetime(logs['timestamp'], errors='coerce')
    logs = logs.dropna(subset=['timestamp'])
    logs = logs.sort_values('timestamp')

    # Hash user/IP for anonymization
    def safe_hash(val):
        return hashlib.sha1(str(val).encode()).hexdigest()[:8] if pd.notna(val) else "NA"
    logs['ip_src_hash'] = logs['ip_src'].apply(safe_hash)
    logs['ip_dst_hash'] = logs['ip_dst'].apply(safe_hash)

    # ----------------- Feature Engineering -----------------
    logs.set_index('timestamp', inplace=True)
    grouped = logs.resample(window_choice)  # <-- dynamic window

    features = pd.DataFrame({
        "event_count": grouped.size(),
        "unique_ips": grouped['ip_src'].nunique(),
        "failed_logins": grouped['message'].apply(lambda x: x.str.contains("failure", case=False, na=False).sum()),
    }).fillna(0)

    # Burstiness + time-of-day encoding
    features['burstiness'] = features['event_count'] / (features['event_count'].mean() + 1)
    features['hour'] = features.index.hour
    features['tod_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['tod_cos'] = np.cos(2 * np.pi * features['hour'] / 24)

    # ----------------- Anomaly Detection -----------------
    iso = IsolationForest(contamination=0.05, random_state=42)
    features['iso_label'] = iso.fit_predict(features[['event_count', 'unique_ips', 'failed_logins',
                                                      'burstiness', 'tod_sin', 'tod_cos']])
    features['iso_score'] = iso.decision_function(features[['event_count', 'unique_ips', 'failed_logins',
                                                            'burstiness', 'tod_sin', 'tod_cos']])

    anomalies = features[features['iso_label'] == -1]

    # ----------------- DBSCAN Clustering -----------------
    db = DBSCAN(eps=0.5, min_samples=5).fit(
        features[['event_count', 'unique_ips', 'failed_logins']]
    )
    features['db_cluster'] = db.labels_
    anomalies = features[features['iso_label'] == -1]

    # ----------------- Summary Metrics -----------------
    st.subheader("📊 Summary Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Log Windows", len(features))
    col2.metric("Anomalous Windows", anomalies.shape[0])
    col3.metric("Anomaly %", f"{100 * anomalies.shape[0] / len(features):.2f}%")

    # ----------------- Live Alerts Panel -----------------
    st.subheader("🚨 Live Anomaly Alerts")
    if anomalies.shape[0] > 0:
        for ts, row in anomalies.tail(5).iterrows():
            st.error(f"**{ts}** | Event Count: {row['event_count']} | Failed Logins: {row['failed_logins']} | Cluster: {row['db_cluster']}")
    else:
        st.success("✅ No anomalies detected in the latest logs.")

    # ----------------- Combined Metrics -----------------
    log_metrics = {
        "avg_event_rate": features["event_count"].mean(),
        "anomaly_rate": (features['iso_label'] == -1).mean(),
        "avg_failed_logins": features["failed_logins"].mean()
    }

    nonlog_metrics = {
        "patch_compliance": 0.85,
        "MFA_coverage": 0.9,
        "backup_success_rate": 0.95
    }

    combined_metrics = {**log_metrics, **nonlog_metrics}

    # ----------------- Business Correlation -----------------
    business_outcomes = {
        "uptime": 0.995,
        "quarterly_revenue_growth": 0.04
    }

    correlations = {}
    for k, v in combined_metrics.items():
        try:
            for b, bv in business_outcomes.items():
                correlations[(k, b)] = round(np.corrcoef([v], [bv])[0, 1], 2)
        except Exception:
            correlations[(k, b)] = 0

    # ----------------- Visualization Tabs -----------------
    st.subheader("📈 Visualizations")
    tabs = st.tabs([
        "Timeline", "Heatmap", "3D Scatter", "Cluster View",
        "Burstiness", "Business Correlation", "Raw Logs"
    ])

    with tabs[0]:
        st.markdown("### Event Count Over Time (with anomalies)")
        fig = px.line(features, x=features.index, y="event_count", title="Event Timeline")
        anomaly_points = features[features['iso_label'] == -1]
        fig.add_scatter(x=anomaly_points.index, y=anomaly_points['event_count'],
                        mode='markers', marker=dict(color='red', size=10), name="Anomalies")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.markdown("### Failed Logins Heatmap by Hour/Day")
        heatmap_data = features.copy()
        heatmap_data['day'] = heatmap_data.index.day
        fig = px.density_heatmap(heatmap_data, x="hour", y="day", z="failed_logins", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.markdown("### 3D Anomaly Scatter Plot")
        fig3d = px.scatter_3d(features, x="event_count", y="unique_ips", z="failed_logins",
                               color=features['db_cluster'].astype(str),
                               title="3D Cluster Space")
        st.plotly_chart(fig3d, use_container_width=True)

    with tabs[3]:
        st.markdown("### Cluster Distribution")
        fig_cluster = px.scatter(features, x="event_count", y="failed_logins",
                                 color=features['db_cluster'].astype(str),
                                 title="DBSCAN Clusters")
        st.plotly_chart(fig_cluster, use_container_width=True)

    with tabs[4]:
        st.markdown("### Burstiness Over Time")
        fig_burst = px.line(features, x=features.index, y="burstiness", title="Burstiness Timeline")
        st.plotly_chart(fig_burst, use_container_width=True)

    with tabs[5]:
        st.markdown("### Business Correlation")
        corr_df = pd.DataFrame([
            {"Metric": k, "Business KPI": b, "Correlation": v}
            for (k, b), v in correlations.items()
        ])
        st.dataframe(corr_df)
        fig_corr = px.bar(corr_df, x="Metric", y="Correlation", color="Business KPI", title="Correlation of Metrics with Business KPIs")
        st.plotly_chart(fig_corr, use_container_width=True)

    with tabs[6]:
        st.markdown("### Raw Log Samples")
        st.dataframe(logs.head(50))

    # ----------------- Download Option -----------------
    st.subheader("⬇️ Download Processed Data")
    st.download_button("Download Features + Labels CSV",
                       data=features.to_csv().encode('utf-8'),
                       file_name="features_report.csv",
                       mime="text/csv")

    # ----------------- Auto Refresh -----------------
    if auto_refresh:
        time.sleep(refresh_rate)
        st.experimental_rerun()

else:
    st.warning("⚠️ Please upload a log.csv file to begin analysis.")
