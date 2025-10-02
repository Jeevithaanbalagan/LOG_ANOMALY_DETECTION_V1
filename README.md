# Project Workflow: Log Anomaly Detection and Clustering

This project is designed to process log data, extract meaningful features, detect anomalies, and visualize results using clustering and machine learning techniques. The workflow is modular and covers both streaming and batch analysis, as well as interactive visualization.

## Workflow Overview

### 1. Log Ingestion
- Log data is ingested from sources such as Kafka topics or CSV files.
- For streaming, a Kafka consumer reads log lines in real time.
- For batch, logs are loaded from files for offline analysis.

### 2. Feature Engineering
- Raw logs are parsed and transformed into feature vectors.
- Features include event counts, unique users/IPs, burstiness, failed login counts, IP entropy, and time-of-day encodings.
- Data is resampled into time windows (e.g., 5 minutes) for aggregation.

### 3. Embedding Generation (Optional)
- For advanced analysis, log lines can be embedded using BERT models.
- Embeddings are generated using `BertTokenizer` and `BertModel` from the `transformers` library.

### 4. Anomaly Detection
- Isolation Forest is used to flag anomalous time windows or log entries.
- The model assigns anomaly labels and scores to each window.
- In streaming mode, River's HalfSpaceTrees can be used for online anomaly scoring and learning.

### 5. Clustering
- DBSCAN is applied to feature vectors to discover clusters of similar log behavior.
- Cluster labels are assigned to each time window or log entry.

### 6. Visualization
- Multiple visualization functions are provided:
  - **Timeline Plot:** Shows event counts over time, highlighting anomalies.
  - **Correlation Heatmap:** Displays feature correlations for exploratory analysis.
  - **3D Scatter Plot:** Visualizes relationships between failed logins, event counts, and IP entropy.
  - **Cluster Plot:** Shows DBSCAN clusters in feature space.

### 7. Alerting
- When anomalies are detected, alerts can be sent to Slack via webhook integration.
- Normal logs are saved to a file for further review.

### 8. Model Retraining
- In streaming mode, the batch anomaly model is periodically retrained as new data accumulates.
- Models are persisted using `joblib` for reuse and evaluation.

## Summary
This project provides a robust pipeline for log analysis, anomaly detection, clustering, and visualization. It supports both real-time and batch workflows, leverages state-of-the-art NLP and ML models, and includes alerting and retraining mechanisms for continuous improvement.