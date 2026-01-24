import pandas as pd
from datetime import datetime, timedelta
import random

# Simulate synthetic user activity logs
users = ["alice", "bob", "charlie", "david"]
events = ["login", "file_download", "email_send", "file_delete"]
data = []

for _ in range(200):
    data.append({
        "timestamp": datetime.now() - timedelta(minutes=random.randint(0, 1000)),
        "user": random.choice(users),
        "event": random.choice(events),
        "bytes": random.randint(0, 50000)
    })

logs = pd.DataFrame(data)
print(logs.head())
logs["timestamp"] = pd.to_datetime(logs["timestamp"]) 
logs["date"] = logs["timestamp"].dt.date 
# Aggregate features 
features = logs.groupby(["user", "date"]).agg( 
total_events=("event", "size"), 
unique_events=("event", "nunique"), 
total_bytes=("bytes", "sum"), 
max_bytes=("bytes", "max") 
).reset_index() 
print(features.head()) 
baselines = features.groupby("user").agg( 
avg_events=("total_events", "mean"), 
avg_bytes=("total_bytes", "mean") 
).reset_index() 
# Merge baselines back to features 
features = features.merge(baselines, on="user") 
features["event_dev"] = features["total_events"] - features["avg_events"] 
features["byte_dev"] = features["total_bytes"] - features["avg_bytes"] 
print(features.head()) 
from sklearn.ensemble import IsolationForest 
from sklearn.preprocessing import StandardScaler 
X = features[["total_events", "unique_events", "total_bytes", "max_bytes"]] 
# Normalize features 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X) 
# Train unsupervised anomaly detector 
model = IsolationForest(contamination=0.1, random_state=42) 
features["anomaly"] = model.fit_predict(X_scaled) 
features["score"] = -model.decision_function(X_scaled) 
print(features[["user", "date", "score", "anomaly"]].head()) 
# Threshold for high risk users 
threshold = features["score"].quantile(0.90) 
# Assign risk levels 
features["risk_level"] = features["score"].apply( 
lambda x: "High" if x > threshold else "Normal" 
) 
alerts = features[features["risk_level"] == "High"] 
print("High Risk Users Detected:\n", alerts[["user", "date","score"]])