import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix

FILE = "data/final_hybrid_cybersecurity_dataset.parquet"
FEATURES = ["flow_pkts/s", "pkt_len_mean", "qber", "sifted_key_rate"]

print("Loading final_hybrid_cybersecurity_dataset.parquet for temporal experiment...")
base = pd.read_parquet(FILE, columns=FEATURES + ["label"])
print(f"Index name: {base.index.name}")
base = base.reset_index() # timestamp becomes column
base = base.replace([np.inf, -np.inf], np.nan).dropna()
base["target"] = (base["label"] != "Benign").astype(int)
base = base.sort_values("timestamp")

# Temporal split: 80% train (earlier), 20% test (later)
n_total = len(base)
split_idx = int(0.80 * n_total)
split_time = base.iloc[split_idx]["timestamp"]
print(f"Temporal cutoff time: {split_time}")

train_pool = base.iloc[:split_idx]
test_pool = base.iloc[split_idx:]

print(f"Train pool: {len(train_pool)} rows (Benign: {(train_pool['target']==0).sum()}, Attack: {(train_pool['target']==1).sum()})")
print(f"Test pool: {len(test_pool)} rows (Benign: {(test_pool['target']==0).sum()}, Attack: {(test_pool['target']==1).sum()})")

def sample_each(frame, n, seed=42):
    return (frame.groupby("target", group_keys=False)
            .sample(n=n, random_state=seed)
            .sample(frac=1, random_state=seed))

# Subsample balanced for evaluation
train_sub = sample_each(train_pool, 24, seed=42)
test_sub = sample_each(test_pool, 12, seed=42)

X_train_raw = train_sub[FEATURES].to_numpy()
y_train = train_sub["target"].to_numpy()
X_test_raw = test_sub[FEATURES].to_numpy()
y_test = test_sub["target"].to_numpy()

scaler = MinMaxScaler(feature_range=(0, np.pi))
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

svm = SVC(kernel="rbf", class_weight="balanced")
svm.fit(X_train, y_train)
preds = svm.predict(X_test)

bacc = balanced_accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="macro")
rec = recall_score(y_test, preds, pos_label=1)
cm = confusion_matrix(y_test, preds)

print("\n--- Temporal Split Baseline Results (RBF-SVM) ---")
print(f"Balanced Accuracy: {bacc:.4f}")
print(f"Macro F1: {f1:.4f}")
print(f"Attack Recall: {rec:.4f}")
print("Confusion Matrix:\n", cm)

# Also test full test set (unbalanced real distribution)
X_test_full_raw = test_pool[FEATURES].to_numpy()
y_test_full = test_pool["target"].to_numpy()
X_test_full = scaler.transform(X_test_full_raw)
preds_full = svm.predict(X_test_full)

bacc_full = balanced_accuracy_score(y_test_full, preds_full)
f1_full = f1_score(y_test_full, preds_full, average="macro")
rec_full = recall_score(y_test_full, preds_full, pos_label=1)
cm_full = confusion_matrix(y_test_full, preds_full)

print("\n--- Temporal Split on FULL Unseen Future Traffic ---")
print(f"Balanced Accuracy: {bacc_full:.4f}")
print(f"Macro F1: {f1_full:.4f}")
print(f"Attack Recall: {rec_full:.4f}")
print("Confusion Matrix:\n", cm_full)
