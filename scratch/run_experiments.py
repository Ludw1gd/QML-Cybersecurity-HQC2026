import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, confusion_matrix

from qiskit.circuit.library import zz_feature_map
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

FILE = "data/SUPER_ENRICHED_hybrid_dataset_v3.parquet"
COLUMNS = [
    "flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "flow_byts/s",
    "flow_pkts/s", "pkt_len_mean", "qber", "sifted_key_rate",
    "is_anomalous_qkd", "label"
]

print("Loading data with column projection...")
df = pd.read_parquet(FILE, columns=COLUMNS)
df = df.replace([np.inf, -np.inf], np.nan).dropna()
df["target"] = (df["label"] != "Benign").astype(int)
print(f"Data shape after cleaning: {df.shape}")

configs = {
    "Exp_A_Classical": ["flow_duration", "tot_fwd_pkts", "tot_bwd_pkts", "flow_byts/s"],
    "Exp_B_QKD_Pure": ["qber", "sifted_key_rate"],
    "Exp_C_Hybrid_Honest": ["flow_pkts/s", "pkt_len_mean", "qber", "sifted_key_rate"],
    "Exp_D_Leakage_Demo": ["flow_pkts/s", "pkt_len_mean", "qber", "sifted_key_rate", "is_anomalous_qkd"],
}

seeds = [42, 43, 44, 45, 46]

def sample_each(frame, n, seed):
    return (frame.groupby("target", group_keys=False)
            .sample(n=n, random_state=seed)
            .sample(frac=1, random_state=seed))

results = []

print("Starting Ablation Multi-Seed Evaluation...")
start_all = time.time()

for exp_name, features in configs.items():
    print(f"\n--- Running {exp_name} (Features: {features}) ---")
    for s in seeds:
        # Stratified train/test split on pool first
        train_pool, test_pool = train_test_split(
            df, test_size=0.20, stratify=df["target"], random_state=s
        )
        # Balanced subsampling
        train_sub = sample_each(train_pool, 24, s)
        test_sub = sample_each(test_pool, 12, s)

        X_train_raw = train_sub[features].to_numpy()
        y_train = train_sub["target"].to_numpy()
        X_test_raw = test_sub[features].to_numpy()
        y_test = test_sub["target"].to_numpy()

        # Scaling strictly on train
        scaler = MinMaxScaler(feature_range=(0, np.pi))
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        # 1. Classical RBF-SVM
        svm = SVC(kernel="rbf", class_weight="balanced")
        t0 = time.time()
        svm.fit(X_train, y_train)
        svm_fit_t = time.time() - t0

        t0 = time.time()
        svm_preds = svm.predict(X_test)
        svm_pred_t = time.time() - t0

        svm_bacc = balanced_accuracy_score(y_test, svm_preds)
        svm_f1 = f1_score(y_test, svm_preds, average="macro")
        svm_rec = recall_score(y_test, svm_preds, pos_label=1)
        cm_svm = confusion_matrix(y_test, svm_preds)

        results.append({
            "experiment": exp_name,
            "model": "RBF-SVM",
            "seed": s,
            "n_qubits": len(features),
            "balanced_accuracy": svm_bacc,
            "macro_f1": svm_f1,
            "attack_recall": svm_rec,
            "fit_time": svm_fit_t,
            "predict_time": svm_pred_t,
            "confusion_matrix": cm_svm.tolist()
        })

        # 2. Quantum QSVC
        fmap = zz_feature_map(feature_dimension=len(features), reps=1, entanglement="linear")
        qkernel = FidelityQuantumKernel(feature_map=fmap)
        qsvc = QSVC(quantum_kernel=qkernel)

        t0 = time.time()
        qsvc.fit(X_train, y_train)
        qsvc_fit_t = time.time() - t0

        t0 = time.time()
        qsvc_preds = qsvc.predict(X_test)
        qsvc_pred_t = time.time() - t0

        qsvc_bacc = balanced_accuracy_score(y_test, qsvc_preds)
        qsvc_f1 = f1_score(y_test, qsvc_preds, average="macro")
        qsvc_rec = recall_score(y_test, qsvc_preds, pos_label=1)
        cm_qsvc = confusion_matrix(y_test, qsvc_preds)

        results.append({
            "experiment": exp_name,
            "model": "QSVC",
            "seed": s,
            "n_qubits": len(features),
            "balanced_accuracy": qsvc_bacc,
            "macro_f1": qsvc_f1,
            "attack_recall": qsvc_rec,
            "fit_time": qsvc_fit_t,
            "predict_time": qsvc_pred_t,
            "confusion_matrix": cm_qsvc.tolist()
        })

        print(f"  Seed {s} -> SVM: F1={svm_f1:.3f}, QSVC: F1={qsvc_f1:.3f} (Fit QSVC: {qsvc_fit_t:.1f}s, Pred QSVC: {qsvc_pred_t:.1f}s)")

total_t = time.time() - start_all
print(f"\nAll experiments finished in {total_t:.2f} seconds ({total_t/60:.2f} minutes).")

res_df = pd.DataFrame(results)
res_df.to_json("scratch/experiment_results.json", orient="records", indent=2)

summary = res_df.groupby(["experiment", "model"])[["balanced_accuracy", "macro_f1", "attack_recall", "fit_time", "predict_time"]].agg(["mean", "std"])
print("\n--- Summary (Mean ± Std) ---")
print(summary)
summary.to_csv("scratch/experiment_summary.csv")
