import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)
from scipy.cluster.hierarchy import linkage, fcluster
from pathlib import Path
from scipy import stats

# === GENERAL CONFIGURATION ===
n_jobs = -1
directory = r'/homes/3/om871/Documents/ML'

features = ['WPPC', 'WPLI', 'Imag_Coherence', 'Coherence']

# === DISTANCE THRESHOLDS PER METRIC ===
distance_dict = {
    'WPPC': [25, 20, 17, 15, 13],
    'WPLI': [35, 30, 25, 20, 18],
    'Coherence': [25, 20, 17, 15, 13],
    'Imag_Coherence': [35, 30, 25, 20, 18]
}

# === LOAD DATA ===
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
clf_data = pd.read_csv(file_path)
y = clf_data.pop('Population')
X_all = clf_data

interpolation = np.linspace(0, 1, 100)

# === CROSS-VALIDATION ===
outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

# === METRIC FILTERS ===
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# === MAIN LOOP ===
for feature_type in features:

    if feature_type not in distance_dict:
        continue

    for dist_thr in distance_dict[feature_type]:
        print(f"\n=== Running Clustering-PC1 ({feature_type}) at distance {dist_thr} ===")

        results_directory = os.path.join(directory, f'Results/ClusteringPC1/{feature_type}_d{dist_thr}')
        Path(results_directory).mkdir(parents=True, exist_ok=True)

        # Initialize storage
        tprs, precisionspr, rocs, APscores = [], [], [], []
        HC_precision, HC_recall, HC_f1 = [], [], []
        MDD_precision, MDD_recall, MDD_f1 = [], [], []
        accuracy, brier_scores = [], []

        # Filter metric subset
        if feature_type in filters:
            X = filters[feature_type](X_all)
        else:
            continue

        # === CV LOOP ===
        for train_ix, test_ix in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            # --- Standardize ---
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # --- Hierarchical clustering (variables) ---
            Z = linkage(X_train_scaled.T, method='ward', metric='euclidean')
            clusters = fcluster(Z, t=dist_thr, criterion='distance')
            cluster_groups = {
                k: X.columns[clusters == k].tolist()
                for k in np.unique(clusters)
            }

            # --- PCA: get PC1 for each cluster ---
            X_train_pc1 = pd.DataFrame(index=X_train.index)
            X_test_pc1 = pd.DataFrame(index=X_test.index)
            for c, cols in cluster_groups.items():
                if len(cols) < 2:
                    X_train_pc1[f"C{c}"] = X_train_scaled[:, X.columns.get_indexer(cols)]
                    X_test_pc1[f"C{c}"] = X_test_scaled[:, X.columns.get_indexer(cols)]
                else:
                    pca = PCA(n_components=1, random_state=42)
                    pca.fit(X_train_scaled[:, X.columns.get_indexer(cols)])
                    X_train_pc1[f"C{c}"] = pca.transform(X_train_scaled[:, X.columns.get_indexer(cols)]).ravel()
                    X_test_pc1[f"C{c}"] = pca.transform(X_test_scaled[:, X.columns.get_indexer(cols)]).ravel()

            # --- Train RF on synthetic features ---
            rf_final = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=n_jobs)
            rf_final.fit(X_train_pc1, y_train)

            probas_ = rf_final.predict_proba(X_test_pc1)
            y_pred = rf_final.predict(X_test_pc1)

            # === METRICS (identical to your original script) ===
            # ROC
            fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
            tprs.append(np.interp(interpolation, fpr, tpr))
            tprs[-1][0] = 0.0
            rocs.append(auc(fpr, tpr))

            # PR
            precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
            APscores.append(average_precision_score(y_test, probas_[:, 1]))
            precisionspr.append(np.interp(interpolation, recall[::-1], precision[::-1]))
            precisionspr[-1][0] = 1.0

            # Classification report
            report = classification_report(y_test, y_pred, target_names=['HC', 'MDD'], output_dict=True)
            HC_precision.append(report['HC']['precision'])
            HC_recall.append(report['HC']['recall'])
            HC_f1.append(report['HC']['f1-score'])
            MDD_precision.append(report['MDD']['precision'])
            MDD_recall.append(report['MDD']['recall'])
            MDD_f1.append(report['MDD']['f1-score'])
            accuracy.append(report['accuracy'])
            brier_scores.append(brier_score_loss(y_test, probas_[:, 1]))

        # === Confidence interval function (same) ===
        def ci95_bf(values, k=5, m=20, n2_n1=0.25, alpha=0.05):
            values = np.array(values)
            mean = np.mean(values)
            sd = np.std(values, ddof=1)
            se = sd * np.sqrt((1 / (k * m)) + n2_n1)
            df = k * m - 1
            tval = stats.t.ppf(1 - alpha/2, df=df)
            ci_lower = mean - tval * se
            ci_upper = mean + tval * se
            return mean, sd, ci_lower, ci_upper

        # === Summary statistics ===
        metrics_summary = []
        metric_names = [
            ("ROC AUC", rocs),
            ("PR AP", APscores),
            ("Accuracy", accuracy),
            ("HC Precision", HC_precision),
            ("HC Recall", HC_recall),
            ("HC F1", HC_f1),
            ("MDD Precision", MDD_precision),
            ("MDD Recall", MDD_recall),
            ("MDD F1", MDD_f1),
            ("Brier Score", brier_scores)
        ]

        for name, values in metric_names:
            mean, sd, ci_lower, ci_upper = ci95_bf(values)
            metrics_summary.append([name, mean, sd, ci_lower, ci_upper])

        metrics_df = pd.DataFrame(metrics_summary, columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"])
        metrics_df.to_excel(os.path.join(results_directory, f"{feature_type}_metrics.xlsx"), index=False)

        # === SAVE ROC AND PR CURVES ===
        mean_tpr = np.mean(tprs, axis=0)
        lower_bound = np.percentile(tprs, 2.5, axis=0)
        upper_bound = np.percentile(tprs, 97.5, axis=0)

        roc_df = pd.DataFrame({
            "FPR": interpolation,
            "TPR_mean": mean_tpr,
            "TPR_lower": lower_bound,
            "TPR_upper": upper_bound
        })
        roc_df.to_excel(os.path.join(results_directory, f"{feature_type}_roc_curve.xlsx"), index=False)

        mean_precision = np.mean(precisionspr, axis=0)
        lower_pr = np.percentile(precisionspr, 2.5, axis=0)
        upper_pr = np.percentile(precisionspr, 97.5, axis=0)

        pr_df = pd.DataFrame({
            "Recall": interpolation,
            "Precision_mean": mean_precision,
            "Precision_lower": lower_pr,
            "Precision_upper": upper_pr
        })
        pr_df.to_excel(os.path.join(results_directory, f"{feature_type}_pr_curve.xlsx"), index=False)

        print(f"✅ Finished {feature_type} (distance={dist_thr}). Results saved in: {results_directory}")
