import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)
from pathlib import Path
from scipy import stats


# === GENERAL CONFIGURATION ===
n_jobs = -1
directory = r'/homes/3/om871/Documents/ML'
features = ['WPPC']

# === RF HYPERPARAMETER GRID (FINAL CLASSIFIER ONLY) ===
max_depth_list = [None]
min_samples_leaf_list = [1]

# === LOAD DATA ===
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
clf_data = pd.read_csv(file_path)

# HC = 0, MDD = 1
y = clf_data.pop('Population')
X_all = clf_data

interpolation = np.linspace(0, 1, 100)

# === CROSS-VALIDATION ===
outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

# Number of selected features
n_features_to_select_list = [30]

# Feature filters
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}


# === CI FUNCTION ===
def ci95_bf(values, k=5, m=20, n2_n1=0.25, alpha=0.05):
    values = np.array(values)
    mean = np.mean(values)
    sd = np.std(values, ddof=1)
    se = sd * np.sqrt((1 / (k * m)) + n2_n1)
    df = k * m - 1
    tval = stats.t.ppf(1 - alpha / 2, df=df)
    return mean, sd, mean - tval * se, mean + tval * se


# === MAIN LOOP ===
for n_features in n_features_to_select_list:

    print(f"\n=== Running RFE with {n_features} features ===")

    base_results_directory = os.path.join(
        directory, f'Results/RFE/RFE_{n_features}_features'
    )
    Path(base_results_directory).mkdir(parents=True, exist_ok=True)

    # --- Loop over RF hyperparameters (FINAL MODEL ONLY) ---
    for max_depth in max_depth_list:
        for min_leaf in min_samples_leaf_list:

            depth_tag = "None" if max_depth is None else str(max_depth)
            combo_tag = f"RF_depth-{depth_tag}_leaf-{min_leaf}"

            results_directory = os.path.join(base_results_directory, combo_tag)
            Path(results_directory).mkdir(parents=True, exist_ok=True)

            print(f"\n--- Final RF config: max_depth={max_depth}, min_samples_leaf={min_leaf} ---")

            for g in features:
                print(f" → Processing feature type: {g}")

                X = filters[g](X_all)
                feature_directory = os.path.join(results_directory, g)
                Path(feature_directory).mkdir(parents=True, exist_ok=True)

                tprs, precisionspr, rocs, APscores = [], [], [], []
                HC_precision, HC_recall, HC_f1 = [], [], []
                MDD_precision, MDD_recall, MDD_f1 = [], [], []
                accuracy, brier_scores = [], []

                # === CV LOOP ===
                for train_ix, test_ix in outer_cv.split(X, y):
                    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

                    # --- RFE (FIXED RF) ---
                    rf_rfe = RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        n_jobs=n_jobs
                    )

                    rfe = RFE(
                        estimator=rf_rfe,
                        n_features_to_select=n_features,
                        step=0.1
                    )
                    rfe.fit(X_train, y_train)

                    X_train_sel = rfe.transform(X_train)
                    X_test_sel = rfe.transform(X_test)

                    # --- Final RF (OPTIMIZED) ---
                    rf_final = RandomForestClassifier(
                        n_estimators=500,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        random_state=42,
                        n_jobs=n_jobs
                    )
                    rf_final.fit(X_train_sel, y_train)

                    probas_ = rf_final.predict_proba(X_test_sel)
                    y_pred = rf_final.predict(X_test_sel)

                    # === ROC ===
                    fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
                    tprs.append(np.interp(interpolation, fpr, tpr))
                    tprs[-1][0] = 0.0
                    rocs.append(auc(fpr, tpr))

                    # === PR ===
                    precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
                    APscores.append(average_precision_score(y_test, probas_[:, 1]))
                    precisionspr.append(
                        np.interp(interpolation, recall[::-1], precision[::-1])
                    )
                    precisionspr[-1][0] = 1.0

                    # === METRICS ===
                    report = classification_report(
                        y_test, y_pred,
                        target_names=['HC', 'MDD'],
                        output_dict=True
                    )
                    HC_precision.append(report['HC']['precision'])
                    HC_recall.append(report['HC']['recall'])
                    HC_f1.append(report['HC']['f1-score'])
                    MDD_precision.append(report['MDD']['precision'])
                    MDD_recall.append(report['MDD']['recall'])
                    MDD_f1.append(report['MDD']['f1-score'])
                    accuracy.append(report['accuracy'])
                    brier_scores.append(
                        brier_score_loss(y_test, probas_[:, 1])
                    )

                # === SAVE METRICS ===
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

                metrics_summary = [
                    [name, *ci95_bf(values)]
                    for name, values in metric_names
                ]

                metrics_df = pd.DataFrame(
                    metrics_summary,
                    columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"]
                )
                metrics_df.to_excel(
                    os.path.join(feature_directory, f"{g}_metrics.xlsx"),
                    index=False
                )

                # === ROC / PR CURVES ===
                tprs_arr = np.array(tprs)
                precisions_arr = np.array(precisionspr)

                roc_df = pd.DataFrame({
                    "FPR": interpolation,
                    "TPR_mean": tprs_arr.mean(axis=0),
                    "TPR_lower": np.percentile(tprs_arr, 2.5, axis=0),
                    "TPR_upper": np.percentile(tprs_arr, 97.5, axis=0)
                })
                roc_df.to_excel(
                    os.path.join(feature_directory, f"{g}_roc_curve.xlsx"),
                    index=False
                )

                pr_df = pd.DataFrame({
                    "Recall": interpolation,
                    "Precision_mean": precisions_arr.mean(axis=0),
                    "Precision_lower": np.percentile(precisions_arr, 2.5, axis=0),
                    "Precision_upper": np.percentile(precisions_arr, 97.5, axis=0)
                })
                pr_df.to_excel(
                    os.path.join(feature_directory, f"{g}_pr_curve.xlsx"),
                    index=False
                )

            print(f"✅ Finished RF config {combo_tag}")

    print(f"\n✅ Finished RFE with {n_features} features (all final RF combinations)")
