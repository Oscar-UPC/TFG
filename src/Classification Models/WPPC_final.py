import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss,
    confusion_matrix
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
        directory, f'Results/final/RFE/RFE_{n_features}_features'
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

                # Confusion matrix accumulator
                conf_mat_sum = np.zeros((2, 2))

                # PR baseline like your old script
                baseline_pr = float(np.mean(y))

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

                    # Confusion matrix per fold
                    conf_mat_sum += confusion_matrix(y_test, y_pred, labels=[0, 1])

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
                    brier_scores.append(brier_score_loss(y_test, probas_[:, 1]))

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

                metrics_summary = [[name, *ci95_bf(values)] for name, values in metric_names]

                metrics_df = pd.DataFrame(
                    metrics_summary,
                    columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"]
                )
                metrics_df.to_excel(
                    os.path.join(feature_directory, f"{g}_metrics.xlsx"),
                    index=False
                )

                # === ROC FIGURE ===
                mean_tpr = np.mean(np.array(tprs), axis=0)
                mean_auc = float(np.mean(rocs))

                plt.figure(figsize=(6, 6))
                plt.plot(interpolation, mean_tpr, lw=2, label=f"Mean ROC ({g})")
                plt.plot([0, 1], [0, 1], linestyle="--", lw=1, label="Luck")
                plt.text(0.05, 0.95, f"Mean AUC: {mean_auc:.2f}",
                         transform=plt.gca().transAxes, fontsize=10, va="top")
                plt.xlabel("FPR (1 - Specificity)")
                plt.ylabel("TPR (Sensitivity)")
                plt.title(f"ROC Curve - {g}")
                plt.legend(loc="lower right", fontsize=9)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(feature_directory, f"{g}_ROC_curve.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close()

                # === PR FIGURE ===
                mean_precision = np.mean(np.array(precisionspr), axis=0)
                mean_ap = float(np.mean(APscores))

                plt.figure(figsize=(6, 6))
                plt.plot(interpolation, mean_precision, lw=2, label=f"Mean PR ({g})")
                plt.axhline(baseline_pr, linestyle="--", lw=1,
                            label="Baseline (Random Classifier)")
                plt.text(0.05, 0.95, f"Mean AP: {mean_ap:.2f}",
                         transform=plt.gca().transAxes, fontsize=10, va="top")
                plt.xlabel("Recall (Sensitivity)")
                plt.ylabel("Precision")
                plt.title(f"PR Curve - {g}")
                plt.legend(loc="lower right", fontsize=9)
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(feature_directory, f"{g}_PR_curve.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close()

                # === MEAN CONFUSION MATRIX ===
                n_folds_total = outer_cv.get_n_splits()
                conf_mat_mean = conf_mat_sum / n_folds_total

                plt.figure(figsize=(5, 4))
                plt.imshow(conf_mat_mean)
                plt.colorbar()

                class_names = ["HC", "MDD"]
                plt.xticks([0, 1], class_names)
                plt.yticks([0, 1], class_names)

                plt.xlabel("Predicted label")
                plt.ylabel("True label")
                plt.title(f"Mean Confusion Matrix – {g}")

                for i in range(2):
                    for j in range(2):
                        plt.text(j, i, f"{conf_mat_mean[i, j]:.2f}",
                                 ha="center", va="center")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(feature_directory, f"{g}_mean_confusion_matrix.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close()

            print(f"✅ Finished RF config {combo_tag}")

    print(f"\n✅ Finished RFE with {n_features} features (all final RF combinations)")
