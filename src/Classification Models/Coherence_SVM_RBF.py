import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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
features = ['Coherence']

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

# === NUMBER OF FINAL FEATURES (RFE output) ===
n_features_to_select_list = [30]

# === FEATURE FILTERS PER CONNECTIVITY METRIC ===
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# === GRID FOR SVM RBF ===
C_values = [0.1, 1.0, 10]
gamma_values = [0.01, 0.1, 1]


# =====================================================================
# === MAIN LOOP =======================================================
# =====================================================================
for n_features in n_features_to_select_list:

    print(f"\n=== Running ANOVA + RFE + SVM RBF with {n_features} features ===")

    results_directory = os.path.join(
        directory,
        f'Results/RFE_prefilter_SVM/RFE_prefilter_{n_features}_features'
    )
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for g in features:
        print(f" → Processing feature type: {g}")

        if g in filters:
            X = filters[g](X_all)

        # Loop over SVM hyperparameters
        for C_val in C_values:
            for gamma_val in gamma_values:

                print(f"     → SVM RBF | C = {C_val} | gamma = {gamma_val}")

                model_directory = os.path.join(
                    results_directory, g, f"SVM_C_{C_val}_gamma_{gamma_val}"
                )
                Path(model_directory).mkdir(parents=True, exist_ok=True)

                # Storage
                tprs, precisionspr, rocs, APscores = [], [], [], []
                HC_precision, HC_recall, HC_f1 = [], [], []
                MDD_precision, MDD_recall, MDD_f1 = [], [], []
                accuracy, brier_scores = [], []

                # === CROSS-VALIDATION LOOP ===
                for train_ix, test_ix in outer_cv.split(X, y):
                    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

                    # === 1) ANOVA PREFILTER ===
                    skb = SelectKBest(score_func=f_classif, k=min(200, X_train.shape[1]))
                    X_train_pref = skb.fit_transform(X_train, y_train)
                    X_test_pref = skb.transform(X_test)

                    # === 2) RFE with RANDOM FOREST (corregido) ===
                    rfe = RFE(
                        estimator=RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1
                        ),
                        n_features_to_select=n_features,
                        step=0.1
                    )
                    rfe.fit(X_train_pref, y_train)

                    X_train_sel = rfe.transform(X_train_pref)
                    X_test_sel = rfe.transform(X_test_pref)

                    # === 3) FINAL MODEL: SVM RBF + StandardScaler ===
                    clf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("svm", SVC(
                            kernel='rbf',
                            C=C_val,
                            gamma=gamma_val,
                            probability=True  # Required for ROC/PR
                        ))
                    ])

                    clf.fit(X_train_sel, y_train)

                    # === 4) PREDICTIONS ===
                    probas_ = clf.predict_proba(X_test_sel)
                    y_pred = clf.predict(X_test_sel)

                    # === 5) ROC CURVE ===
                    fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
                    interp_tpr = np.interp(interpolation, fpr, tpr)
                    interp_tpr[0] = 0.0
                    tprs.append(interp_tpr)
                    rocs.append(auc(fpr, tpr))

                    # === 6) PR CURVE ===
                    precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
                    APscores.append(average_precision_score(y_test, probas_[:, 1]))
                    interp_prec = np.interp(interpolation, recall[::-1], precision[::-1])
                    interp_prec[0] = 1.0
                    precisionspr.append(interp_prec)

                    # === 7) CLASSIFICATION REPORT ===
                    report = classification_report(
                        y_test, y_pred, target_names=['HC', 'MDD'], output_dict=True
                    )
                    HC_precision.append(report['HC']['precision'])
                    HC_recall.append(report['HC']['recall'])
                    HC_f1.append(report['HC']['f1-score'])
                    MDD_precision.append(report['MDD']['precision'])
                    MDD_recall.append(report['MDD']['recall'])
                    MDD_f1.append(report['MDD']['f1-score'])
                    accuracy.append(report['accuracy'])
                    brier_scores.append(brier_score_loss(y_test, probas_[:, 1]))

                # === CONFIDENCE INTERVAL FUNCTION ===
                def ci95_bf(values, k=5, m=20, n2_n1=0.25, alpha=0.05):
                    values = np.array(values)
                    mean = np.mean(values)
                    sd = np.std(values, ddof=1)
                    se = sd * np.sqrt((1 / (k * m)) + n2_n1)
                    df = k * m - 1
                    tval = stats.t.ppf(1 - alpha/2, df=df)
                    return mean, sd, mean - tval * se, mean + tval * se

                # === SAVE SUMMARY METRICS ===
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
                    metrics_summary.append([name, *ci95_bf(values)])

                metrics_df = pd.DataFrame(
                    metrics_summary,
                    columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"]
                )
                metrics_df.to_excel(os.path.join(model_directory, f"{g}_metrics.xlsx"), index=False)

                # === SAVE ROC CURVE ===
                mean_tpr = np.mean(tprs, axis=0)
                lower_tpr = np.percentile(tprs, 2.5, axis=0)
                upper_tpr = np.percentile(tprs, 97.5, axis=0)

                roc_df = pd.DataFrame({
                    "FPR": interpolation,
                    "TPR_mean": mean_tpr,
                    "TPR_lower": lower_tpr,
                    "TPR_upper": upper_tpr
                })
                roc_df.to_excel(os.path.join(model_directory, f"{g}_roc_curve.xlsx"), index=False)

                # === SAVE PR CURVE ===
                mean_prec = np.mean(precisionspr, axis=0)
                lower_prec = np.percentile(precisionspr, 2.5, axis=0)
                upper_prec = np.percentile(precisionspr, 97.5, axis=0)

                pr_df = pd.DataFrame({
                    "Recall": interpolation,
                    "Precision_mean": mean_prec,
                    "Precision_lower": lower_prec,
                    "Precision_upper": upper_prec
                })
                pr_df.to_excel(os.path.join(model_directory, f"{g}_pr_curve.xlsx"), index=False)

    print(f"✅ Finished {n_features} features. Results saved in: {results_directory}")
