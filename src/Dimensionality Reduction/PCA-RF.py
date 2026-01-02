import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)
from pathlib import Path
from scipy import stats  # for t/z critical values


# === GENERAL CONFIGURATION ===
n_jobs = -1
directory = r'/homes/3/om871/Documents/ML'

# === PCA components based on EDA (95% explained variance) ===
pca_components = {
    'Coherence': 36,
    'Imag_Coherence': 41,
    'WPLI': 41,
    'WPPC': 34
}

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

# Feature filters for each connectivity metric
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# === MAIN LOOP OVER FEATURE TYPES ===
for g, n_components in pca_components.items():
    print(f"\n=== Running PCA for {g} with {n_components} components (95% variance) ===")

    results_directory = os.path.join(directory, f'Results/PCA_95Var')
    feature_directory = os.path.join(results_directory, g)
    Path(feature_directory).mkdir(parents=True, exist_ok=True)

    # Initialize storage structures
    tprs, precisionspr, rocs, APscores = [], [], [], []
    HC_precision, HC_recall, HC_f1 = [], [], []
    MDD_precision, MDD_recall, MDD_f1 = [], [], []
    accuracy, brier_scores = [], []

    # Filter features
    X = filters[g](X_all)

    # === CROSS-VALIDATION LOOP ===
    for train_ix, test_ix in outer_cv.split(X, y):
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        # Pipeline: Standardize → PCA → Random Forest
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('classifier', RandomForestClassifier(
                n_estimators=500,
                random_state=42,
                n_jobs=-1
            ))
        ])

        pipe.fit(X_train, y_train)

        # Predictions
        probas_ = pipe.predict_proba(X_test)
        y_pred = pipe.predict(X_test)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
        tprs.append(np.interp(interpolation, fpr, tpr))
        tprs[-1][0] = 0.0
        rocs.append(auc(fpr, tpr))

        # Precision-Recall
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

    # --- 95% CI FUNCTION ---
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

    # --- METRICS SUMMARY ---
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

    metrics_summary = []
    for name, values in metric_names:
        mean, sd, ci_lower, ci_upper = ci95_bf(values)
        metrics_summary.append([name, mean, sd, ci_lower, ci_upper])

    metrics_df = pd.DataFrame(metrics_summary, columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"])
    metrics_df.to_excel(os.path.join(feature_directory, f"{g}_metrics.xlsx"), index=False)

    # === SAVE ROC AND PR CURVE DATA ===
    mean_tpr = np.mean(tprs, axis=0)
    lower_bound = np.percentile(tprs, 2.5, axis=0)
    upper_bound = np.percentile(tprs, 97.5, axis=0)
    roc_df = pd.DataFrame({
        "FPR": interpolation,
        "TPR_mean": mean_tpr,
        "TPR_lower": lower_bound,
        "TPR_upper": upper_bound
    })
    roc_df.to_excel(os.path.join(feature_directory, f"{g}_roc_curve.xlsx"), index=False)

    mean_precision = np.mean(precisionspr, axis=0)
    lower_pr = np.percentile(precisionspr, 2.5, axis=0)
    upper_pr = np.percentile(precisionspr, 97.5, axis=0)
    pr_df = pd.DataFrame({
        "Recall": interpolation,
        "Precision_mean": mean_precision,
        "Precision_lower": lower_pr,
        "Precision_upper": upper_pr
    })
    pr_df.to_excel(os.path.join(feature_directory, f"{g}_pr_curve.xlsx"), index=False)

    print(f"✅ Finished {g} ({n_components} PCs, 95% var). Results saved in: {feature_directory}")
