import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
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
from scipy.stats import kruskal


# === CUSTOM FUNCTION: Kruskal–Wallis SCORE ===
def kruskal_score(X, y):
    """
    Compute Kruskal–Wallis H-test for each feature.
    Returns arrays of H-statistics and p-values.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target labels (categorical).

    Returns
    -------
    H : array, shape (n_features,)
        Kruskal–Wallis H statistic for each feature.
    p : array, shape (n_features,)
        Two-sided p-value for each feature.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    unique_classes = np.unique(y)

    H_values = np.zeros(X.shape[1])
    p_values = np.zeros(X.shape[1])

    for i in range(X.shape[1]):
        groups = [X[y == cls, i] for cls in unique_classes]
        H, p = kruskal(*groups)
        H_values[i] = H
        p_values[i] = p

    return H_values, p_values


# === GENERAL CONFIGURATION ===
n_jobs = -1
directory = r'/homes/3/om871/Documents/ML'

features = ['WPPC', 'WPLI', 'Imag_Coherence', 'Coherence']

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

# Number of selected features to test
n_features_to_select_list = [10, 20, 30]

# Feature filters for each connectivity metric
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# === MAIN LOOP OVER NUMBER OF FEATURES ===
for n_features in n_features_to_select_list:
    print(f"\n=== Running SelectKBest (Kruskal–Wallis) with {n_features} features ===")

    # Create results directory for this configuration
    results_directory = os.path.join(directory, f'Results/SelectKBest_Kruskal/SelectKBest_{n_features}_features')
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    # Initialize storage structures
    tprs = {f: [] for f in features}
    precisionspr = {f: [] for f in features}
    rocs = {f: [] for f in features}
    APscores = {f: [] for f in features}
    HC_precision = {f: [] for f in features}
    HC_recall = {f: [] for f in features}
    HC_f1 = {f: [] for f in features}
    MDD_precision = {f: [] for f in features}
    MDD_recall = {f: [] for f in features}
    MDD_f1 = {f: [] for f in features}
    accuracy = {f: [] for f in features}
    brier_scores = {f: [] for f in features}

    # === LOOP OVER FEATURE TYPES ===
    for g in features:
        print(f" → Processing feature type: {g}")

        if g in filters:
            X = filters[g](X_all)

        feature_directory = os.path.join(results_directory, g)
        Path(feature_directory).mkdir(parents=True, exist_ok=True)

        # === CROSS-VALIDATION LOOP ===
        for train_ix, test_ix in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            # Define pipeline with SelectKBest (Kruskal–Wallis) + Random Forest
            pipe = Pipeline([
                ('selector', SelectKBest(score_func=kruskal_score, k=n_features)),
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
            tprs[g].append(np.interp(interpolation, fpr, tpr))
            tprs[g][-1][0] = 0.0
            rocs[g].append(auc(fpr, tpr))

            # Precision-Recall
            precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
            APscores[g].append(average_precision_score(y_test, probas_[:, 1]))
            precisionspr[g].append(np.interp(interpolation, recall[::-1], precision[::-1]))
            precisionspr[g][-1][0] = 1.0

            # Classification report
            report = classification_report(y_test, y_pred, target_names=['HC', 'MDD'], output_dict=True)
            HC_precision[g].append(report['HC']['precision'])
            HC_recall[g].append(report['HC']['recall'])
            HC_f1[g].append(report['HC']['f1-score'])
            MDD_precision[g].append(report['MDD']['precision'])
            MDD_recall[g].append(report['MDD']['recall'])
            MDD_f1[g].append(report['MDD']['f1-score'])
            accuracy[g].append(report['accuracy'])
            brier_scores[g].append(brier_score_loss(y_test, probas_[:, 1]))


        # === Confidence intervals (Bouckaert & Frank correction) ===
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

        # === Compute and save metrics summary ===
        metrics_summary = []
        metric_names = [
            ("ROC AUC", rocs[g]),
            ("PR AP", APscores[g]),
            ("Accuracy", accuracy[g]),
            ("HC Precision", HC_precision[g]),
            ("HC Recall", HC_recall[g]),
            ("HC F1", HC_f1[g]),
            ("MDD Precision", MDD_precision[g]),
            ("MDD Recall", MDD_recall[g]),
            ("MDD F1", MDD_f1[g]),
            ("Brier Score", brier_scores[g])
        ]

        for name, values in metric_names:
            mean, sd, ci_lower, ci_upper = ci95_bf(values)
            metrics_summary.append([name, mean, sd, ci_lower, ci_upper])

        metrics_df = pd.DataFrame(metrics_summary, columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"])
        metrics_df.to_excel(os.path.join(feature_directory, f"{g}_metrics.xlsx"), index=False)

        # === Save ROC and PR curve data ===
        mean_tpr = np.mean(tprs[g], axis=0)
        lower_bound = np.percentile(tprs[g], 2.5, axis=0)
        upper_bound = np.percentile(tprs[g], 97.5, axis=0)

        roc_df = pd.DataFrame({
            "FPR": interpolation,
            "TPR_mean": mean_tpr,
            "TPR_lower": lower_bound,
            "TPR_upper": upper_bound
        })
        roc_df.to_excel(os.path.join(feature_directory, f"{g}_roc_curve.xlsx"), index=False)

        mean_precision = np.mean(precisionspr[g], axis=0)
        lower_pr = np.percentile(precisionspr[g], 2.5, axis=0)
        upper_pr = np.percentile(precisionspr[g], 97.5, axis=0)

        pr_df = pd.DataFrame({
            "Recall": interpolation,
            "Precision_mean": mean_precision,
            "Precision_lower": lower_pr,
            "Precision_upper": upper_pr
        })
        pr_df.to_excel(os.path.join(feature_directory, f"{g}_pr_curve.xlsx"), index=False)

    print(f"✅ Finished {n_features} features. Results saved in: {results_directory}")
