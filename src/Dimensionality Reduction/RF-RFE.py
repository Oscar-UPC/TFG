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
from scipy import stats  # for t/z critical values


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

# === MAIN LOOP over number of features ===
for n_features in n_features_to_select_list:

    print(f"\n=== Running RFE with {n_features} features ===")

    # Directory for this configuration
    results_directory = os.path.join(directory, f'Results/RFE/RFE_{n_features}_features')
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    # Initialize storage
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

    for g in features:
        print(f" → Processing feature type: {g}")

        if g in filters:
            X = filters[g](X_all)

        feature_directory = os.path.join(results_directory, g)
        Path(feature_directory).mkdir(parents=True, exist_ok=True)

        # === CV Loop ===
        for train_ix, test_ix in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
            y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

            # RFE
            rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
            rfe = RFE(estimator=rf, n_features_to_select=n_features, step=0.1)
            rfe.fit(X_train, y_train)

            # Transform train/test
            X_train_sel = rfe.transform(X_train)
            X_test_sel = rfe.transform(X_test)

            # Train final model
            rf_final = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
            rf_final.fit(X_train_sel, y_train)

            # Predictions
            probas_ = rf_final.predict_proba(X_test_sel)
            y_pred = rf_final.predict(X_test_sel)

            # ROC
            fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
            tprs[g].append(np.interp(interpolation, fpr, tpr))
            tprs[g][-1][0] = 0.0
            rocs[g].append(auc(fpr, tpr))

            # PR
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

        def ci95_bf(values, k=5, m=20, n2_n1=0.25, alpha=0.05):
            """
            Compute 95% confidence interval using the Bouckaert & Frank (2004)
            corrected repeated k-fold cross-validation t-test.
        
            Parameters
            ----------
            values : array-like
                List or array with one metric value per fold (total k*m values).
            k : int, optional
                Number of folds per repetition (default=5).
            m : int, optional
                Number of repetitions (default=20).
            n2_n1 : float, optional
                Ratio test/train sizes (default=0.25 for 80/20 split).
            alpha : float, optional
                Significance level (default=0.05 → 95% CI).
            """
        
            values = np.array(values)
            mean = np.mean(values)
            sd = np.std(values, ddof=1)
        
            # --- Corrected Standard Error (Bouckaert & Frank 2004) ---
            se = sd * np.sqrt((1 / (k * m)) + n2_n1)
        
            # --- Critical t value ---
            df = k * m - 1
            tval = stats.t.ppf(1 - alpha/2, df=df)
        
            # --- Confidence interval ---
            ci_lower = mean - tval * se
            ci_upper = mean + tval * se
        
            return mean, sd, ci_lower, ci_upper
        
        
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


        # Compute summary statistics
        for name, values in metric_names:
            mean, sd, ci_lower, ci_upper = ci95_bf(values)
            metrics_summary.append([name, mean, sd, ci_lower, ci_upper])

        # Save metrics summary
        metrics_df = pd.DataFrame(metrics_summary, columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"])
        metrics_df.to_excel(os.path.join(feature_directory, f"{g}_metrics.xlsx"), index=False)

        # === SAVE ROC AND PR CURVE DATA ===
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


