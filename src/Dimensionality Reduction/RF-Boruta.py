import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)
from boruta import BorutaPy
from pathlib import Path
from scipy import stats


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

# Feature filters for each connectivity metric
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# === MAIN LOOP over feature groups ===
for g in features:
    print(f"\n=== Running ANOVA F-test + Boruta for feature type: {g} ===")

    results_directory = os.path.join(directory, f'Results/Boruta_prefilter/{g}')
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    # Initialize storage
    tprs, precisionspr, rocs, APscores = [], [], [], []
    HC_precision, HC_recall, HC_f1 = [], [], []
    MDD_precision, MDD_recall, MDD_f1 = [], [], []
    accuracy, brier_scores = [], []

    # Filter by group
    if g in filters:
        X = filters[g](X_all)

    # === CV Loop ===
    for fold_idx, (train_ix, test_ix) in enumerate(outer_cv.split(X, y), start=1):
        print(f" → Fold {fold_idx} ...")
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

        # === 1) Pre-filter using ANOVA F-test ===
        skb = SelectKBest(score_func=f_classif, k=min(200, X_train.shape[1]))
        X_train_pref = skb.fit_transform(X_train, y_train)
        X_test_pref = skb.transform(X_test)
        prefiltered_features = X.columns[skb.get_support()]

        # === 2) Boruta feature selection ===
        rf_boruta = RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            n_jobs=n_jobs,
            class_weight='balanced',
            random_state=42
        )

        boruta_selector = BorutaPy(
            estimator=rf_boruta,
            n_estimators='auto',
            perc=100,
            two_step=True,
            max_iter=100,
            random_state=42
        )

        boruta_selector.fit(X_train_pref, y_train.values)

        selected_features = prefiltered_features[boruta_selector.support_]
        tentative_features = prefiltered_features[boruta_selector.support_weak_]

        # === Save selected features for stability analysis ===
        feature_log = pd.DataFrame({
            "fold": [fold_idx] * len(selected_features),
            "feature": selected_features,
            "status": "accepted"
        })

        if len(tentative_features) > 0:
            tentative_log = pd.DataFrame({
                "fold": [fold_idx] * len(tentative_features),
                "feature": tentative_features,
                "status": "tentative"
            })
            feature_log = pd.concat([feature_log, tentative_log], axis=0, ignore_index=True)

        feature_log.to_csv(
            os.path.join(results_directory, f"{g}_selected_features.csv"),
            index=False,
            mode='a',
            header=not os.path.exists(os.path.join(results_directory, f"{g}_selected_features.csv"))
        )

        # === 3) Transform train/test ===
        X_train_sel = pd.DataFrame(X_train_pref, columns=prefiltered_features)[selected_features]
        X_test_sel = pd.DataFrame(X_test_pref, columns=prefiltered_features)[selected_features]

        # === 4) Train final Random Forest ===
        rf_final = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
        rf_final.fit(X_train_sel, y_train)

        # === 5) Predictions ===
        probas_ = rf_final.predict_proba(X_test_sel)
        y_pred = rf_final.predict(X_test_sel)

        # === 6) ROC ===
        fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
        tprs.append(np.interp(interpolation, fpr, tpr))
        tprs[-1][0] = 0.0
        rocs.append(auc(fpr, tpr))

        # === 7) Precision-Recall ===
        precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
        APscores.append(average_precision_score(y_test, probas_[:, 1]))
        precisionspr.append(np.interp(interpolation, recall[::-1], precision[::-1]))
        precisionspr[-1][0] = 1.0

        # === 8) Classification report ===
        report = classification_report(y_test, y_pred, target_names=['HC', 'MDD'], output_dict=True)
        HC_precision.append(report['HC']['precision'])
        HC_recall.append(report['HC']['recall'])
        HC_f1.append(report['HC']['f1-score'])
        MDD_precision.append(report['MDD']['precision'])
        MDD_recall.append(report['MDD']['recall'])
        MDD_f1.append(report['MDD']['f1-score'])
        accuracy.append(report['accuracy'])
        brier_scores.append(brier_score_loss(y_test, probas_[:, 1]))

    # === Confidence interval function ===
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

    # === Summary metrics ===
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
    metrics_df.to_excel(os.path.join(results_directory, f"{g}_metrics.xlsx"), index=False)

    # === ROC / PR curve data ===
    mean_tpr = np.mean(tprs, axis=0)
    lower_bound = np.percentile(tprs, 2.5, axis=0)
    upper_bound = np.percentile(tprs, 97.5, axis=0)

    roc_df = pd.DataFrame({
        "FPR": interpolation,
        "TPR_mean": mean_tpr,
        "TPR_lower": lower_bound,
        "TPR_upper": upper_bound
    })
    roc_df.to_excel(os.path.join(results_directory, f"{g}_roc_curve.xlsx"), index=False)

    mean_precision = np.mean(precisionspr, axis=0)
    lower_pr = np.percentile(precisionspr, 2.5, axis=0)
    upper_pr = np.percentile(precisionspr, 97.5, axis=0)

    pr_df = pd.DataFrame({
        "Recall": interpolation,
        "Precision_mean": mean_precision,
        "Precision_lower": lower_pr,
        "Precision_upper": upper_pr
    })
    pr_df.to_excel(os.path.join(results_directory, f"{g}_pr_curve.xlsx"), index=False)

    print(f"✅ Finished ANOVA + Boruta for {g}. Results saved in: {results_directory}")
