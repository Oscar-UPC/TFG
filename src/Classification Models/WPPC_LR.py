import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
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
from sklearn.ensemble import RandomForestClassifier


# === GENERAL CONFIGURATION ===
n_jobs = -1
directory = r'/homes/3/om871/Documents/ML'

features = ['WPPC']

# === LOAD DATA ===
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
clf_data = pd.read_csv(file_path)

y = clf_data.pop('Population')
X_all = clf_data

interpolation = np.linspace(0, 1, 100)

outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=20,
    random_state=42
)

n_features_to_select_list = [30]

filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

C_values = [0.1, 1.0, 10]


for n_features in n_features_to_select_list:

    print(f"\n=== Running RFE + LR with {n_features} features ===")

    results_directory = os.path.join(
        directory,
        f'Results/RFE_LR/RFE_{n_features}_features'
    )
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for g in features:
        print(f" → Processing feature type: {g}")

        if g in filters:
            X = filters[g](X_all)

        for C_val in C_values:
            print(f"     → Logistic Regression with C = {C_val}")

            model_directory = os.path.join(
                results_directory, g, f"LR_C_{C_val}"
            )
            Path(model_directory).mkdir(parents=True, exist_ok=True)

            # metric storage
            tprs, precisionspr, rocs, APscores = [], [], [], []
            HC_precision, HC_recall, HC_f1 = [], [], []
            MDD_precision, MDD_recall, MDD_f1 = [], [], []
            accuracy, brier_scores = [], []

            # CV loop
            for train_ix, test_ix in outer_cv.split(X, y):

                X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

                # === RFE WITH RANDOM FOREST ===
                rfe_estimator = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=-1
                )

                rfe = RFE(
                    estimator=rfe_estimator,
                    n_features_to_select=n_features,
                    step=0.1
                )
                rfe.fit(X_train, y_train)

                X_train_sel = rfe.transform(X_train)
                X_test_sel = rfe.transform(X_test)

                # === FINAL MODEL: LR + SCALER ===
                clf = Pipeline([
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(
                        C=C_val,
                        penalty='l2',
                        solver='liblinear',
                        max_iter=500
                    ))
                ])

                clf.fit(X_train_sel, y_train)

                probas_ = clf.predict_proba(X_test_sel)
                y_pred = clf.predict(X_test_sel)

                # ROC
                fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
                interp_tpr = np.interp(interpolation, fpr, tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                rocs.append(auc(fpr, tpr))

                # PR
                precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
                APscores.append(average_precision_score(y_test, probas_[:, 1]))
                interp_prec = np.interp(interpolation, recall[::-1], precision[::-1])
                interp_prec[0] = 1.0
                precisionspr.append(interp_prec)

                # classification stats
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

            def ci95(values, k=5, m=20, n2_n1=0.25):
                vals = np.array(values)
                mean = np.mean(vals)
                sd = np.std(vals, ddof=1)
                se = sd * np.sqrt((1/(k*m)) + n2_n1)
                df = k*m - 1
                tval = stats.t.ppf(0.975, df=df)
                return mean, sd, mean - tval*se, mean + tval*se

            metric_list = [
                ("ROC AUC", rocs),
                ("PR AP", APscores),
                ("Accuracy", accuracy),
                ("HC Precision", HC_precision),
                ("HC Recall", HC_recall),
                ("HC F1", HC_f1),
                ("MDD Precision", MDD_precision),
                ("MDD Recall", MDD_recall),
                ("MDD F1", MDD_f1),
                ("Brier Score", brier_scores),
            ]

            summary = []
            for name, vals in metric_list:
                summary.append([name, *ci95(vals)])

            pd.DataFrame(
                summary,
                columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"]
            ).to_excel(os.path.join(model_directory, f"{g}_metrics.xlsx"), index=False)

            # save ROC curve
            mean_tpr = np.mean(tprs, axis=0)
            lower = np.percentile(tprs, 2.5, axis=0)
            upper = np.percentile(tprs, 97.5, axis=0)

            pd.DataFrame({
                "FPR": interpolation,
                "TPR_mean": mean_tpr,
                "TPR_lower": lower,
                "TPR_upper": upper
            }).to_excel(os.path.join(model_directory, f"{g}_roc_curve.xlsx"), index=False)

            # save PR curve
            mean_prec = np.mean(precisionspr, axis=0)
            lower_pr = np.percentile(precisionspr, 2.5, axis=0)
            upper_pr = np.percentile(precisionspr, 97.5, axis=0)

            pd.DataFrame({
                "Recall": interpolation,
                "Precision_mean": mean_prec,
                "Precision_lower": lower_pr,
                "Precision_upper": upper_pr
            }).to_excel(os.path.join(model_directory, f"{g}_pr_curve.xlsx"), index=False)

    print(f"✅ Finished {n_features} features. Results saved in: {results_directory}")
