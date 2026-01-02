import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
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


# CONFIG
directory = r'/homes/3/om871/Documents/ML'
features = ['WPPC']

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
gamma_values = [0.01, 0.1, 1]


for n_features in n_features_to_select_list:

    print(f"\n=== Running RFE + SVM RBF with {n_features} features ===")

    results_directory = os.path.join(
        directory,
        f'Results/RFE_SVM/RFE_{n_features}_features'
    )
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    for g in features:

        print(f" → Processing {g}")

        if g in filters:
            X = filters[g](X_all)

        for C_val in C_values:
            for gamma_val in gamma_values:

                print(f"     → C={C_val}, gamma={gamma_val}")

                model_dir = os.path.join(
                    results_directory, g, f"SVM_C_{C_val}_gamma_{gamma_val}"
                )
                Path(model_dir).mkdir(parents=True, exist_ok=True)

                # storage
                tprs, precisionspr, rocs, APscores = [], [], [], []
                HC_precision, HC_recall, HC_f1 = [], [], []
                MDD_precision, MDD_recall, MDD_f1 = [], [], []
                accuracy, brier_scores = [], []

                for train_ix, test_ix in outer_cv.split(X, y):
                    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

                    # === RFE with RANDOM FOREST (correct) ===
                    rfe = RFE(
                        estimator=RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            n_jobs=-1
                        ),
                        n_features_to_select=n_features,
                        step=0.1
                    )
                    rfe.fit(X_train, y_train)

                    X_train_sel = rfe.transform(X_train)
                    X_test_sel = rfe.transform(X_test)

                    # === Final model: SVM RBF + SCALER ===
                    clf = Pipeline([
                        ("scaler", StandardScaler()),
                        ("svm", SVC(
                            kernel='rbf',
                            C=C_val,
                            gamma=gamma_val,
                            probability=True
                        ))
                    ])

                    clf.fit(X_train_sel, y_train)
                    probas_ = clf.predict_proba(X_test_sel)
                    y_pred = clf.predict(X_test_sel)

                    # === metrics ===
                    fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
                    itp = np.interp(interpolation, fpr, tpr)
                    itp[0] = 0.0
                    tprs.append(itp)
                    rocs.append(auc(fpr, tpr))

                    precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
                    APscores.append(average_precision_score(y_test, probas_[:, 1]))
                    ip = np.interp(interpolation, recall[::-1], precision[::-1])
                    ip[0] = 1.0
                    precisionspr.append(ip)

                    rep = classification_report(
                        y_test, y_pred, target_names=['HC', 'MDD'], output_dict=True
                    )
                    HC_precision.append(rep['HC']['precision'])
                    HC_recall.append(rep['HC']['recall'])
                    HC_f1.append(rep['HC']['f1-score'])
                    MDD_precision.append(rep['MDD']['precision'])
                    MDD_recall.append(rep['MDD']['recall'])
                    MDD_f1.append(rep['MDD']['f1-score'])
                    accuracy.append(rep['accuracy'])
                    brier_scores.append(brier_score_loss(y_test, probas_[:, 1]))

                # CI
                def ci95(values, k=5, m=20, n2_n1=0.25):
                    vals = np.array(values)
                    mean = np.mean(vals)
                    sd = np.std(vals, ddof=1)
                    se = sd*np.sqrt((1/(k*m))+n2_n1)
                    df = k*m - 1
                    tval = stats.t.ppf(0.975, df=df)
                    return mean, sd, mean - tval*se, mean + tval*se

                summary = []
                items = [
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

                for name, vals in items:
                    summary.append([name, *ci95(vals)])

                pd.DataFrame(
                    summary,
                    columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"]
                ).to_excel(os.path.join(model_dir, f"{g}_metrics.xlsx"), index=False)

                # ROC curve export
                mtpr = np.mean(tprs, axis=0)
                lo = np.percentile(tprs, 2.5, axis=0)
                up = np.percentile(tprs, 97.5, axis=0)

                pd.DataFrame({
                    "FPR": interpolation,
                    "TPR_mean": mtpr,
                    "TPR_lower": lo,
                    "TPR_upper": up
                }).to_excel(os.path.join(model_dir, f"{g}_roc_curve.xlsx"), index=False)

                # PR curve export
                mpr = np.mean(precisionspr, axis=0)
                lo = np.percentile(precisionspr, 2.5, axis=0)
                up = np.percentile(precisionspr, 97.5, axis=0)

                pd.DataFrame({
                    "Recall": interpolation,
                    "Precision_mean": mpr,
                    "Precision_lower": lo,
                    "Precision_upper": up
                }).to_excel(os.path.join(model_dir, f"{g}_pr_curve.xlsx"), index=False)

    print(f"✅ Finished {n_features} features. Results saved in: {results_directory}")
