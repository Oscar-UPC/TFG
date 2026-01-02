import os
import numpy as np
import pandas as pd
import pymrmr
from sklearn.model_selection import RepeatedStratifiedKFold
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
features = ['Imag_Coherence']

# === HYPERPARAMETER GRID (RF) ===
max_depth_list = [None, 10, 20, 30]        
min_samples_leaf_list = [1, 2, 5, 10]     

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
n_features_to_select_list = [30]

# Feature filters for each connectivity metric
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# === Función de intervalos de confianza ===
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


# === MAIN LOOP ===
for n_features in n_features_to_select_list:
    print(f"\n=== Running mRMR with {n_features} features ===")

    base_results_directory = os.path.join(directory, f'Results/mRMR/mRMR_{n_features}_features')
    Path(base_results_directory).mkdir(parents=True, exist_ok=True)

    # Loop over all RF hyperparameter combinations
    for max_depth in max_depth_list:
        for min_leaf in min_samples_leaf_list:

            depth_tag = "None" if max_depth is None else str(max_depth)
            combo_tag = f"RF_depth-{depth_tag}_leaf-{min_leaf}"

            results_directory = os.path.join(base_results_directory, combo_tag)
            Path(results_directory).mkdir(parents=True, exist_ok=True)

            print(f"\n--- RF config: max_depth={max_depth}, min_samples_leaf={min_leaf} ---")
            print(f"Saving to: {results_directory}")

            # Inicializar contenedores por métrica
            tprs, precisionspr, rocs, APscores = {}, {}, {}, {}
            HC_precision, HC_recall, HC_f1 = {}, {}, {}
            MDD_precision, MDD_recall, MDD_f1 = {}, {}, {}
            accuracy, brier_scores = {}, {}

            for g in features:
                print(f" → Processing feature type: {g}")

                if g in filters:
                    X = filters[g](X_all)
                else:
                    continue

                feature_directory = os.path.join(results_directory, g)
                Path(feature_directory).mkdir(parents=True, exist_ok=True)

                tprs[g], precisionspr[g], rocs[g], APscores[g] = [], [], [], []
                HC_precision[g], HC_recall[g], HC_f1[g] = [], [], []
                MDD_precision[g], MDD_recall[g], MDD_f1[g] = [], [], []
                accuracy[g], brier_scores[g] = [], []

                # === CROSS-VALIDATION LOOP ===
                for train_ix, test_ix in outer_cv.split(X, y):
                    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
                    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

                    # Preparar dataframe para pymrmr
                    df_train = X_train.copy()
                    df_train.insert(0, 'target', y_train.values)

                    # Selección mRMR (modo MID)
                    selected_features = pymrmr.mRMR(df_train, 'MID', n_features)
                    X_train_sel = X_train[selected_features]
                    X_test_sel = X_test[selected_features]

                    # === Entrenar modelo (con hiperparámetros) ===
                    clf = RandomForestClassifier(
                        n_estimators=500,
                        max_depth=max_depth,
                        min_samples_leaf=min_leaf,
                        random_state=42,
                        n_jobs=n_jobs
                    )
                    clf.fit(X_train_sel, y_train)

                    # === Predicciones ===
                    probas_ = clf.predict_proba(X_test_sel)
                    y_pred = clf.predict(X_test_sel)

                    # === ROC ===
                    fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
                    tprs[g].append(np.interp(interpolation, fpr, tpr))
                    tprs[g][-1][0] = 0.0
                    rocs[g].append(auc(fpr, tpr))

                    # === PR ===
                    precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
                    APscores[g].append(average_precision_score(y_test, probas_[:, 1]))
                    precisionspr[g].append(np.interp(interpolation, recall[::-1], precision[::-1]))
                    precisionspr[g][-1][0] = 1.0

                    # === Métricas de clasificación ===
                    report = classification_report(
                        y_test, y_pred,
                        target_names=['HC', 'MDD'],
                        output_dict=True
                    )
                    HC_precision[g].append(report['HC']['precision'])
                    HC_recall[g].append(report['HC']['recall'])
                    HC_f1[g].append(report['HC']['f1-score'])
                    MDD_precision[g].append(report['MDD']['precision'])
                    MDD_recall[g].append(report['MDD']['recall'])
                    MDD_f1[g].append(report['MDD']['f1-score'])
                    accuracy[g].append(report['accuracy'])
                    brier_scores[g].append(brier_score_loss(y_test, probas_[:, 1]))

                # === Guardar métricas ===
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

                metrics_df = pd.DataFrame(
                    metrics_summary,
                    columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"]
                )
                metrics_df.to_excel(os.path.join(feature_directory, f"{g}_metrics.xlsx"), index=False)

                # === ROC y PR ===
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

            print(f"✅ Finished RF config {combo_tag}. Results saved in: {results_directory}")

    print(f"\n✅ Finished {n_features} features (all RF combinations). Base folder: {base_results_directory}")
