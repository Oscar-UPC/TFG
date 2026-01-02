import os
import numpy as np
import pandas as pd
import pymrmr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    classification_report,
    brier_score_loss
)
from pathlib import Path
from scipy import stats

# ============================================================
#               CONFIGURACIÓN GENERAL
# ============================================================
n_jobs = -1
directory = r'/homes/3/om871/Documents/ML'

features = ['Coherence']

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

# Pre-filtrado y selección final
mrmr_pre_filters = [100, 75, 50]
final_feature_targets = [10, 20, 30]

filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# ============================================================
#        FUNCIÓN INTERVALO DE CONFIANZA BOUCKAERT & FRANK
# ============================================================
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

# ============================================================
#                LOOP PRINCIPAL DEL PIPELINE
# ============================================================
for metric in features:
    # Filtrar columnas de esta métrica
    X_metric = filters[metric](X_all)

    print(f"\n\n==========================================")
    print(f" MÉTRICA: {metric}")
    print(f"==========================================")

    # Cross-validation
    for mrmr_k in mrmr_pre_filters:
        for final_k in final_feature_targets:

            model_tag = f"mRMR{mrmr_k}_RFE{final_k}"

            print(f"\n→ Ejecutando {model_tag} para {metric}")

            # Carpeta destino
            results_dir = os.path.join(
                directory,
                "Results",
                "Hybrid",
                model_tag,
                metric
            )
            Path(results_dir).mkdir(parents=True, exist_ok=True)

            # Diccionarios para almacenar resultados de CV
            tprs, precisionspr, roc_scores, ap_scores = [], [], [], []
            HC_precision, HC_recall, HC_f1 = [], [], []
            MDD_precision, MDD_recall, MDD_f1 = [], [], []
            accuracy, brier_scores = [], []

            # CROSS-VALIDATION
            for train_ix, test_ix in outer_cv.split(X_metric, y):
                X_train, X_test = X_metric.iloc[train_ix], X_metric.iloc[test_ix]
                y_train, y_test = y.iloc[train_ix], y.iloc[test_ix]

                # ==== mRMR pre-filtrado ====
                df_train = X_train.copy()
                df_train.insert(0, "target", y_train.values)

                selected_mrmr = pymrmr.mRMR(df_train, 'MID', mrmr_k)

                X_train_mrmr = X_train[selected_mrmr]
                X_test_mrmr = X_test[selected_mrmr]

                # ==== RFE ====
                base_rf = RandomForestClassifier(
                    n_estimators=200,
                    n_jobs=-1,
                    random_state=42
                )

                selector = RFE(
                    estimator=base_rf,
                    n_features_to_select=final_k,
                    step=0.1
                )
                selector.fit(X_train_mrmr, y_train)

                final_feats = X_train_mrmr.columns[selector.support_]

                X_train_sel = X_train_mrmr[final_feats]
                X_test_sel = X_test_mrmr[final_feats]

                # ==== Modelo final ====
                clf = RandomForestClassifier(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1
                )
                clf.fit(X_train_sel, y_train)

                probas = clf.predict_proba(X_test_sel)[:, 1]
                y_pred = clf.predict(X_test_sel)

                # ===== ROC =====
                fpr, tpr, _ = roc_curve(y_test, probas)
                tprs.append(np.interp(interpolation, fpr, tpr))
                tprs[-1][0] = 0.0
                roc_scores.append(auc(fpr, tpr))

                # ===== PR =====
                precision, recall, _ = precision_recall_curve(y_test, probas)
                ap_scores.append(average_precision_score(y_test, probas))
                precisionspr.append(np.interp(interpolation, recall[::-1], precision[::-1]))
                precisionspr[-1][0] = 1.0

                # ==== Classification report ====
                report = classification_report(y_test, y_pred, output_dict=True)

                HC_precision.append(report['0']['precision'])
                HC_recall.append(report['0']['recall'])
                HC_f1.append(report['0']['f1-score'])

                MDD_precision.append(report['1']['precision'])
                MDD_recall.append(report['1']['recall'])
                MDD_f1.append(report['1']['f1-score'])

                accuracy.append(report['accuracy'])
                brier_scores.append(brier_score_loss(y_test, probas))

            # ======================================================
            #               GUARDAR MÉTRICAS
            # ======================================================
            metric_list = [
                ("ROC AUC", roc_scores),
                ("PR AP", ap_scores),
                ("Accuracy", accuracy),
                ("HC Precision", HC_precision),
                ("HC Recall", HC_recall),
                ("HC F1", HC_f1),
                ("MDD Precision", MDD_precision),
                ("MDD Recall", MDD_recall),
                ("MDD F1", MDD_f1),
                ("Brier Score", brier_scores)
            ]

            summary = []
            for name, vals in metric_list:
                mean, sd, lo, hi = ci95_bf(vals)
                summary.append([name, mean, sd, lo, hi])

            pd.DataFrame(
                summary,
                columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"]
            ).to_excel(os.path.join(results_dir, f"{metric}_metrics.xlsx"), index=False)

            # ===== Guardar curvas =====
            pd.DataFrame({
                "FPR": interpolation,
                "TPR_mean": np.mean(tprs, axis=0),
                "TPR_lower": np.percentile(tprs, 2.5, axis=0),
                "TPR_upper": np.percentile(tprs, 97.5, axis=0)
            }).to_excel(os.path.join(results_dir, f"{metric}_roc_curve.xlsx"), index=False)

            pd.DataFrame({
                "Recall": interpolation,
                "Precision_mean": np.mean(precisionspr, axis=0),
                "Precision_lower": np.percentile(precisionspr, 2.5, axis=0),
                "Precision_upper": np.percentile(precisionspr, 97.5, axis=0)
            }).to_excel(os.path.join(results_dir, f"{metric}_pr_curve.xlsx"), index=False)

            print(f"   ✔ Guardado en: {results_dir}")

print("\n\n======================")
print(" FINALIZADO COMPLETAMENTE ")
print("======================")
