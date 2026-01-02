import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ================= CONFIG =================
directory = r'/homes/3/om871/Documents/ML'
results_dir = os.path.join(directory, 'Results/Explainability/Coherence')
Path(results_dir).mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FEATURES_RFE = 30
TOP_K = 10

# ================= LOAD DATA =================
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
data = pd.read_csv(file_path)

y = data.pop('Population')  # HC=0, MDD=1
X_all = data.copy()

# ================= FILTER COHERENCE =================
X = X_all.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$')

# ================= PIPELINE =================

# --- 1) SelectKBest prefilter ---
skb = SelectKBest(
    score_func=f_classif,
    k=min(200, X.shape[1])
)
X_pref = skb.fit_transform(X, y)
pref_features = X.columns[skb.get_support()]

# --- 2) RFE ---
rf_rfe = RandomForestClassifier(
    n_estimators=200,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rfe = RFE(
    estimator=rf_rfe,
    n_features_to_select=N_FEATURES_RFE,
    step=0.1
)
X_sel = rfe.fit_transform(X_pref, y)
final_features = pref_features[rfe.get_support()]

# --- 3) Final RF ---
rf_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_final.fit(X_sel, y)

# ================= FEATURE IMPORTANCE =================
importances = rf_final.feature_importances_

features_df = pd.DataFrame({
    'Feature': final_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top10 = features_df.head(TOP_K)

# --- Save CSV ---
csv_path = os.path.join(results_dir, 'Top_10_features_Coherence.csv')
top10.to_csv(csv_path, index=False)

# ================= BAR PLOT =================
plt.figure(figsize=(8, 5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=top10,
    color='steelblue'
)
plt.title('Top 10 Feature Importances (Coherence)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'Top_10_Feature_Importance_Coherence.png'))
plt.close()

# ================= BOXPLOTS (TOP 2) =================
top2_features = top10['Feature'].iloc[:2].tolist()

for feat in top2_features:
    df_box = pd.DataFrame({
        'Value': X_all[feat],
        'Population': y.map({0: 'HC', 1: 'MDD'})
    })

    plt.figure(figsize=(6, 5))
    sns.boxplot(
        x='Population',
        y='Value',
        data=df_box,
        palette={'HC': 'lightblue', 'MDD': 'lightgreen'}
    )
    plt.title(f'Distribution of {feat}', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'Boxplot_{feat}.png'))
    plt.close()

# ================= SAVE MODEL =================
with open(os.path.join(results_dir, 'RF_final_Coherence.pkl'), 'wb') as f:
    pickle.dump(rf_final, f)

print("✅ Explainability analysis for COHERENCE completed.")
