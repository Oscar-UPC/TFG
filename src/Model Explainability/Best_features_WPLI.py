import os
import numpy as np
import pandas as pd
import pymrmr
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ================= CONFIG =================
directory = r'/homes/3/om871/Documents/ML'
results_dir = os.path.join(directory, 'Results/Explainability/WPLI')
Path(results_dir).mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FEATURES_MRMR = 30
TOP_K = 10

# ================= LOAD DATA =================
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
data = pd.read_csv(file_path)

y = data.pop('Population')  # HC=0, MDD=1
X_all = data.copy()

# ================= FILTER WPLI =================
X = X_all.filter(regex='WPLI')

# ================= mRMR SELECTION =================
df_mrmr = X.copy()
df_mrmr.insert(0, 'target', y.values)

selected_features = pymrmr.mRMR(
    df_mrmr,
    'MID',
    N_FEATURES_MRMR
)

X_sel = X[selected_features]

# ================= FINAL RANDOM FOREST =================
rf_final = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_final.fit(X_sel, y)

# ================= FEATURE IMPORTANCE =================
importances = rf_final.feature_importances_

features_df = pd.DataFrame({
    'Feature': selected_features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

top10 = features_df.head(TOP_K)

# --- Save CSV ---
top10.to_csv(
    os.path.join(results_dir, 'Top_10_features_WPLI.csv'),
    index=False
)

# ================= BAR PLOT =================
plt.figure(figsize=(8, 5))
sns.barplot(
    x='Importance',
    y='Feature',
    data=top10,
    color='steelblue'
)
plt.title('Top 10 Feature Importances (WPLI)', fontsize=14)
plt.tight_layout()
plt.savefig(
    os.path.join(results_dir, 'Top_10_Feature_Importance_WPLI.png')
)
plt.close()

# ================= BOXPLOTS (TOP 2 FEATURES) =================
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
    plt.savefig(
        os.path.join(results_dir, f'Boxplot_{feat}.png')
    )
    plt.close()

# ================= SAVE MODEL =================
with open(
    os.path.join(results_dir, 'RF_final_WPLI.pkl'),
    'wb'
) as f:
    pickle.dump(rf_final, f)

print("✅ Explainability analysis for WPLI completed.")
