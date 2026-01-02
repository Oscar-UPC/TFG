# =====================================================
# === HIERARCHICAL CLUSTERING - ALL METRICS (EXCEL OUTPUT) ===
# =====================================================

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

# === CONFIGURATION ===
directory = r'C:\Users\Oscar\Desktop\Trabajo\ENGINE'

# Define distance thresholds per metric
distance_dict = {
    'WPPC': [20],
    'WPLI': [30],
    'Coherence': [20],
    'Imag_Coherence': [30]
}

# === LOAD DATA ===
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
clf_data = pd.read_csv(file_path)

# HC = 0, MDD = 1
y = clf_data.pop('Population')
X_all = clf_data

# === LOOP OVER METRICS ===
for feature_type, distances in distance_dict.items():

    print(f"\n============================")
    print(f"=== PROCESSING: {feature_type} ===")
    print(f"============================")

    # --- Filter metric columns ---
    if feature_type == 'Coherence':
        # exclude Imag_Coherence from Coherence
        X = X_all.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$')
    else:
        X = X_all.filter(regex=feature_type)

    if X.empty:
        print(f"[WARNING] No columns found for {feature_type}. Skipping.")
        continue

    print(f"Selected {X.shape[1]} {feature_type} features.")

    # --- Output directory ---
    results_directory = os.path.join(directory, f'Results/EDA/HierarchicalClustering/{feature_type}')
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    # --- Transpose & Standardize ---
    X_t = X.T
    scaler = StandardScaler()
    X_t_scaled = scaler.fit_transform(X_t)

    # --- Hierarchical clustering ---
    Z = linkage(X_t_scaled, method='ward', metric='euclidean')

    # --- Dendrogram ---
    plt.figure(figsize=(12, 6))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=6, color_threshold=0.7 * np.max(Z[:, 2]))
    plt.title(f"Hierarchical Clustering Dendrogram - {feature_type}")
    plt.xlabel("Metrics")
    plt.ylabel("Euclidean distance")
    plt.tight_layout()
    dendro_path = os.path.join(results_directory, f"Dendrogram_{feature_type}.png")
    plt.savefig(dendro_path, dpi=300)
    plt.close()
    print(f" → Dendrogram saved: {dendro_path}")

    # --- Cut tree and export clusters to Excel ---
    for d in distances:
        clusters = fcluster(Z, t=d, criterion='distance')
        n_clusters = len(np.unique(clusters))
        print(f"\nDistance threshold = {d} → {n_clusters} clusters")

        cluster_df = pd.DataFrame({'Metric': X_t.index, 'Cluster': clusters})

        excel_path = os.path.join(results_directory, f'{feature_type}_clusters_d{d}.xlsx')

        # Use 'openpyxl' if 'xlsxwriter' not installed
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for c in sorted(cluster_df['Cluster'].unique()):
                metrics = cluster_df.loc[cluster_df['Cluster'] == c, 'Metric']
                metrics.to_frame(name=f'Cluster_{c}').to_excel(writer, sheet_name=f'Cluster_{c}', index=False)

        print(f"   → Excel saved: {excel_path}")

print("\n✅ Hierarchical clustering completed for all metrics.")
