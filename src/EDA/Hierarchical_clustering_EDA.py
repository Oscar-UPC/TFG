# =====================================================
# === HIERARCHICAL CLUSTERING: EXPLORATORY ANALYSIS ===
# =====================================================

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler


# =====================================================
# === GENERAL CONFIGURATION ===
# =====================================================

# Number of parallel jobs (kept for consistency, not explicitly used here)
n_jobs = -1

# Root directory of the project
directory = r'C:\Users\Oscar\Desktop\Trabajo\ENGINE'

# Connectivity metrics to be analyzed
features = ['WPPC', 'WPLI', 'Imag_Coherence', 'Coherence']


# =====================================================
# === LOAD DATA ===
# =====================================================

# Load cleaned dataset
file_path = os.path.join(directory, 'Cleaned_data', 'Cleaned_data.csv')
clf_data = pd.read_csv(file_path)

# Binary labels: HC = 0, MDD = 1
y = clf_data.pop('Population')

# Feature matrix (all connectivity features)
X_all = clf_data


# =====================================================
# === FEATURE FILTERS ===
# =====================================================

# Dictionary mapping each connectivity metric to its corresponding column filter
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence')
                              .filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}


# =====================================================
# === OUTPUT DIRECTORY ===
# =====================================================

# Root directory for hierarchical clustering results
results_directory = os.path.join(
    directory, 'Results', 'EDA', 'HierarchicalClustering'
)
Path(results_directory).mkdir(parents=True, exist_ok=True)


# =====================================================
# === MAIN LOOP OVER CONNECTIVITY METRICS ===
# =====================================================

for g in features:
    print(f"\n=== Processing connectivity metric: {g} ===")

    # -------------------------------------------------
    # 1) Select feature subset corresponding to the
    #    current connectivity metric
    # -------------------------------------------------
    if g not in filters:
        print(f"[WARNING] No filter defined for {g}. Skipping.")
        continue

    X = filters[g](X_all)

    if X.empty:
        print(f"[WARNING] Filter for {g} returned zero features. Skipping.")
        continue

    # -------------------------------------------------
    # 2) Create output directory for this metric
    # -------------------------------------------------
    feature_directory = os.path.join(results_directory, g)
    Path(feature_directory).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------
    # 3) Transpose data so that each connectivity
    #    feature is treated as an observation
    #
    #    Shape after transpose:
    #    (n_features, n_subjects)
    # -------------------------------------------------
    X_t = X.T

    # -------------------------------------------------
    # 4) Standardize features across subjects
    # -------------------------------------------------
    scaler = StandardScaler()
    X_t_scaled = scaler.fit_transform(X_t)

    # -------------------------------------------------
    # 5) Perform hierarchical clustering
    #    - Ward linkage
    #    - Euclidean distance
    # -------------------------------------------------
    Z = linkage(X_t_scaled, method='ward', metric='euclidean')

    # -------------------------------------------------
    # 6) Plot and save dendrogram
    # -------------------------------------------------
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        leaf_rotation=90,
        leaf_font_size=6,
        labels=X_t.index.tolist(),
        color_threshold=0.7 * np.max(Z[:, 2])
    )

    plt.title(f"Hierarchical Clustering Dendrogram - {g}")
    plt.xlabel("Connectivity Features")
    plt.ylabel("Euclidean Distance")
    plt.tight_layout()

    fig_path = os.path.join(feature_directory, f"Dendrogram_{g}.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    print(f" → Dendrogram saved at: {fig_path}")

    # -------------------------------------------------
    # 7) (Optional) Export cluster assignments for
    #    different numbers of clusters
    # -------------------------------------------------
    k_list = [5, 10, 20]  # User-defined cluster cuts

    assign_df = pd.DataFrame(index=X_t.index)

    for k in k_list:
        assign_df[f'Cluster_k{k}'] = fcluster(
            Z, k, criterion='maxclust'
        )

    csv_path = os.path.join(
        feature_directory, f"ClusterAssignments_{g}.csv"
    )
    assign_df.to_csv(csv_path, index=True)

    print(f" → Cluster assignments saved at: {csv_path}")
