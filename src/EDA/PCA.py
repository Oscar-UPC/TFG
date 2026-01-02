import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# Config
directory = r'C:\\Users\\Oscar\\Desktop\\Trabajo\\ENGINE'
metrics = ['Coherence', 'Imag_Coherence', 'WPLI', 'WPPC']
n_components = 50  # number of components

# Read dataset
file_path = os.path.join(directory, 'Cleaned_data', 'Cleaned_data.csv')
df = pd.read_csv(file_path)

# Split labels and features
y = df.pop('Population')
X_all = df

# Define filters
filters = {
    'Coherence': lambda d: d.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda d: d.filter(regex='Imag_Coherence'),
    'WPLI': lambda d: d.filter(regex='WPLI'),
    'WPPC': lambda d: d.filter(regex='WPPC'),
}

# Output directory
root = Path(directory) / 'Results' / 'Unsupervised' / 'PCA' / 'Simple'
root.mkdir(parents=True, exist_ok=True)

for m in metrics:
    if m not in filters:
        continue
    X = filters[m](X_all)
    if X.shape[1] == 0:
        continue

    print(f"\n>>> PCA for {m} | samples={X.shape[0]} | features={X.shape[1]}")

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1]), random_state=0)
    X_pca_reduced = pca.fit_transform(X_scaled)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Cumulative explained variance and index where it reaches 95%
    cum_evr = pca.explained_variance_ratio_.cumsum()
    k95 = int(np.searchsorted(cum_evr, 0.95) + 1)  # +1 because PCs are 1-indexed
    print(f"{m}: 95% cumulative variance reached at k = {k95} PCs "
          f"(cumEVR={cum_evr[k95-1]:.3f})")
    
    # Plot cumulative explained variance
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(cum_evr) + 1), cum_evr, marker='o')
    plt.xlabel('Principal component')
    plt.ylabel('Cumulative explained variance')
    
    # Reference lines
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    plt.axvline(x=k95, color='r', linestyle='--')
    
    # Annotate k95
    plt.text(k95 - 0.5, 0.02 + min(0.98, cum_evr[k95-1]),
             f'k = {k95}', ha='right', va='bottom', fontsize=9)
        
    plt.legend()
    plt.title(f'Cumulative explained variance - {m}')
    
    # Ticks: X in steps of 5, Y in steps of 0.2
    plt.xticks(range(0, len(cum_evr) + 1, 2))
    plt.yticks(np.arange(0, 1.01, 0.1))
    
    plt.tight_layout()
    
    out_file = root / f'{m}_pca_cumulative.png'
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✔ Saved PCA cumulative variance plot for {m} -> {out_file}")

# 2x2 subplot with cumulative explained variance for all metrics
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Order: (1,1) Coherence, (1,2) Imag_Coherence, (2,1) WPLI, (2,2) WPPC
ordered_metrics = ['Coherence', 'Imag_Coherence', 'WPLI', 'WPPC']

for ax, m in zip(axes.flatten(), ordered_metrics):
    if m not in filters:
        continue
    X = filters[m](X_all)
    if X.shape[1] == 0:
        continue

    # Standardize
    X_scaled = StandardScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=min(50, X_scaled.shape[1]), random_state=0)
    pca.fit(X_scaled)

    # Cumulative explained variance
    cum_evr = pca.explained_variance_ratio_.cumsum()
    k95 = int(np.searchsorted(cum_evr, 0.95) + 1)

    # Plot in subplot
    ax.plot(range(1, len(cum_evr) + 1), cum_evr, marker='o')
    ax.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax.axvline(x=k95, color='r', linestyle='--')
    ax.text(k95 - 0.5, min(1.0, cum_evr[k95-1] + 0.02),
            f'k = {k95}', ha='right', va='bottom', fontsize=9)

    ax.set_title(f'{m}')
    ax.set_xlabel('Principal component')
    ax.set_ylabel('Cumulative explained variance')
    ax.set_xticks(range(0, len(cum_evr) + 1, 2))
    ax.set_yticks(np.arange(0, 1.01, 0.1))
    ax.set_ylim(0, 1.05)

plt.tight_layout()
out_file = root / 'pca_cumulative_all_metrics.png'
plt.savefig(out_file, dpi=300)
plt.close()
print(f"✔ Saved 2x2 subplot with cumulative variance curves -> {out_file}")