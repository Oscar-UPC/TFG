import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Initial config
directory = r'C:\\Users\\Oscar\\Desktop\\Trabajo\\ENGINE'
metrics = ['WPPC', 'WPLI', 'Imag_Coherence', 'Coherence']

# Read dataset
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
df = pd.read_csv(file_path)

# Split y and X
y = df.pop('Population')
X_all = df

# Define filters
filters = {
    'Coherence': lambda d: d.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda d: d.filter(regex='Imag_Coherence'),
    'WPLI': lambda d: d.filter(regex='WPLI'),
    'WPPC': lambda d: d.filter(regex='WPPC'),
}

# Output directory for figures
results_dir = Path(directory) / 'Results' / 'Unsupervised' / 'Correlation' / 'New'
results_dir.mkdir(parents=True, exist_ok=True)

# Correlation matrices per metric
for m in metrics:
    if m not in filters:
        print(f"[WARNING] No filter defined for '{m}'. Skipping.")
        continue

    X = filters[m](X_all)
    if X.shape[1] == 0:
        print(f"[WARNING] Filter '{m}' returned no columns. Check your CSV headers.")
        continue
  
    corr = X.corr().abs()
    print(f"Absolute correlation matrix for {m} (shape {corr.shape}):")
    print(corr.head())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', vmin=0, vmax=1,
            xticklabels=False, yticklabels=False)
    plt.title(f'Absolute correlation matrix - {m}')
    plt.tight_layout()
    
    out_file = results_dir / f'correlation_{m}.png'
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"✔ Saved absolute correlation heatmap for {m} -> {out_file}")
    

# 2x2 subplot with all metrics
    
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Order: (1,1) Coherence, (1,2) Imag_Coherence, (2,1) WPLI, (2,2) WPPC
ordered_metrics = ['Coherence', 'Imag_Coherence', 'WPLI', 'WPPC']

for ax, m in zip(axes.flatten(), ordered_metrics):
    if m not in filters:
        continue
    X = filters[m](X_all)
    if X.shape[1] == 0:
        continue

    corr = X.corr().abs()
    sns.heatmap(corr, cmap='coolwarm', vmin=0, vmax=1,
                xticklabels=False, yticklabels=False,
                ax=ax)
    ax.set_title(f'Absolute correlation - {m}')

plt.tight_layout()
out_file = results_dir / 'correlation_all_metrics.png'
plt.savefig(out_file, dpi=300)
plt.close()
print(f"✔ Saved 2x2 subplot with all metrics -> {out_file}")