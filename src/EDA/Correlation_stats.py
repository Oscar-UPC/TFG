
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Initial config
directory = r'C:\\Users\\Oscar\\Desktop\\Trabajo\\ENGINE'
metrics = ['WPPC', 'WPLI', 'Imag_Coherence', 'Coherence']

# Read dataset (same step as your original script)
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
df = pd.read_csv(file_path)

# Split y and X
y = df.pop('Population')
X_all = df

# Define filters (EXACT same as in your script)
filters = {
    'Coherence': lambda d: d.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda d: d.filter(regex='Imag_Coherence'),
    'WPLI': lambda d: d.filter(regex='WPLI'),
    'WPPC': lambda d: d.filter(regex='WPPC'),
}

# Output directory
results_dir = Path(directory) / 'Results' / 'Unsupervised' / 'Correlation' / 'New'
results_dir.mkdir(parents=True, exist_ok=True)

# Dictionary for saving stats
correlation_stats = {}

print("\n====== CORRELATION PAIR ANALYSIS ======\n")

for m in metrics:
    if m not in filters:
        print(f"[WARNING] No filter for '{m}'. Skipping.")
        continue

    # Extract subset of features for that metric
    X = filters[m](X_all)
    cols = X.columns
    n_features = len(cols)

    if n_features < 2:
        print(f"[WARNING] '{m}' has fewer than 2 columns. Skipping.")
        continue

    # Compute absolute Pearson correlation
    corr = X.corr().abs()

    # Extract upper triangle (excludes diagonal)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    # Flatten and drop NaN
    values = upper.unstack().dropna()

    # Thresholds
    high_mask = (values >= 0.7) & (values < 0.9)
    very_high_mask = values >= 0.9

    # Counts
    high_corr_count = high_mask.sum()
    very_high_corr_count = very_high_mask.sum()

    total_pairs = len(values)

    correlation_stats[m] = {
        "n_features": n_features,
        "total_pairs": total_pairs,
        "high_corr_pairs (0.7-0.9)": high_corr_count,
        "high_corr_pct": high_corr_count / total_pairs * 100,
        "very_high_corr_pairs (>=0.9)": very_high_corr_count,
        "very_high_corr_pct": very_high_corr_count / total_pairs * 100
    }

    # Console output
    print(f"=== {m} ===")
    print(f"Features: {n_features}")
    print(f"Total pairs: {total_pairs}")
    print(f"High corr (0.7–0.9): {high_corr_count} ({high_corr_count/total_pairs*100:.3f}%)")
    print(f"Very high corr (>=0.9): {very_high_corr_count} ({very_high_corr_count/total_pairs*100:.3f}%)\n")

# Save stats
stats_df = pd.DataFrame.from_dict(correlation_stats, orient='index')
stats_df.to_csv(results_dir / "correlation_threshold_counts.csv")

print("✔ Finished. Summary saved to correlation_threshold_counts.csv")
