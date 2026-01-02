import pandas as pd
import re
from collections import Counter
from math import log2

# ============================================================
# 1. Helper functions (same logic as before)
# ============================================================

def extract_band(feature: str) -> str:
    """
    Extract the frequency band from a feature name based on common band keywords.
    Expected bands: Delta, Theta, Alpha, Beta, Gamma.
    """
    bands = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
    for b in bands:
        if b.lower() in feature.lower():
            return b
    return "Unknown"


def extract_regions(feature: str) -> list[str]:
    """
    Extract region tokens from a feature name.
    Regions are assumed to follow the pattern: letters + digits (e.g., Fp1, F3, T7).
    """
    parts = feature.split("_")
    return [p for p in parts if re.match(r"^[A-Za-z]+[0-9]+$", p)]


def entropy(counter: Counter) -> float:
    """
    Compute Shannon entropy from a Counter of categorical counts.
    """
    total = sum(counter.values())
    return -sum((c / total) * log2(c / total) for c in counter.values())


# ============================================================
# 2. Function to analyze a single cluster
# ============================================================

def analyze_single_cluster(cluster_name: str, features: list[str]) -> dict:
    """
    Given a list of feature names belonging to one cluster, summarize:
    - Most frequent band and its proportion
    - Band entropy (diversity)
    - Most frequent region and its proportion
    - Region entropy (diversity)
    """
    bands = [extract_band(f) for f in features]
    regions = [r for f in features for r in extract_regions(f)]

    band_counts = Counter(bands)
    region_counts = Counter(regions)

    # Majority band is always defined because 'features' should be non-empty
    majority_band, majority_band_count = band_counts.most_common(1)[0]

    # Region counts may be empty if region tokens are not detected
    majority_region, majority_region_count = (
        region_counts.most_common(1)[0] if region_counts else ("None", 0)
    )

    return {
        "cluster": cluster_name,
        "n_features": len(features),

        "majority_band": majority_band,
        "majority_band_pct": majority_band_count / len(features),
        "band_entropy": entropy(band_counts),

        "majority_region": majority_region,
        "majority_region_pct": (majority_region_count / len(regions)) if regions else 0,
        "region_entropy": entropy(region_counts) if regions else 0,
    }


# ============================================================
# 3. Main function: analyze ALL clusters stored in an Excel file
# ============================================================

def analyze_clusters_from_excel(path_to_excel: str, export_csv: bool = True) -> pd.DataFrame:
    """
    Reads an Excel file where each sheet corresponds to a cluster and contains
    a list of feature names (one per row). For each sheet, computes summary
    statistics using analyze_single_cluster().

    If export_csv=True, saves a CSV summary next to the input file.
    """
    xls = pd.ExcelFile(path_to_excel)
    sheet_names = xls.sheet_names

    results = []

    for sheet in sheet_names:
        # Read sheet (no header assumed)
        df = pd.read_excel(path_to_excel, sheet_name=sheet, header=None)

        # Assume the first column contains feature names
        col = df.columns[0]
        features = df[col].dropna().astype(str).tolist()

        cluster_result = analyze_single_cluster(sheet, features)
        results.append(cluster_result)

        print(f"[OK] Processed cluster '{sheet}' ({len(features)} features)")

    results_df = pd.DataFrame(results)

    if export_csv:
        out_path = path_to_excel.replace(".xlsx", "_cluster_analysis.csv")
        results_df.to_csv(out_path, index=False)
        print("\nCSV exported to:", out_path)

    return results_df


# ============================================================
# 4. Script execution
# ============================================================

excel_path = (
    r"C:\Users\Oscar\Desktop\Trabajo\ENGINE\Results\EDA\HierarchicalClustering\WPLI\WPLI_clusters_d30.xlsx"
)

# Run analysis and export CSV summary
df_results = analyze_clusters_from_excel(excel_path, export_csv=True)

# Also export results to Excel
output_excel = excel_path.replace(".xlsx", "_cluster_analysis.xlsx")
df_results.to_excel(output_excel, index=False)

print("\nOverall summary:")
print(df_results.head())

print("\nExcel file exported to:", output_excel)
