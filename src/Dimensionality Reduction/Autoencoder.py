import os
import numpy as np
import pandas as pd
from pathlib import Path

# === Sklearn ===
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_curve, auc,
    average_precision_score,
    precision_recall_curve,
    brier_score_loss
)
from scipy import stats  # for t/z critical values

# === PyTorch ===
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# === GENERAL CONFIGURATION ===
n_jobs = -1
directory = r'/homes/3/om871/Documents/ML'
features = ['WPPC', 'WPLI', 'Imag_Coherence', 'Coherence']

# === LOAD DATA ===
file_path = os.path.join(directory, 'Cleaned_data/Cleaned_data.csv')
clf_data = pd.read_csv(file_path)

# HC = 0, MDD = 1
y = clf_data.pop('Population')
X_all = clf_data

# === Feature filters ===
filters = {
    'Coherence': lambda df: df.filter(regex='Coherence').filter(regex='^((?!Imag_Coherence).)*$'),
    'Imag_Coherence': lambda df: df.filter(regex='Imag_Coherence'),
    'WPLI': lambda df: df.filter(regex='WPLI'),
    'WPPC': lambda df: df.filter(regex='WPPC'),
}

# === CV ===
outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=42)

# === Dimensionalities to test (latent size) ===
latent_dims_list = [10, 20, 30]

# === Torch helpers ===
def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden=128, dropout=0.2):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim)
        )
    def forward(self, x):
        z = self.enc(x)
        x_rec = self.dec(z)
        return x_rec, z

def train_autoencoder(
    X_train_np, 
    input_dim, latent_dim,
    noise_std=0.1, batch_size=16, 
    max_epochs=400, patience=20,
    lr=1e-3, weight_decay=1e-4, 
    val_split=0.2, seed=42
):
    torch.manual_seed(seed)
    device = get_device()

    X_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    dataset = TensorDataset(X_tensor)
    n_total = len(dataset)
    n_val = int(val_split * n_total) if n_total > 5 else min(5, max(1, n_total//5))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = DenoisingAutoencoder(input_dim, latent_dim, hidden=min(128, max(32, input_dim//4)), dropout=0.2).to(device)
    criterion = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = np.inf
    best_state = None
    wait = 0

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            # Denoising: añadir ruido gaussiano sólo en el forward de entrenamiento
            noise = torch.randn_like(xb) * noise_std
            xb_noisy = xb + noise
            xr, _ = model(xb_noisy)
            loss = criterion(xr, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                xr, _ = model(xb)
                loss = criterion(xr, xb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= max(1, len(val_loader.dataset))

        # Early stopping
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model

def encode_with_model(model, X_np):
    device = get_device()
    with torch.no_grad():
        X_tensor = torch.tensor(X_np, dtype=torch.float32).to(device)
        _, Z = model(X_tensor)
        return Z.cpu().numpy()

def ci95_bf(values, k=5, m=20, n2_n1=0.25, alpha=0.05):
    """
    Bouckaert & Frank (2004) corrected repeated k-fold CI
    """
    values = np.array(values, dtype=float)
    mean = np.mean(values)
    sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
    se = sd * np.sqrt((1 / (k * m)) + n2_n1)
    df = k * m - 1
    tval = stats.t.ppf(1 - alpha/2, df=df) if df > 0 else 0.0
    ci_lower = mean - tval * se
    ci_upper = mean + tval * se
    return mean, sd, ci_lower, ci_upper

# === Interpolación para curvas ===
interpolation = np.linspace(0, 1, 100)

# === MAIN LOOP over latent dims ===
for z_dim in latent_dims_list:

    print(f"\n=== Running AE+RF with latent dim = {z_dim} ===")

    results_directory = os.path.join(directory, f'Results/AE_RF/AE_{z_dim}_latent')
    Path(results_directory).mkdir(parents=True, exist_ok=True)

    # Storage
    tprs = {f: [] for f in features}
    precisionspr = {f: [] for f in features}
    rocs = {f: [] for f in features}
    APscores = {f: [] for f in features}
    HC_precision = {f: [] for f in features}
    HC_recall = {f: [] for f in features}
    HC_f1 = {f: [] for f in features}
    MDD_precision = {f: [] for f in features}
    MDD_recall = {f: [] for f in features}
    MDD_f1 = {f: [] for f in features}
    accuracy = {f: [] for f in features}
    brier_scores = {f: [] for f in features}

    for g in features:
        print(f" → Processing feature type: {g}")

        if g in filters:
            X = filters[g](X_all)
        else:
            X = X_all

        feature_directory = os.path.join(results_directory, g)
        Path(feature_directory).mkdir(parents=True, exist_ok=True)

        X_np_full = X.to_numpy(dtype=float)
        y_np_full = y.to_numpy()

        # === CV Loop ===
        for train_ix, test_ix in outer_cv.split(X_np_full, y_np_full):
            X_train_raw, X_test_raw = X_np_full[train_ix], X_np_full[test_ix]
            y_train, y_test = y_np_full[train_ix], y_np_full[test_ix]

            # --- Standardize per-fold ---
            scaler = StandardScaler(with_mean=True, with_std=True)
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            input_dim = X_train.shape[1]

            # --- Train AE on X_train only ---
            ae = train_autoencoder(
                X_train_np=X_train,
                input_dim=input_dim,
                latent_dim=z_dim,
                noise_std=0.1,
                batch_size=16,
                max_epochs=400,
                patience=30,
                lr=1e-3,
                weight_decay=1e-4,
                val_split=0.2,
                seed=42
            )

            # --- Encode train/test ---
            X_train_z = encode_with_model(ae, X_train)
            X_test_z = encode_with_model(ae, X_test)

            # --- RF on latent codes ---
            rf_final = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)
            rf_final.fit(X_train_z, y_train)

            probas_ = rf_final.predict_proba(X_test_z)
            y_pred = rf_final.predict(X_test_z)

            # ROC
            fpr, tpr, _ = roc_curve(y_test, probas_[:, 1])
            # proteger interpolación si FPR no está ordenado/único
            fpr_sorted_idx = np.argsort(fpr)
            fpr_sorted = fpr[fpr_sorted_idx]
            tpr_sorted = tpr[fpr_sorted_idx]
            # eliminar duplicados en FPR para np.interp
            fpr_unique, unique_idx = np.unique(fpr_sorted, return_index=True)
            tpr_unique = tpr_sorted[unique_idx]

            tprs[g].append(np.interp(interpolation, fpr_unique, tpr_unique))
            tprs[g][-1][0] = 0.0
            rocs[g].append(auc(fpr, tpr))

            # PR
            precision, recall, _ = precision_recall_curve(y_test, probas_[:, 1])
            APscores[g].append(average_precision_score(y_test, probas_[:, 1]))
            # ordenar recall ascendente para interp sobre [0,1]
            recall_sorted_idx = np.argsort(recall)
            recall_sorted = recall[recall_sorted_idx]
            precision_sorted = precision[recall_sorted_idx]
            recall_unique, unique_idx2 = np.unique(recall_sorted, return_index=True)
            precision_unique = precision_sorted[unique_idx2]

            precisionspr[g].append(np.interp(interpolation, recall_unique, precision_unique))
            precisionspr[g][-1][0] = 1.0

            # Classification report
            report = classification_report(y_test, y_pred, target_names=['HC', 'MDD'], output_dict=True, zero_division=0)
            HC_precision[g].append(report['HC']['precision'])
            HC_recall[g].append(report['HC']['recall'])
            HC_f1[g].append(report['HC']['f1-score'])
            MDD_precision[g].append(report['MDD']['precision'])
            MDD_recall[g].append(report['MDD']['recall'])
            MDD_f1[g].append(report['MDD']['f1-score'])
            accuracy[g].append(report['accuracy'])
            brier_scores[g].append(brier_score_loss(y_test, probas_[:, 1]))

        # === Summary & exports ===
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

        metrics_df = pd.DataFrame(metrics_summary, columns=["Metric", "Mean", "Std", "CI Lower", "CI Upper"])
        metrics_df.to_excel(os.path.join(feature_directory, f"{g}_metrics.xlsx"), index=False)

        # ROC / PR curve envelopes
        mean_tpr = np.mean(tprs[g], axis=0) if len(tprs[g]) else np.zeros_like(interpolation)
        lower_bound = np.percentile(tprs[g], 2.5, axis=0) if len(tprs[g]) else np.zeros_like(interpolation)
        upper_bound = np.percentile(tprs[g], 97.5, axis=0) if len(tprs[g]) else np.zeros_like(interpolation)

        roc_df = pd.DataFrame({
            "FPR": interpolation,
            "TPR_mean": mean_tpr,
            "TPR_lower": lower_bound,
            "TPR_upper": upper_bound
        })
        roc_df.to_excel(os.path.join(feature_directory, f"{g}_roc_curve.xlsx"), index=False)

        mean_precision = np.mean(precisionspr[g], axis=0) if len(precisionspr[g]) else np.zeros_like(interpolation)
        lower_pr = np.percentile(precisionspr[g], 2.5, axis=0) if len(precisionspr[g]) else np.zeros_like(interpolation)
        upper_pr = np.percentile(precisionspr[g], 97.5, axis=0) if len(precisionspr[g]) else np.zeros_like(interpolation)

        pr_df = pd.DataFrame({
            "Recall": interpolation,
            "Precision_mean": mean_precision,
            "Precision_lower": lower_pr,
            "Precision_upper": upper_pr
        })
        pr_df.to_excel(os.path.join(feature_directory, f"{g}_pr_curve.xlsx"), index=False)

    print(f"✅ Finished latent={z_dim}. Results saved in: {results_directory}")
