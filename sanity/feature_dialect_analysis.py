"""
Feature-Level Dialect Analysis
===============================

Identifies which specific acoustic features (F0, formants, duration, timbre)
make KSK-MTB closer than KSK-KAN, grounding the dialect finding in
interpretable biology.

Analyses:
  1. Per-feature effect sizes (Cohen's d) for KSK-MTB vs KSK-KAN
  2. LDA projection showing community separation
  3. Random Forest feature importance + confusion matrix
  4. Feature-by-feature statistical comparison table
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/natalia_mac/Downloads/Chimpanzee Pant-Hoots")
RESULTS_DIR = BASE_DIR / "results" / "feature_analysis"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Features to use (skip call, durat2, select)
FEATURES = [
    "duration", "F0start", "F0end", "F0max", "F0min", "F0mean",
    "lmmean", "lmmax", "F0loc", "trfak", "trmean", "trmax",
    "Pfstart", "Pfend", "Pfmax", "Pfmin", "Pfmean",
    "Pfmaxamp", "Pfminamp", "Pfmaxloc", "Pfminloc", "Pfmaxdif",
    "noise_mean", "noise_max",
]

# Feature categories for interpretation
FEATURE_CATEGORIES = {
    "Fundamental Freq": ["F0start", "F0end", "F0max", "F0min", "F0mean", "F0loc"],
    "Pitch Formant": ["Pfstart", "Pfend", "Pfmax", "Pfmin", "Pfmean",
                       "Pfmaxamp", "Pfminamp", "Pfmaxloc", "Pfminloc", "Pfmaxdif"],
    "Duration": ["duration"],
    "Loudness": ["lmmean", "lmmax"],
    "Timbre/Roughness": ["trfak", "trmean", "trmax"],
    "Noise": ["noise_mean", "noise_max"],
}

CATEGORY_COLORS = {
    "Fundamental Freq": "#e74c3c",
    "Pitch Formant": "#3498db",
    "Duration": "#2ecc71",
    "Loudness": "#f39c12",
    "Timbre/Roughness": "#9b59b6",
    "Noise": "#95a5a6",
}


def get_feature_category(feature):
    for cat, feats in FEATURE_CATEGORIES.items():
        if feature in feats:
            return cat
    return "Other"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_csv(filepath):
    """Load acoustic feature CSV (header on row 3, data from row 4)."""
    df = pd.read_csv(filepath, skiprows=2)
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    return df


def load_all_data():
    """Load all 6 CSV files, return combined DataFrame with community labels."""
    dfs = []
    for comm in ["KSK", "MTB", "KAN"]:
        for vtype, label in [("buildups", "pant"), ("climaxes", "hoot")]:
            filepath = BASE_DIR / f"{comm}_{vtype}.csv"
            df = load_csv(filepath)
            df["community"] = comm
            df["vtype"] = label
            dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    # Select only the features we want + metadata
    available = [f for f in FEATURES if f in combined.columns]
    missing = [f for f in FEATURES if f not in combined.columns]
    if missing:
        print(f"  Warning: missing features: {missing}")

    # Convert to numeric, drop rows with NaN in features
    for f in available:
        combined[f] = pd.to_numeric(combined[f], errors="coerce")

    n_before = len(combined)
    combined = combined.dropna(subset=available)
    n_after = len(combined)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows with NaN values")

    return combined, available


# ---------------------------------------------------------------------------
# Analysis 1: Per-Feature Effect Sizes
# ---------------------------------------------------------------------------
def cohens_d(x, y):
    """Compute Cohen's d (pooled SD)."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof=1)**2 + (ny - 1) * y.std(ddof=1)**2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (x.mean() - y.mean()) / pooled_std


def analysis_effect_sizes(data, features):
    """Compute and compare effect sizes for KSK-MTB vs KSK-KAN per feature."""
    print("\n[Analysis 1] Per-Feature Effect Sizes")

    results = []

    for vtype in ["pant", "hoot"]:
        vdata = data[data["vtype"] == vtype]
        ksk = vdata[vdata["community"] == "KSK"]
        mtb = vdata[vdata["community"] == "MTB"]
        kan = vdata[vdata["community"] == "KAN"]

        for feat in features:
            d_ksk_mtb = cohens_d(ksk[feat].values, mtb[feat].values)
            d_ksk_kan = cohens_d(ksk[feat].values, kan[feat].values)

            # Bootstrap CI on the difference |d_KSK_KAN| - |d_KSK_MTB|
            rng = np.random.default_rng(42)
            boot_diffs = []
            for _ in range(5000):
                b_ksk = rng.choice(ksk[feat].values, size=len(ksk), replace=True)
                b_mtb = rng.choice(mtb[feat].values, size=len(mtb), replace=True)
                b_kan = rng.choice(kan[feat].values, size=len(kan), replace=True)
                bd_mtb = cohens_d(b_ksk, b_mtb)
                bd_kan = cohens_d(b_ksk, b_kan)
                boot_diffs.append(abs(bd_kan) - abs(bd_mtb))
            boot_diffs = np.array(boot_diffs)
            ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

            results.append({
                "vtype": vtype,
                "feature": feat,
                "category": get_feature_category(feat),
                "d_KSK_MTB": d_ksk_mtb,
                "d_KSK_KAN": d_ksk_kan,
                "abs_diff": abs(d_ksk_kan) - abs(d_ksk_mtb),
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })

    df_results = pd.DataFrame(results)

    # Plot for each vtype
    for vtype in ["pant", "hoot"]:
        vtype_label = "Buildups (Pants)" if vtype == "pant" else "Climaxes (Hoots)"
        vr = df_results[df_results["vtype"] == vtype].sort_values("abs_diff", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Left: paired bar chart of |d| values
        ax = axes[0]
        y_pos = np.arange(len(vr))
        bar_height = 0.35

        ax.barh(y_pos - bar_height/2, vr["d_KSK_MTB"].abs().values,
                bar_height, label="KSK vs MTB", color="#3498db", alpha=0.7)
        ax.barh(y_pos + bar_height/2, vr["d_KSK_KAN"].abs().values,
                bar_height, label="KSK vs KAN", color="#e74c3c", alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['feature']} ({r['category'][:3]})" for _, r in vr.iterrows()],
                          fontsize=8)
        ax.set_xlabel("|Cohen's d|")
        ax.set_title(f"Effect Sizes: {vtype_label}", fontweight="bold")
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.3)

        # Right: difference plot (|d_KAN| - |d_MTB|)
        ax = axes[1]
        colors = [CATEGORY_COLORS.get(get_feature_category(f), "#333") for f in vr["feature"]]
        ax.barh(y_pos, vr["abs_diff"].values, color=colors, alpha=0.7)
        ax.errorbar(vr["abs_diff"].values, y_pos,
                    xerr=[vr["abs_diff"].values - vr["ci_lo"].values,
                          vr["ci_hi"].values - vr["abs_diff"].values],
                    fmt="none", color="black", capsize=2, linewidth=0.8)

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"{r['feature']}" for _, r in vr.iterrows()], fontsize=8)
        ax.set_xlabel("|d(KSK-KAN)| - |d(KSK-MTB)|")
        ax.set_title(f"Difference in Effect Sizes: {vtype_label}\n"
                     "(positive = KAN more different than MTB from KSK)",
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        # Legend for categories
        from matplotlib.patches import Patch
        legend_patches = [Patch(color=c, label=cat, alpha=0.7)
                         for cat, c in CATEGORY_COLORS.items()]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"effect_size_comparison_{vtype}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: effect_size_comparison_{vtype}.png")

    # Print top features where KAN is more different than MTB
    print("\n  Top features where |d(KSK-KAN)| >> |d(KSK-MTB)| (shared KSK-MTB dialect):")
    for vtype in ["pant", "hoot"]:
        vtype_label = "Buildups" if vtype == "pant" else "Climaxes"
        vr = df_results[df_results["vtype"] == vtype].sort_values("abs_diff", ascending=False)
        print(f"\n  {vtype_label}:")
        for _, row in vr.head(5).iterrows():
            sig = "*" if row["ci_lo"] > 0 else ""
            print(f"    {row['feature']:12s} ({row['category']:16s}): "
                  f"d(KSK-MTB)={row['d_KSK_MTB']:+.3f}, "
                  f"d(KSK-KAN)={row['d_KSK_KAN']:+.3f}, "
                  f"diff={row['abs_diff']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}] {sig}")

    return df_results


# ---------------------------------------------------------------------------
# Analysis 2: LDA Projection
# ---------------------------------------------------------------------------
def analysis_lda(data, features):
    """3-group LDA on raw features, project and visualize."""
    print("\n[Analysis 2] LDA Projection")

    for vtype in ["pant", "hoot"]:
        vtype_label = "Buildups (Pants)" if vtype == "pant" else "Climaxes (Hoots)"
        vdata = data[data["vtype"] == vtype]

        X = vdata[features].values
        y = vdata["community"].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        lda = LinearDiscriminantAnalysis()
        X_lda = lda.fit_transform(X_s, y)

        # Mahalanobis-like distances in LDA space
        centroids = {}
        for comm in ["KSK", "MTB", "KAN"]:
            mask = y == comm
            centroids[comm] = X_lda[mask].mean(axis=0)

        dist_ksk_mtb = np.linalg.norm(centroids["KSK"] - centroids["MTB"])
        dist_ksk_kan = np.linalg.norm(centroids["KSK"] - centroids["KAN"])
        dist_mtb_kan = np.linalg.norm(centroids["MTB"] - centroids["KAN"])

        print(f"\n  {vtype_label} — LDA centroid distances:")
        print(f"    KSK-MTB: {dist_ksk_mtb:.3f}")
        print(f"    KSK-KAN: {dist_ksk_kan:.3f}")
        print(f"    MTB-KAN: {dist_mtb_kan:.3f}")
        print(f"    Ratio KSK-KAN / KSK-MTB: {dist_ksk_kan / dist_ksk_mtb:.2f}x")

        # LDA loadings
        print(f"\n  Top LDA loadings (LD1):")
        loadings = lda.scalings_[:, 0]
        sorted_idx = np.argsort(np.abs(loadings))[::-1]
        for i in sorted_idx[:8]:
            print(f"    {features[i]:12s}: {loadings[i]:+.4f}")

        # Plot
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = {"KSK": "#2ecc71", "MTB": "#f39c12", "KAN": "#e74c3c"}
        comm_labels = {"KSK": "Kasekela", "MTB": "Mitumba", "KAN": "Kanyawara"}

        for comm in ["KSK", "MTB", "KAN"]:
            mask = y == comm
            ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
                      c=colors[comm], label=f"{comm_labels[comm]} (n={mask.sum()})",
                      alpha=0.5, s=25, edgecolors="black", linewidths=0.3)

        # Plot centroids
        for comm in ["KSK", "MTB", "KAN"]:
            ax.scatter(*centroids[comm], c=colors[comm], s=200, marker="D",
                      edgecolors="black", linewidths=2, zorder=5)

        # Draw distance lines
        for (a, b), ls in [(("KSK", "MTB"), "-"), (("KSK", "KAN"), "--"), (("MTB", "KAN"), ":")]:
            ax.plot([centroids[a][0], centroids[b][0]],
                    [centroids[a][1], centroids[b][1]],
                    "k" + ls, alpha=0.4, linewidth=1)

        ax.set_xlabel(f"LD1 ({lda.explained_variance_ratio_[0]*100:.1f}% variance)")
        ax.set_ylabel(f"LD2 ({lda.explained_variance_ratio_[1]*100:.1f}% variance)")
        ax.set_title(f"LDA Projection: {vtype_label}\n"
                     f"KSK-MTB={dist_ksk_mtb:.2f}, KSK-KAN={dist_ksk_kan:.2f}, "
                     f"MTB-KAN={dist_mtb_kan:.2f}",
                     fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"lda_projection_{vtype}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: lda_projection_{vtype}.png")


# ---------------------------------------------------------------------------
# Analysis 3: Random Forest Feature Importance
# ---------------------------------------------------------------------------
def analysis_random_forest(data, features):
    """Random Forest classifier with permutation importance and confusion matrix."""
    print("\n[Analysis 3] Random Forest Feature Importance")

    for vtype in ["pant", "hoot"]:
        vtype_label = "Buildups (Pants)" if vtype == "pant" else "Climaxes (Hoots)"
        vdata = data[data["vtype"] == vtype].copy()

        X = vdata[features].values
        y = vdata["community"].values

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # Cross-validated predictions for confusion matrix
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)

        y_pred = cross_val_predict(rf, X_s, y, cv=cv)

        cm = confusion_matrix(y, y_pred, labels=["KSK", "MTB", "KAN"])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        print(f"\n  {vtype_label} — Confusion Matrix (normalized):")
        print(f"  {'':>8s}  {'KSK':>6s}  {'MTB':>6s}  {'KAN':>6s}")
        for i, comm in enumerate(["KSK", "MTB", "KAN"]):
            print(f"  {comm:>8s}  {cm_norm[i,0]:6.2f}  {cm_norm[i,1]:6.2f}  {cm_norm[i,2]:6.2f}")

        # Key metric: KSK↔MTB confusion vs KSK↔KAN confusion
        ksk_as_mtb = cm_norm[0, 1]  # KSK misclassified as MTB
        mtb_as_ksk = cm_norm[1, 0]  # MTB misclassified as KSK
        ksk_as_kan = cm_norm[0, 2]  # KSK misclassified as KAN
        kan_as_ksk = cm_norm[2, 0]  # KAN misclassified as KSK

        print(f"\n  KSK↔MTB confusion: {(ksk_as_mtb + mtb_as_ksk)/2:.3f}")
        print(f"  KSK↔KAN confusion: {(ksk_as_kan + kan_as_ksk)/2:.3f}")

        # Fit full model for feature importance
        rf.fit(X_s, y)
        perm_imp = permutation_importance(rf, X_s, y, n_repeats=30,
                                          random_state=42, n_jobs=-1)

        # Sort by importance
        sorted_idx = np.argsort(perm_imp.importances_mean)[::-1]

        print(f"\n  Top features (permutation importance):")
        for i in sorted_idx[:8]:
            print(f"    {features[i]:12s} ({get_feature_category(features[i]):16s}): "
                  f"{perm_imp.importances_mean[i]:.4f} ± {perm_imp.importances_std[i]:.4f}")

        # Plot confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        im = ax.imshow(cm_norm, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        comm_labels = ["KSK\n(Kasekela)", "MTB\n(Mitumba)", "KAN\n(Kanyawara)"]
        ax.set_xticklabels(comm_labels)
        ax.set_yticklabels(comm_labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(3):
            for j in range(3):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm[i,j]})",
                       ha="center", va="center", color=color, fontsize=10)
        ax.set_title(f"RF Confusion Matrix: {vtype_label}\n(5-fold CV)", fontweight="bold")
        plt.colorbar(im, ax=ax)

        # Plot feature importance
        ax = axes[1]
        top_n = min(15, len(features))
        top_idx = sorted_idx[:top_n][::-1]  # reverse for horizontal bar
        colors = [CATEGORY_COLORS.get(get_feature_category(features[i]), "#333") for i in top_idx]
        ax.barh(range(top_n), perm_imp.importances_mean[top_idx], color=colors, alpha=0.7)
        ax.errorbar(perm_imp.importances_mean[top_idx], range(top_n),
                    xerr=perm_imp.importances_std[top_idx],
                    fmt="none", color="black", capsize=2)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([features[i] for i in top_idx], fontsize=8)
        ax.set_xlabel("Permutation Importance")
        ax.set_title(f"Feature Importance: {vtype_label}", fontweight="bold")
        ax.grid(axis="x", alpha=0.3)

        from matplotlib.patches import Patch
        legend_patches = [Patch(color=c, label=cat, alpha=0.7)
                         for cat, c in CATEGORY_COLORS.items()]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7)

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"rf_analysis_{vtype}.png",
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: rf_analysis_{vtype}.png")


# ---------------------------------------------------------------------------
# Analysis 4: Feature-by-Feature Comparison Table
# ---------------------------------------------------------------------------
def analysis_feature_table(data, features):
    """Per-feature stats: mean±SD per community, Kruskal-Wallis, pairwise Mann-Whitney."""
    print("\n[Analysis 4] Feature-by-Feature Community Comparison")

    rows = []
    for vtype in ["pant", "hoot"]:
        vdata = data[data["vtype"] == vtype]
        ksk = vdata[vdata["community"] == "KSK"]
        mtb = vdata[vdata["community"] == "MTB"]
        kan = vdata[vdata["community"] == "KAN"]

        for feat in features:
            ksk_vals = ksk[feat].values
            mtb_vals = mtb[feat].values
            kan_vals = kan[feat].values

            # Kruskal-Wallis
            kw_stat, kw_p = stats.kruskal(ksk_vals, mtb_vals, kan_vals)

            # Pairwise Mann-Whitney U (two-sided)
            u_km, p_km = stats.mannwhitneyu(ksk_vals, mtb_vals, alternative="two-sided")
            u_kk, p_kk = stats.mannwhitneyu(ksk_vals, kan_vals, alternative="two-sided")
            u_mk, p_mk = stats.mannwhitneyu(mtb_vals, kan_vals, alternative="two-sided")

            # Bonferroni correction (3 comparisons)
            p_km_corr = min(p_km * 3, 1.0)
            p_kk_corr = min(p_kk * 3, 1.0)
            p_mk_corr = min(p_mk * 3, 1.0)

            row = {
                "vtype": vtype,
                "feature": feat,
                "category": get_feature_category(feat),
                "KSK_mean": ksk_vals.mean(),
                "KSK_sd": ksk_vals.std(),
                "MTB_mean": mtb_vals.mean(),
                "MTB_sd": mtb_vals.std(),
                "KAN_mean": kan_vals.mean(),
                "KAN_sd": kan_vals.std(),
                "KW_stat": kw_stat,
                "KW_p": kw_p,
                "p_KSK_MTB": p_km_corr,
                "p_KSK_KAN": p_kk_corr,
                "p_MTB_KAN": p_mk_corr,
                "sig_KSK_MTB": "***" if p_km_corr < 0.001 else "**" if p_km_corr < 0.01 else "*" if p_km_corr < 0.05 else "n.s.",
                "sig_KSK_KAN": "***" if p_kk_corr < 0.001 else "**" if p_kk_corr < 0.01 else "*" if p_kk_corr < 0.05 else "n.s.",
                "sig_MTB_KAN": "***" if p_mk_corr < 0.001 else "**" if p_mk_corr < 0.01 else "*" if p_mk_corr < 0.05 else "n.s.",
            }
            rows.append(row)

    df_table = pd.DataFrame(rows)
    df_table.to_csv(RESULTS_DIR / "feature_comparison_table.csv", index=False)
    print(f"  Saved: feature_comparison_table.csv")

    # Highlight "shared dialect" features: KSK-MTB NOT sig, but KSK-KAN IS sig
    print("\n  'Shared dialect' features (KSK-MTB n.s. but KSK-KAN significant):")
    for vtype in ["pant", "hoot"]:
        vtype_label = "Buildups" if vtype == "pant" else "Climaxes"
        vt = df_table[df_table["vtype"] == vtype]
        shared = vt[(vt["p_KSK_MTB"] >= 0.05) & (vt["p_KSK_KAN"] < 0.05)]

        if len(shared) > 0:
            print(f"\n  {vtype_label}:")
            for _, row in shared.iterrows():
                print(f"    {row['feature']:12s} ({row['category']:16s}): "
                      f"KSK-MTB p={row['p_KSK_MTB']:.4f} {row['sig_KSK_MTB']}, "
                      f"KSK-KAN p={row['p_KSK_KAN']:.4f} {row['sig_KSK_KAN']}")
        else:
            print(f"\n  {vtype_label}: none found")

    # Also show features where ALL pairs are significant (general dialect markers)
    print("\n  Features significant for ALL community pairs:")
    for vtype in ["pant", "hoot"]:
        vtype_label = "Buildups" if vtype == "pant" else "Climaxes"
        vt = df_table[df_table["vtype"] == vtype]
        all_sig = vt[(vt["p_KSK_MTB"] < 0.05) & (vt["p_KSK_KAN"] < 0.05) & (vt["p_MTB_KAN"] < 0.05)]
        if len(all_sig) > 0:
            print(f"\n  {vtype_label}: {', '.join(all_sig['feature'].values)}")
        else:
            print(f"\n  {vtype_label}: none")

    return df_table


# ---------------------------------------------------------------------------
# Summary Report
# ---------------------------------------------------------------------------
def write_report(effect_df, feature_table):
    """Write a text summary of all findings."""
    lines = [
        "=" * 70,
        "FEATURE-LEVEL DIALECT ANALYSIS REPORT",
        "Which acoustic features make KSK-MTB closer than KSK-KAN?",
        "=" * 70,
        "",
    ]

    for vtype in ["pant", "hoot"]:
        vtype_label = "BUILDUPS (Pants)" if vtype == "pant" else "CLIMAXES (Hoots)"
        lines.append(f"--- {vtype_label} ---")
        lines.append("")

        # Top shared-dialect features
        vt = feature_table[feature_table["vtype"] == vtype]
        shared = vt[(vt["p_KSK_MTB"] >= 0.05) & (vt["p_KSK_KAN"] < 0.05)]
        lines.append(f"Shared dialect features (KSK≈MTB but KSK≠KAN): {len(shared)}")
        for _, row in shared.iterrows():
            lines.append(f"  {row['feature']:12s} ({row['category']}): "
                        f"KSK={row['KSK_mean']:.1f}±{row['KSK_sd']:.1f}, "
                        f"MTB={row['MTB_mean']:.1f}±{row['MTB_sd']:.1f}, "
                        f"KAN={row['KAN_mean']:.1f}±{row['KAN_sd']:.1f}")
        lines.append("")

        # Top effect size differences
        ve = effect_df[effect_df["vtype"] == vtype].sort_values("abs_diff", ascending=False)
        lines.append("Top features by |d(KSK-KAN)| - |d(KSK-MTB)|:")
        for _, row in ve.head(5).iterrows():
            lines.append(f"  {row['feature']:12s}: d(KSK-MTB)={row['d_KSK_MTB']:+.3f}, "
                        f"d(KSK-KAN)={row['d_KSK_KAN']:+.3f}, diff={row['abs_diff']:.3f}")
        lines.append("")

        # Category summary
        lines.append("Effect size difference by feature category:")
        for cat in FEATURE_CATEGORIES:
            cat_feats = FEATURE_CATEGORIES[cat]
            cat_data = ve[ve["feature"].isin(cat_feats)]
            if len(cat_data) > 0:
                mean_diff = cat_data["abs_diff"].mean()
                lines.append(f"  {cat:20s}: mean diff = {mean_diff:.3f} "
                            f"({'KAN more different' if mean_diff > 0 else 'MTB more different'})")
        lines.append("")

    report = "\n".join(lines)
    (RESULTS_DIR / "feature_analysis_report.txt").write_text(report)
    print(f"\n  Saved: feature_analysis_report.txt")
    print("\n" + report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FEATURE-LEVEL DIALECT ANALYSIS")
    print("Which acoustic features make KSK-MTB closer than KSK-KAN?")
    print("=" * 70)

    # Load data
    print("\n[Loading Data]")
    data, features = load_all_data()
    print(f"  Loaded {len(data)} calls with {len(features)} features")
    for comm in ["KSK", "MTB", "KAN"]:
        for vtype in ["pant", "hoot"]:
            n = len(data[(data["community"] == comm) & (data["vtype"] == vtype)])
            print(f"    {comm} {vtype}: {n}")

    # Run analyses
    effect_df = analysis_effect_sizes(data, features)
    analysis_lda(data, features)
    analysis_random_forest(data, features)
    feature_table = analysis_feature_table(data, features)

    # Summary
    write_report(effect_df, feature_table)
    print("\nDone!")


if __name__ == "__main__":
    main()
