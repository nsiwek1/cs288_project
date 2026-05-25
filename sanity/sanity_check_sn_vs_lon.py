"""
Sanity Check: Train cycle on SN, test generalization to LON
============================================================

SN (Kasekela) and LON (Mitumba) are chimps from different communities.
If the CycleGAN captures caller-specific acoustic structure, the cycle
trained on SN should reconstruct SN well but break down on LON.
"""

import json
import sys
import importlib.util
from pathlib import Path
from collections import defaultdict

# Remove project dir from sys.path to prevent code.py shadowing stdlib 'code'
_project_dir = str(Path(__file__).parent)
if _project_dir in sys.path:
    sys.path.remove(_project_dir)

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Re-add project dir and import code.py as chimp_code
sys.path.insert(0, _project_dir)
_spec = importlib.util.spec_from_file_location(
    "chimp_code",
    str(Path(__file__).parent / "code.py"),
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CycleGAN = _mod.CycleGAN
EMBEDDING_CACHE = _mod.EMBEDDING_CACHE
RESULTS_DIR = _mod.RESULTS_DIR
DEVICE = _mod.DEVICE
N_EPOCHS = _mod.N_EPOCHS

SANITY_DIR = RESULTS_DIR / "sanity_sn_vs_lon"
SANITY_DIR.mkdir(exist_ok=True)

TRAIN_CALLER = "SN"
TEST_CALLER = "LON"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_caller_from_filename(filename):
    """Extract caller ID from filename like 'KSK026_SN_BE3' or 'MTB052_LON_BE3'."""
    parts = filename.split("_")
    for part in parts:
        if part in (TRAIN_CALLER, TEST_CALLER):
            return part
    return None


def filter_by_caller(embeddings, metadata, caller):
    """Filter embeddings/metadata to a specific caller."""
    mask = np.array([get_caller_from_filename(m["filename"]) == caller
                     for m in metadata])
    return embeddings[mask], [m for m, keep in zip(metadata, mask) if keep]


def split_by_vtype(embeddings, metadata):
    """Split into pants (buildups) and hoots (climaxes)."""
    pant_mask = np.array([m["vtype"] == "pant" for m in metadata])
    return embeddings[pant_mask], embeddings[~pant_mask]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"SANITY CHECK: Train on {TRAIN_CALLER}, Test on {TEST_CALLER}")
    print("=" * 60)

    # Load cached embeddings
    print("\n[1] Loading embeddings...")
    data = np.load(EMBEDDING_CACHE, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata"]))
    print(f"  Total embeddings: {len(embeddings)}")

    # Filter to train and test callers
    train_emb, train_meta = filter_by_caller(embeddings, metadata, TRAIN_CALLER)
    test_emb, test_meta = filter_by_caller(embeddings, metadata, TEST_CALLER)

    train_pants, train_hoots = split_by_vtype(train_emb, train_meta)
    test_pants, test_hoots = split_by_vtype(test_emb, test_meta)

    print(f"\n  {TRAIN_CALLER}:  {len(train_pants)} pants, {len(train_hoots)} hoots")
    print(f"  {TEST_CALLER}: {len(test_pants)} pants, {len(test_hoots)} hoots")

    if len(test_pants) < 1 or len(test_hoots) < 1:
        print(f"\n  WARNING: {TEST_CALLER} has very few samples in one category.")
        print("  Results should be interpreted with caution.")

    # Train CycleGAN on train caller only
    print(f"\n[2] Training CycleGAN on {TRAIN_CALLER} calls only...")
    small_batch = min(8, len(train_pants), len(train_hoots))
    print(f"    Using batch_size={small_batch}")

    scaler_p = StandardScaler().fit(train_pants)
    scaler_h = StandardScaler().fit(train_hoots)
    p_s = scaler_p.transform(train_pants)
    h_s = scaler_h.transform(train_hoots)

    pant_ds = TensorDataset(torch.tensor(p_s, dtype=torch.float32))
    hoot_ds = TensorDataset(torch.tensor(h_s, dtype=torch.float32))
    pant_loader = DataLoader(pant_ds, batch_size=small_batch, shuffle=True, drop_last=True)
    hoot_loader = DataLoader(hoot_ds, batch_size=small_batch, shuffle=True, drop_last=True)

    model = CycleGAN(device=DEVICE)
    for epoch in range(N_EPOCHS):
        losses = model.train_epoch(pant_loader, hoot_loader)
        if (epoch + 1) % 100 == 0:
            print(f"      Epoch {epoch+1:4d} | G_cycle: {losses['G_cycle']:.4f}")
    print(f"    Final G_cycle: {losses['G_cycle']:.4f}")

    # Evaluate on both callers
    print("\n[3] Evaluating cycle reconstruction...")

    results = {}
    for name, pants, hoots in [(TRAIN_CALLER, train_pants, train_hoots),
                                (TEST_CALLER, test_pants, test_hoots)]:
        pants_s = scaler_p.transform(pants)
        hoots_s = scaler_h.transform(hoots)

        pant_losses = model.compute_cycle_loss(pants_s, "pant")
        hoot_losses = model.compute_cycle_loss(hoots_s, "hoot")

        results[name] = {
            "pant_losses": pant_losses,
            "hoot_losses": hoot_losses,
            "pant_mean": pant_losses.mean(),
            "hoot_mean": hoot_losses.mean(),
            "combined_losses": np.concatenate([pant_losses, hoot_losses]),
            "combined_mean": np.concatenate([pant_losses, hoot_losses]).mean(),
            "n_pants": len(pants),
            "n_hoots": len(hoots),
        }

        print(f"\n  {name}:")
        print(f"    Pant cycle loss:  {pant_losses.mean():.4f} +/- {pant_losses.std():.4f}  (n={len(pants)})")
        print(f"    Hoot cycle loss:  {hoot_losses.mean():.4f} +/- {hoot_losses.std():.4f}  (n={len(hoots)})")
        print(f"    Combined:         {results[name]['combined_mean']:.4f}")

    # Statistical comparison
    print(f"\n[4] Statistical comparison ({TRAIN_CALLER} vs {TEST_CALLER})...")
    train_combined = results[TRAIN_CALLER]["combined_losses"]
    test_combined = results[TEST_CALLER]["combined_losses"]

    u_stat, p_value = stats.mannwhitneyu(train_combined, test_combined, alternative="less")
    print(f"\n  Mann-Whitney U ({TRAIN_CALLER} < {TEST_CALLER}): U={u_stat:.1f}, p={p_value:.4e}")

    # Effect size
    n1, n2 = len(train_combined), len(test_combined)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)
    print(f"  Effect size (rank-biserial): {effect_size:.3f}")

    # Bootstrap CI on mean difference
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(10000):
        boot_train = rng.choice(train_combined, size=n1, replace=True)
        boot_test = rng.choice(test_combined, size=n2, replace=True)
        diffs.append(boot_test.mean() - boot_train.mean())
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  Mean diff ({TEST_CALLER} - {TRAIN_CALLER}): {test_combined.mean() - train_combined.mean():.4f}")
    print(f"  95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    if p_value < 0.05:
        print(f"\n  RESULT: {TEST_CALLER} has significantly higher cycle loss than {TRAIN_CALLER}.")
        print(f"  -> The {TRAIN_CALLER}-trained cycle DOES break down on {TEST_CALLER}, as expected.")
    else:
        print(f"\n  RESULT: No significant difference between {TRAIN_CALLER} and {TEST_CALLER}.")
        print(f"  -> The cycle generalizes across these callers (unexpected).")

    # Also get the KSK-wide baseline for context
    print("\n[5] Context: all-KSK baseline...")
    ksk_mask = np.array([m["community"] == "KSK" for m in metadata])
    ksk_emb = embeddings[ksk_mask]
    ksk_meta = [m for m, k in zip(metadata, ksk_mask) if k]

    # Exclude train and test callers
    other_mask = np.array([get_caller_from_filename(m["filename"]) not in (TRAIN_CALLER, TEST_CALLER)
                           for m in ksk_meta])
    other_emb = ksk_emb[other_mask]
    other_meta = [m for m, k in zip(ksk_meta, other_mask) if k]
    other_pants_mask = np.array([m["vtype"] == "pant" for m in other_meta])

    if other_pants_mask.sum() > 0 and (~other_pants_mask).sum() > 0:
        other_pants = other_emb[other_pants_mask]
        other_hoots = other_emb[~other_pants_mask]

        other_pants_s = scaler_p.transform(other_pants)
        other_hoots_s = scaler_h.transform(other_hoots)

        other_pant_losses = model.compute_cycle_loss(other_pants_s, "pant")
        other_hoot_losses = model.compute_cycle_loss(other_hoots_s, "hoot")
        other_combined = np.concatenate([other_pant_losses, other_hoot_losses])

        results["Other KSK"] = {
            "pant_losses": other_pant_losses,
            "hoot_losses": other_hoot_losses,
            "combined_losses": other_combined,
            "combined_mean": other_combined.mean(),
            "n_pants": len(other_pants),
            "n_hoots": len(other_hoots),
        }

        print(f"  Other KSK callers (excl. {TRAIN_CALLER} & {TEST_CALLER}): "
              f"combined={other_combined.mean():.4f} (n={len(other_combined)})")

    # Plot
    print("\n[6] Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot
    ax = axes[0]
    plot_data = []
    plot_labels = []
    plot_colors = []
    for name, color in [(TRAIN_CALLER, "#2ecc71"), (TEST_CALLER, "#e74c3c"), ("Other KSK", "#3498db")]:
        if name in results:
            plot_data.append(results[name]["combined_losses"])
            n = len(results[name]["combined_losses"])
            plot_labels.append(f"{name}\nn={n}")
            plot_colors.append(color)

    bp = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True, widths=0.6,
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))
    for patch, c in zip(bp["boxes"], plot_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)

    # Plot individual points
    for i, d in enumerate(plot_data):
        x = np.random.normal(i + 1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=15, color=plot_colors[i], edgecolors="black",
                   linewidths=0.3, zorder=3)

    ax.set_ylabel("L1 Cycle Reconstruction Error")
    ax.set_title(f"{TRAIN_CALLER}-Trained Cycle: Generalization Test\n"
                 f"(expect {TEST_CALLER} > {TRAIN_CALLER} if dialects differ)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add p-value annotation
    if len(plot_data) >= 2:
        y_max = max(d.max() for d in plot_data[:2]) * 1.05
        ax.plot([1, 2], [y_max, y_max], "k-", linewidth=1)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        ax.text(1.5, y_max * 1.01, f"p={p_value:.4e} {sig}",
                ha="center", va="bottom", fontsize=9)

    # Bar chart: pant vs hoot breakdown
    ax = axes[1]
    names = [TRAIN_CALLER, TEST_CALLER]
    x = np.arange(len(names))
    width = 0.35

    pant_means = [results[n]["pant_losses"].mean() for n in names]
    hoot_means = [results[n]["hoot_losses"].mean() for n in names]
    pant_stds = [results[n]["pant_losses"].std() for n in names]
    hoot_stds = [results[n]["hoot_losses"].std() for n in names]

    ax.bar(x - width/2, pant_means, width, yerr=pant_stds, label="Pant (buildup)",
           color="#3498db", alpha=0.7, capsize=3)
    ax.bar(x + width/2, hoot_means, width, yerr=hoot_stds, label="Hoot (climax)",
           color="#e74c3c", alpha=0.7, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n(pant n={results[n]['n_pants']}, hoot n={results[n]['n_hoots']})"
                        for n in names])
    ax.set_ylabel("Mean L1 Cycle Reconstruction Error")
    ax.set_title(f"Pant vs Hoot Breakdown\n({TRAIN_CALLER}-trained model)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"Sanity Check: Does the {TRAIN_CALLER}-trained cycle break down on {TEST_CALLER}?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SANITY_DIR / f"{TRAIN_CALLER.lower()}_vs_{TEST_CALLER.lower()}_sanity.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot")

    # Save raw results
    save_data = {}
    for name in results:
        safe_name = name.replace(" ", "_")
        save_data[f"{safe_name}_pant_losses"] = results[name]["pant_losses"]
        save_data[f"{safe_name}_hoot_losses"] = results[name]["hoot_losses"]
    np.savez(SANITY_DIR / f"{TRAIN_CALLER.lower()}_vs_{TEST_CALLER.lower()}_raw.npz", **save_data)

    # Summary report
    lines = [
        "=" * 60,
        f"SANITY CHECK: {TRAIN_CALLER}-trained CycleGAN -> {TEST_CALLER} generalization",
        "=" * 60,
        "",
        f"Training data: {TRAIN_CALLER} ({results[TRAIN_CALLER]['n_pants']} pants, {results[TRAIN_CALLER]['n_hoots']} hoots)",
        f"Test data:     {TEST_CALLER} ({results[TEST_CALLER]['n_pants']} pants, {results[TEST_CALLER]['n_hoots']} hoots)",
        "",
        f"{TRAIN_CALLER}  combined cycle loss: {results[TRAIN_CALLER]['combined_mean']:.4f}",
        f"{TEST_CALLER} combined cycle loss: {results[TEST_CALLER]['combined_mean']:.4f}",
        f"Difference ({TEST_CALLER} - {TRAIN_CALLER}):   {results[TEST_CALLER]['combined_mean'] - results[TRAIN_CALLER]['combined_mean']:.4f}",
        "",
        f"Mann-Whitney U ({TRAIN_CALLER} < {TEST_CALLER}): U={u_stat:.1f}, p={p_value:.4e}",
        f"Effect size: {effect_size:.3f}",
        f"95% CI on diff: [{ci_lo:.4f}, {ci_hi:.4f}]",
        "",
    ]
    if "Other KSK" in results:
        lines.append(f"Other KSK callers: {results['Other KSK']['combined_mean']:.4f} "
                     f"(n={len(results['Other KSK']['combined_losses'])})")
        lines.append("")

    if p_value < 0.05:
        lines.append(f"CONCLUSION: {TEST_CALLER} has significantly higher reconstruction error.")
        lines.append(f"The {TRAIN_CALLER}-trained cycle breaks down on {TEST_CALLER}.")
    else:
        lines.append("CONCLUSION: No significant difference detected.")
        lines.append(f"The cycle generalizes from {TRAIN_CALLER} to {TEST_CALLER}.")

    report = "\n".join(lines)
    report_path = SANITY_DIR / f"{TRAIN_CALLER.lower()}_vs_{TEST_CALLER.lower()}_report.txt"
    report_path.write_text(report)

    print("\n" + report)
    print("\nDone!")


if __name__ == "__main__":
    main()
