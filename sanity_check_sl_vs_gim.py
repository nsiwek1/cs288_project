"""
Sanity Check: Train cycle on SL, test generalization to GIM
============================================================

SL and GIM are Kasekela chimps with the strongest opposing dialects.
If the CycleGAN captures caller-specific acoustic structure, the cycle
trained on SL should reconstruct SL well but break down on GIM.

Data:
  SL:  14 buildups, 12 climaxes
  GIM:  2 buildups,  4 climaxes
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
BATCH_SIZE = _mod.BATCH_SIZE
N_EPOCHS = _mod.N_EPOCHS
COMMUNITIES = _mod.COMMUNITIES
train_cyclegan_single = _mod.train_cyclegan_single

BASE_DIR = Path("/Users/natalia_mac/Downloads/Chimpanzee Pant-Hoots")
SANITY_DIR = RESULTS_DIR / "sanity_sl_vs_gim"
SANITY_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_caller_from_filename(filename):
    """Extract caller ID from filename like 'KSK267_GIM_CS2'."""
    parts = filename.split("_")
    # Naming: {RecID}_{Caller}_{CallType} but RecID can have underscores
    # e.g., KSK241_1_SL_BE2 -> caller is SL
    # Strategy: caller is always 2nd-to-last segment, call type is last
    # But some call types have numbers like CS10, BE14
    # Actually from the data: the caller is a short alpha code (2-3 chars)
    # and appears between the recording ID and the call type code
    # Let's match known callers
    for i, part in enumerate(parts):
        if part in ("SL", "GIM"):
            return part
    return None


def filter_by_caller(embeddings, metadata, caller):
    """Filter embeddings/metadata to a specific caller."""
    mask = np.array([get_caller_from_filename(m["filename"]) == caller
                     for m in metadata])
    filtered_emb = embeddings[mask]
    filtered_meta = [m for m, keep in zip(metadata, mask) if keep]
    return filtered_emb, filtered_meta


def split_by_vtype(embeddings, metadata):
    """Split into pants (buildups) and hoots (climaxes)."""
    pant_mask = np.array([m["vtype"] == "pant" for m in metadata])
    hoot_mask = ~pant_mask
    return embeddings[pant_mask], embeddings[hoot_mask]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("SANITY CHECK: Train on SL, Test on GIM")
    print("=" * 60)

    # Load cached embeddings
    print("\n[1] Loading embeddings...")
    data = np.load(EMBEDDING_CACHE, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata"]))
    print(f"  Total embeddings: {len(embeddings)}")

    # Filter to SL and GIM
    sl_emb, sl_meta = filter_by_caller(embeddings, metadata, "SL")
    gim_emb, gim_meta = filter_by_caller(embeddings, metadata, "GIM")

    sl_pants, sl_hoots = split_by_vtype(sl_emb, sl_meta)
    gim_pants, gim_hoots = split_by_vtype(gim_emb, gim_meta)

    print(f"\n  SL:  {len(sl_pants)} pants, {len(sl_hoots)} hoots")
    print(f"  GIM: {len(gim_pants)} pants, {len(gim_hoots)} hoots")

    if len(gim_pants) < 1 or len(gim_hoots) < 1:
        print("\n  WARNING: GIM has very few samples in one category.")
        print("  Results should be interpreted with caution.")

    # Train CycleGAN on SL only (small batch size for small dataset)
    print("\n[2] Training CycleGAN on SL calls only...")
    small_batch = min(4, len(sl_pants), len(sl_hoots))
    print(f"    Using batch_size={small_batch} (dataset is small)")

    scaler_p = StandardScaler().fit(sl_pants)
    scaler_h = StandardScaler().fit(sl_hoots)
    p_s = scaler_p.transform(sl_pants)
    h_s = scaler_h.transform(sl_hoots)

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

    # Evaluate on both SL and GIM
    print("\n[3] Evaluating cycle reconstruction...")

    results = {}
    for name, pants, hoots in [("SL", sl_pants, sl_hoots),
                                ("GIM", gim_pants, gim_hoots)]:
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
    print("\n[4] Statistical comparison (SL vs GIM)...")
    sl_combined = results["SL"]["combined_losses"]
    gim_combined = results["GIM"]["combined_losses"]

    u_stat, p_value = stats.mannwhitneyu(sl_combined, gim_combined, alternative="less")
    print(f"\n  Mann-Whitney U (SL < GIM): U={u_stat:.1f}, p={p_value:.4f}")

    # Effect size
    n1, n2 = len(sl_combined), len(gim_combined)
    effect_size = 1 - (2 * u_stat) / (n1 * n2)
    print(f"  Effect size (rank-biserial): {effect_size:.3f}")

    # Bootstrap CI on mean difference
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(10000):
        boot_sl = rng.choice(sl_combined, size=n1, replace=True)
        boot_gim = rng.choice(gim_combined, size=n2, replace=True)
        diffs.append(boot_gim.mean() - boot_sl.mean())
    diffs = np.array(diffs)
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    print(f"  Mean diff (GIM - SL): {gim_combined.mean() - sl_combined.mean():.4f}")
    print(f"  95% Bootstrap CI: [{ci_lo:.4f}, {ci_hi:.4f}]")

    if p_value < 0.05:
        print("\n  RESULT: GIM has significantly higher cycle loss than SL.")
        print("  -> The SL-trained cycle DOES break down on GIM, as expected.")
    else:
        print("\n  RESULT: No significant difference between SL and GIM.")
        print("  -> The cycle generalizes across these callers (unexpected).")

    # Also get the KSK-wide baseline for context
    print("\n[5] Context: all-KSK baseline...")
    ksk_mask = np.array([m["community"] == "KSK" for m in metadata])
    ksk_emb = embeddings[ksk_mask]
    ksk_meta = [m for m, k in zip(metadata, ksk_mask) if k]

    # Exclude SL and GIM
    other_mask = np.array([get_caller_from_filename(m["filename"]) not in ("SL", "GIM")
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

        print(f"  Other KSK callers (excl. SL & GIM): "
              f"combined={other_combined.mean():.4f} (n={len(other_combined)})")

    # Plot
    print("\n[6] Generating plots...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Boxplot: SL vs GIM vs Other KSK
    ax = axes[0]
    plot_data = []
    plot_labels = []
    plot_colors = []
    for name, color in [("SL", "#2ecc71"), ("GIM", "#e74c3c"), ("Other KSK", "#3498db")]:
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
    ax.set_title("SL-Trained Cycle: Generalization Test\n"
                 "(expect GIM > SL if dialects differ)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add p-value annotation
    if len(plot_data) >= 2:
        y_max = max(d.max() for d in plot_data[:2]) * 1.05
        ax.plot([1, 2], [y_max, y_max], "k-", linewidth=1)
        sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
        ax.text(1.5, y_max * 1.01, f"p={p_value:.4f} {sig}",
                ha="center", va="bottom", fontsize=9)

    # Bar chart: pant vs hoot breakdown
    ax = axes[1]
    names = ["SL", "GIM"]
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
    ax.set_title("Pant vs Hoot Breakdown\n(SL-trained model)", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Sanity Check: Does the SL-trained cycle break down on GIM?",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(SANITY_DIR / "sl_vs_gim_sanity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {SANITY_DIR / 'sl_vs_gim_sanity.png'}")

    # Save raw results
    save_data = {}
    for name in results:
        safe_name = name.replace(" ", "_")
        save_data[f"{safe_name}_pant_losses"] = results[name]["pant_losses"]
        save_data[f"{safe_name}_hoot_losses"] = results[name]["hoot_losses"]
    np.savez(SANITY_DIR / "sl_vs_gim_raw.npz", **save_data)
    print(f"  Saved: {SANITY_DIR / 'sl_vs_gim_raw.npz'}")

    # Summary report
    lines = [
        "=" * 60,
        "SANITY CHECK: SL-trained CycleGAN -> GIM generalization",
        "=" * 60,
        "",
        f"Training data: SL ({results['SL']['n_pants']} pants, {results['SL']['n_hoots']} hoots)",
        f"Test data:     GIM ({results['GIM']['n_pants']} pants, {results['GIM']['n_hoots']} hoots)",
        "",
        f"SL  combined cycle loss: {results['SL']['combined_mean']:.4f}",
        f"GIM combined cycle loss: {results['GIM']['combined_mean']:.4f}",
        f"Difference (GIM - SL):   {results['GIM']['combined_mean'] - results['SL']['combined_mean']:.4f}",
        "",
        f"Mann-Whitney U (SL < GIM): U={u_stat:.1f}, p={p_value:.4f}",
        f"Effect size: {effect_size:.3f}",
        f"95% CI on diff: [{ci_lo:.4f}, {ci_hi:.4f}]",
        "",
    ]
    if "Other KSK" in results:
        lines.append(f"Other KSK callers: {results['Other KSK']['combined_mean']:.4f} "
                     f"(n={len(results['Other KSK']['combined_losses'])})")
        lines.append("")

    if p_value < 0.05:
        lines.append("CONCLUSION: GIM has significantly higher reconstruction error.")
        lines.append("The SL-trained cycle breaks down on GIM, consistent with")
        lines.append("opposing dialects between these two callers.")
    else:
        lines.append("CONCLUSION: No significant difference detected.")
        lines.append("The cycle generalizes from SL to GIM. This may indicate")
        lines.append("shared acoustic structure or insufficient GIM sample size.")

    report = "\n".join(lines)
    (SANITY_DIR / "sl_vs_gim_report.txt").write_text(report)
    print(f"  Saved: {SANITY_DIR / 'sl_vs_gim_report.txt'}")

    print("\n" + report)
    print("\nDone!")


if __name__ == "__main__":
    main()


