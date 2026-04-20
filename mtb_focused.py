"""
Focused analysis: MTB-trained CycleGAN applied to KSK and KAN.
Hypothesis: MTB <-> KSK should have lower cycle loss than MTB <-> KAN
because Mitumba and Kasekela are neighboring communities.
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/Users/natalia_mac/Downloads/Chimpanzee Pant-Hoots")
EMBEDDING_CACHE = BASE_DIR / "embeddings_cache.npz"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EMBEDDING_DIM = 768
HIDDEN_DIM = 256
N_HIDDEN_LAYERS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
N_EPOCHS = 500
CYCLE_WEIGHT = 10.0
IDENTITY_WEIGHT = 5.0
ADVERSARIAL_WEIGHT = 1.0
BATCH_SIZE = 16
N_RUNS = 10               # multiple random restarts for robustness

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COMMUNITIES = {"KSK": "Kasekela", "MTB": "Mitumba", "KAN": "Kanyawara"}


# ---------------------------------------------------------------------------
# Model Architecture (same as main code)
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, output_dim=EMBEDDING_DIM,
                 hidden_dim=HIDDEN_DIM, n_layers=N_HIDDEN_LAYERS):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        self.residual = (input_dim == output_dim)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            return x + out
        return out


class Discriminator(nn.Module):
    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class CycleGAN:
    def __init__(self, device=DEVICE):
        self.device = device
        self.G_p2h = Generator().to(device)
        self.G_h2p = Generator().to(device)
        self.D_h = Discriminator().to(device)
        self.D_p = Discriminator().to(device)

        self.opt_G = optim.AdamW(
            list(self.G_p2h.parameters()) + list(self.G_h2p.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.999)
        )
        self.opt_D = optim.AdamW(
            list(self.D_h.parameters()) + list(self.D_p.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.999)
        )
        self.sched_G = optim.lr_scheduler.CosineAnnealingLR(self.opt_G, T_max=N_EPOCHS)
        self.sched_D = optim.lr_scheduler.CosineAnnealingLR(self.opt_D, T_max=N_EPOCHS)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def train_epoch(self, pant_loader, hoot_loader):
        self.G_p2h.train(); self.G_h2p.train()
        self.D_h.train(); self.D_p.train()

        epoch_losses = defaultdict(float)
        n_batches = 0

        for (pants,), (hoots,) in zip(pant_loader, hoot_loader):
            pants = pants.to(self.device)
            hoots = hoots.to(self.device)
            min_bs = min(pants.size(0), hoots.size(0))
            pants, hoots = pants[:min_bs], hoots[:min_bs]

            self.opt_D.zero_grad()
            fake_hoots = self.G_p2h(pants).detach()
            fake_pants = self.G_h2p(hoots).detach()
            real_label = torch.ones(min_bs, 1, device=self.device) * 0.9
            fake_label = torch.zeros(min_bs, 1, device=self.device) + 0.1
            loss_D = ((self.mse(self.D_h(hoots), real_label) +
                       self.mse(self.D_h(fake_hoots), fake_label)) * 0.5 +
                      (self.mse(self.D_p(pants), real_label) +
                       self.mse(self.D_p(fake_pants), fake_label)) * 0.5)
            loss_D.backward()
            self.opt_D.step()

            self.opt_G.zero_grad()
            fake_hoots = self.G_p2h(pants)
            cycle_pants = self.G_h2p(fake_hoots)
            fake_pants = self.G_h2p(hoots)
            cycle_hoots = self.G_p2h(fake_pants)

            loss_adv = (self.mse(self.D_h(fake_hoots), real_label) +
                        self.mse(self.D_p(fake_pants), real_label)) * ADVERSARIAL_WEIGHT
            loss_cycle = (self.l1(cycle_pants, pants) +
                          self.l1(cycle_hoots, hoots)) * CYCLE_WEIGHT
            loss_identity = (self.l1(self.G_h2p(pants), pants) +
                             self.l1(self.G_p2h(hoots), hoots)) * IDENTITY_WEIGHT

            loss_G = loss_adv + loss_cycle + loss_identity
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.G_p2h.parameters()) + list(self.G_h2p.parameters()),
                max_norm=1.0
            )
            self.opt_G.step()
            epoch_losses["G_cycle"] += loss_cycle.item()
            n_batches += 1

        self.sched_G.step()
        self.sched_D.step()
        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def compute_cycle_loss(self, embeddings, vtype):
        self.G_p2h.eval(); self.G_h2p.eval()
        emb = torch.tensor(embeddings, dtype=torch.float32).to(self.device)
        if vtype == "pant":
            reconstructed = self.G_h2p(self.G_p2h(emb))
        else:
            reconstructed = self.G_p2h(self.G_h2p(emb))
        return (emb - reconstructed).abs().mean(dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FOCUSED ANALYSIS: MTB-Trained Model on KSK vs KAN")
    print("=" * 70)

    # Load cached embeddings
    data = np.load(EMBEDDING_CACHE, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata"]))

    def get_emb(community, vtype):
        mask = np.array([(m["community"] == community and m["vtype"] == vtype)
                         for m in metadata])
        return embeddings[mask]

    mtb_pants = get_emb("MTB", "pant")
    mtb_hoots = get_emb("MTB", "hoot")
    ksk_pants = get_emb("KSK", "pant")
    ksk_hoots = get_emb("KSK", "hoot")
    kan_pants = get_emb("KAN", "pant")
    kan_hoots = get_emb("KAN", "hoot")

    print(f"MTB: {len(mtb_pants)} pants, {len(mtb_hoots)} hoots")
    print(f"KSK: {len(ksk_pants)} pants, {len(ksk_hoots)} hoots")
    print(f"KAN: {len(kan_pants)} pants, {len(kan_hoots)} hoots")
    print(f"Device: {DEVICE}")
    print(f"Running {N_RUNS} independent models (different random seeds)")

    # Collect per-sample losses across multiple runs for robustness
    all_losses = {comm: {vtype: [] for vtype in ["pant", "hoot"]}
                  for comm in ["MTB", "KSK", "KAN"]}

    for run in range(N_RUNS):
        seed = run * 42 + 7
        torch.manual_seed(seed)
        np.random.seed(seed)

        print(f"\n  --- Run {run+1}/{N_RUNS} (seed={seed}) ---")

        # Fit scalers on MTB
        scaler_p = StandardScaler().fit(mtb_pants)
        scaler_h = StandardScaler().fit(mtb_hoots)

        mtb_p_s = scaler_p.transform(mtb_pants)
        mtb_h_s = scaler_h.transform(mtb_hoots)

        pant_ds = TensorDataset(torch.tensor(mtb_p_s, dtype=torch.float32))
        hoot_ds = TensorDataset(torch.tensor(mtb_h_s, dtype=torch.float32))
        pant_loader = DataLoader(pant_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        hoot_loader = DataLoader(hoot_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        model = CycleGAN(device=DEVICE)

        for epoch in range(N_EPOCHS):
            losses = model.train_epoch(pant_loader, hoot_loader)
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1:4d} | G_cycle: {losses['G_cycle']:.4f}")

        # Evaluate on all communities
        for comm, (pants, hoots) in [("MTB", (mtb_pants, mtb_hoots)),
                                      ("KSK", (ksk_pants, ksk_hoots)),
                                      ("KAN", (kan_pants, kan_hoots))]:
            p_s = scaler_p.transform(pants)
            h_s = scaler_h.transform(hoots)
            pant_losses = model.compute_cycle_loss(p_s, "pant")
            hoot_losses = model.compute_cycle_loss(h_s, "hoot")
            all_losses[comm]["pant"].append(pant_losses)
            all_losses[comm]["hoot"].append(hoot_losses)

        # Print this run's results
        for comm in ["MTB", "KSK", "KAN"]:
            p_mean = all_losses[comm]["pant"][-1].mean()
            h_mean = all_losses[comm]["hoot"][-1].mean()
            print(f"    {comm}: pant={p_mean:.4f}, hoot={h_mean:.4f}")

    # Average across runs: for each sample, take the mean loss across runs
    avg_losses = {}
    for comm in ["MTB", "KSK", "KAN"]:
        for vtype in ["pant", "hoot"]:
            stacked = np.stack(all_losses[comm][vtype], axis=0)  # (N_RUNS, n_samples)
            avg_losses[(comm, vtype)] = stacked.mean(axis=0)  # (n_samples,)

    # Combined losses per community
    combined = {}
    for comm in ["MTB", "KSK", "KAN"]:
        combined[comm] = np.concatenate([avg_losses[(comm, "pant")],
                                         avg_losses[(comm, "hoot")]])

    print("\n" + "=" * 70)
    print("RESULTS (averaged over {} runs)".format(N_RUNS))
    print("=" * 70)
    for comm in ["MTB", "KSK", "KAN"]:
        p = avg_losses[(comm, "pant")]
        h = avg_losses[(comm, "hoot")]
        c = combined[comm]
        print(f"  {comm} ({COMMUNITIES[comm]}): "
              f"pant={p.mean():.4f}+/-{p.std():.4f}, "
              f"hoot={h.mean():.4f}+/-{h.std():.4f}, "
              f"combined={c.mean():.4f}+/-{c.std():.4f}")

    # --- Statistical tests: KSK vs KAN under MTB model ---
    print("\n--- Statistical Tests: KSK vs KAN (under MTB model) ---\n")

    for vtype in ["pant", "hoot", "combined"]:
        if vtype == "combined":
            ksk_losses = combined["KSK"]
            kan_losses = combined["KAN"]
            mtb_losses = combined["MTB"]
        else:
            ksk_losses = avg_losses[("KSK", vtype)]
            kan_losses = avg_losses[("KAN", vtype)]
            mtb_losses = avg_losses[("MTB", vtype)]

        # MTB vs KSK
        u1, p1 = stats.mannwhitneyu(mtb_losses, ksk_losses, alternative="two-sided")
        n1, n2 = len(mtb_losses), len(ksk_losses)
        es1 = 1 - (2 * u1) / (n1 * n2)

        # MTB vs KAN
        u2, p2 = stats.mannwhitneyu(mtb_losses, kan_losses, alternative="two-sided")
        n3 = len(kan_losses)
        es2 = 1 - (2 * u2) / (n1 * n3)

        # KSK vs KAN (the key test)
        u3, p3 = stats.mannwhitneyu(ksk_losses, kan_losses, alternative="two-sided")
        es3 = 1 - (2 * u3) / (n2 * n3)

        # Bootstrap CI for KSK-KAN difference
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(10000):
            boot_ksk = rng.choice(ksk_losses, size=len(ksk_losses), replace=True)
            boot_kan = rng.choice(kan_losses, size=len(kan_losses), replace=True)
            diffs.append(boot_kan.mean() - boot_ksk.mean())
        diffs = np.array(diffs)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

        print(f"  {vtype.upper()}:")
        print(f"    MTB (in-group): {mtb_losses.mean():.4f} +/- {mtb_losses.std():.4f}")
        print(f"    KSK (neighbor): {ksk_losses.mean():.4f} +/- {ksk_losses.std():.4f}")
        print(f"    KAN (distant):  {kan_losses.mean():.4f} +/- {kan_losses.std():.4f}")
        print(f"    MTB vs KSK: U={u1:.0f}, p={p1:.4f}, effect={es1:.3f}")
        print(f"    MTB vs KAN: U={u2:.0f}, p={p2:.4f}, effect={es2:.3f}")
        print(f"    KSK vs KAN: U={u3:.0f}, p={p3:.4f}, effect={es3:.3f}, "
              f"diff={kan_losses.mean()-ksk_losses.mean():.4f}, "
              f"95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        ordering = mtb_losses.mean() < ksk_losses.mean() < kan_losses.mean()
        print(f"    Ordering MTB < KSK < KAN: {'YES' if ordering else 'NO'}")
        print()

    # --- Plots ---
    # 1. Main boxplot: MTB model cycle loss by community
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    colors = {"MTB": "#f39c12", "KSK": "#2ecc71", "KAN": "#e74c3c"}

    for idx, (vtype, title) in enumerate([
        ("pant", "Pant Cycle Loss\n(pant -> synth. hoot -> synth. pant)"),
        ("hoot", "Hoot Cycle Loss\n(hoot -> synth. pant -> synth. hoot)"),
        ("combined", "Combined Cycle Loss"),
    ]):
        ax = axes[idx]
        data_to_plot = []
        labels = []
        box_colors = []

        for comm in ["MTB", "KSK", "KAN"]:
            if vtype == "combined":
                losses = combined[comm]
            else:
                losses = avg_losses[(comm, vtype)]
            data_to_plot.append(losses)
            labels.append(f"{comm}\n({COMMUNITIES[comm]})\nn={len(losses)}")
            box_colors.append(colors[comm])

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        # Mean markers
        means = [d.mean() for d in data_to_plot]
        ax.scatter(range(1, 4), means, color="black", marker="D", s=50, zorder=3)

        # Significance brackets
        y_max = max(d.max() for d in data_to_plot)
        y_step = (y_max - min(d.min() for d in data_to_plot)) * 0.08

        # KSK vs KAN bracket
        if vtype == "combined":
            ksk_l, kan_l = combined["KSK"], combined["KAN"]
        else:
            ksk_l = avg_losses[("KSK", vtype)]
            kan_l = avg_losses[("KAN", vtype)]
        _, p_val = stats.mannwhitneyu(ksk_l, kan_l, alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        bracket_y = y_max + y_step
        ax.plot([2, 3], [bracket_y, bracket_y], "k-", linewidth=1.5)
        ax.plot([2, 2], [bracket_y - y_step*0.2, bracket_y], "k-", linewidth=1.5)
        ax.plot([3, 3], [bracket_y - y_step*0.2, bracket_y], "k-", linewidth=1.5)
        ax.text(2.5, bracket_y + y_step*0.1, f"{sig} (p={p_val:.4f})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("L1 Reconstruction Error")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("MTB-Trained CycleGAN: Cycle Loss by Community\n"
                 "Hypothesis: MTB < KSK (neighbor) < KAN (distant)\n"
                 f"(averaged over {N_RUNS} independent runs)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mtb_focused_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'mtb_focused_boxplots.png'}")

    # 2. Violin plot for finer distribution view
    fig, ax = plt.subplots(figsize=(10, 6))
    data_combined = [combined["MTB"], combined["KSK"], combined["KAN"]]
    parts = ax.violinplot(data_combined, positions=[1, 2, 3], showmeans=True,
                          showmedians=True, showextrema=False)

    for i, (pc, comm) in enumerate(zip(parts["bodies"], ["MTB", "KSK", "KAN"])):
        pc.set_facecolor(colors[comm])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"MTB\n(Mitumba)\nn={len(combined['MTB'])}",
                        f"KSK\n(Kasekela)\nn={len(combined['KSK'])}",
                        f"KAN\n(Kanyawara)\nn={len(combined['KAN'])}"])
    ax.set_ylabel("Combined L1 Reconstruction Error")
    ax.set_title("MTB-Trained CycleGAN: Distribution of Cycle Loss\n"
                 f"(averaged over {N_RUNS} runs)",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add significance
    _, p_ksk_kan = stats.mannwhitneyu(combined["KSK"], combined["KAN"], alternative="two-sided")
    sig = "***" if p_ksk_kan < 0.001 else "**" if p_ksk_kan < 0.01 else "*" if p_ksk_kan < 0.05 else "n.s."
    y_max = max(d.max() for d in data_combined)
    y_step = (y_max - min(d.min() for d in data_combined)) * 0.06
    bracket_y = y_max + y_step
    ax.plot([2, 3], [bracket_y, bracket_y], "k-", linewidth=1.5)
    ax.plot([2, 2], [bracket_y - y_step*0.3, bracket_y], "k-", linewidth=1.5)
    ax.plot([3, 3], [bracket_y - y_step*0.3, bracket_y], "k-", linewidth=1.5)
    ax.text(2.5, bracket_y + y_step*0.1, f"KSK vs KAN: {sig} (p={p_ksk_kan:.4f})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "mtb_focused_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'mtb_focused_violin.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
