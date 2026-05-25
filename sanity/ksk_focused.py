"""
Focused analysis: KSK-trained CycleGAN applied to MTB and KAN.
Hypothesis: KSK < MTB (neighbor) < KAN (distant)
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
N_RUNS = 10

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
COMMUNITIES = {"KSK": "Kasekela", "MTB": "Mitumba", "KAN": "Kanyawara"}


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
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.999))
        self.opt_D = optim.AdamW(
            list(self.D_h.parameters()) + list(self.D_p.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.999))
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
                list(self.G_p2h.parameters()) + list(self.G_h2p.parameters()), max_norm=1.0)
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


def main():
    print("=" * 70)
    print("FOCUSED ANALYSIS: KSK-Trained Model on MTB vs KAN")
    print("Hypothesis: KSK < MTB (neighbor) < KAN (distant)")
    print("=" * 70)

    data = np.load(EMBEDDING_CACHE, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata"]))

    def get_emb(community, vtype):
        mask = np.array([(m["community"] == community and m["vtype"] == vtype)
                         for m in metadata])
        return embeddings[mask]

    ksk_pants = get_emb("KSK", "pant")
    ksk_hoots = get_emb("KSK", "hoot")
    mtb_pants = get_emb("MTB", "pant")
    mtb_hoots = get_emb("MTB", "hoot")
    kan_pants = get_emb("KAN", "pant")
    kan_hoots = get_emb("KAN", "hoot")

    print(f"KSK (training): {len(ksk_pants)} pants, {len(ksk_hoots)} hoots")
    print(f"MTB (neighbor):  {len(mtb_pants)} pants, {len(mtb_hoots)} hoots")
    print(f"KAN (distant):   {len(kan_pants)} pants, {len(kan_hoots)} hoots")
    print(f"Device: {DEVICE}")
    print(f"Running {N_RUNS} independent models (different random seeds)")

    all_losses = {comm: {vtype: [] for vtype in ["pant", "hoot"]}
                  for comm in ["KSK", "MTB", "KAN"]}

    for run in range(N_RUNS):
        seed = run * 42 + 7
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n  --- Run {run+1}/{N_RUNS} (seed={seed}) ---")

        # Fit scalers on KSK
        scaler_p = StandardScaler().fit(ksk_pants)
        scaler_h = StandardScaler().fit(ksk_hoots)

        ksk_p_s = scaler_p.transform(ksk_pants)
        ksk_h_s = scaler_h.transform(ksk_hoots)

        pant_ds = TensorDataset(torch.tensor(ksk_p_s, dtype=torch.float32))
        hoot_ds = TensorDataset(torch.tensor(ksk_h_s, dtype=torch.float32))
        pant_loader = DataLoader(pant_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        hoot_loader = DataLoader(hoot_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        model = CycleGAN(device=DEVICE)
        for epoch in range(N_EPOCHS):
            losses = model.train_epoch(pant_loader, hoot_loader)
            if (epoch + 1) % 200 == 0:
                print(f"    Epoch {epoch+1:4d} | G_cycle: {losses['G_cycle']:.4f}")

        # Evaluate on all communities
        for comm, (pants, hoots) in [("KSK", (ksk_pants, ksk_hoots)),
                                      ("MTB", (mtb_pants, mtb_hoots)),
                                      ("KAN", (kan_pants, kan_hoots))]:
            p_s = scaler_p.transform(pants)
            h_s = scaler_h.transform(hoots)
            all_losses[comm]["pant"].append(model.compute_cycle_loss(p_s, "pant"))
            all_losses[comm]["hoot"].append(model.compute_cycle_loss(h_s, "hoot"))

        for comm in ["KSK", "MTB", "KAN"]:
            p_mean = all_losses[comm]["pant"][-1].mean()
            h_mean = all_losses[comm]["hoot"][-1].mean()
            print(f"    {comm}: pant={p_mean:.4f}, hoot={h_mean:.4f}")

    # Average across runs
    avg_losses = {}
    for comm in ["KSK", "MTB", "KAN"]:
        for vtype in ["pant", "hoot"]:
            stacked = np.stack(all_losses[comm][vtype], axis=0)
            avg_losses[(comm, vtype)] = stacked.mean(axis=0)

    combined = {}
    for comm in ["KSK", "MTB", "KAN"]:
        combined[comm] = np.concatenate([avg_losses[(comm, "pant")],
                                         avg_losses[(comm, "hoot")]])

    print("\n" + "=" * 70)
    print(f"RESULTS (averaged over {N_RUNS} runs)")
    print("=" * 70)
    for comm in ["KSK", "MTB", "KAN"]:
        p = avg_losses[(comm, "pant")]
        h = avg_losses[(comm, "hoot")]
        c = combined[comm]
        print(f"  {comm} ({COMMUNITIES[comm]}): "
              f"pant={p.mean():.4f}+/-{p.std():.4f}, "
              f"hoot={h.mean():.4f}+/-{h.std():.4f}, "
              f"combined={c.mean():.4f}+/-{c.std():.4f}")

    # --- Statistical tests ---
    print("\n--- Statistical Tests: MTB vs KAN (under KSK model) ---\n")

    for vtype in ["pant", "hoot", "combined"]:
        if vtype == "combined":
            ksk_l = combined["KSK"]
            mtb_l = combined["MTB"]
            kan_l = combined["KAN"]
        else:
            ksk_l = avg_losses[("KSK", vtype)]
            mtb_l = avg_losses[("MTB", vtype)]
            kan_l = avg_losses[("KAN", vtype)]

        # KSK vs MTB
        u1, p1 = stats.mannwhitneyu(ksk_l, mtb_l, alternative="two-sided")
        es1 = 1 - (2 * u1) / (len(ksk_l) * len(mtb_l))

        # KSK vs KAN
        u2, p2 = stats.mannwhitneyu(ksk_l, kan_l, alternative="two-sided")
        es2 = 1 - (2 * u2) / (len(ksk_l) * len(kan_l))

        # MTB vs KAN (the key test)
        u3, p3 = stats.mannwhitneyu(mtb_l, kan_l, alternative="two-sided")
        es3 = 1 - (2 * u3) / (len(mtb_l) * len(kan_l))

        # Bootstrap CI for MTB-KAN difference
        rng = np.random.default_rng(42)
        diffs = []
        for _ in range(10000):
            boot_mtb = rng.choice(mtb_l, size=len(mtb_l), replace=True)
            boot_kan = rng.choice(kan_l, size=len(kan_l), replace=True)
            diffs.append(boot_kan.mean() - boot_mtb.mean())
        diffs = np.array(diffs)
        ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])

        print(f"  {vtype.upper()}:")
        print(f"    KSK (in-group):  {ksk_l.mean():.4f} +/- {ksk_l.std():.4f}")
        print(f"    MTB (neighbor):  {mtb_l.mean():.4f} +/- {mtb_l.std():.4f}")
        print(f"    KAN (distant):   {kan_l.mean():.4f} +/- {kan_l.std():.4f}")
        print(f"    KSK vs MTB: U={u1:.0f}, p={p1:.4f}, effect={es1:.3f}")
        print(f"    KSK vs KAN: U={u2:.0f}, p={p2:.4f}, effect={es2:.3f}")
        print(f"    MTB vs KAN: U={u3:.0f}, p={p3:.4f}, effect={es3:.3f}, "
              f"diff={kan_l.mean()-mtb_l.mean():.4f}, "
              f"95%CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
        ordering = ksk_l.mean() < mtb_l.mean() < kan_l.mean()
        print(f"    Ordering KSK < MTB < KAN: {'YES' if ordering else 'NO'}")
        print()

    # --- Plots ---
    colors = {"KSK": "#2ecc71", "MTB": "#f39c12", "KAN": "#e74c3c"}

    # 1. Boxplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    for idx, (vtype, title) in enumerate([
        ("pant", "Pant Cycle Loss\n(pant -> synth. hoot -> synth. pant)"),
        ("hoot", "Hoot Cycle Loss\n(hoot -> synth. pant -> synth. hoot)"),
        ("combined", "Combined Cycle Loss"),
    ]):
        ax = axes[idx]
        data_to_plot = []
        labels = []
        box_colors = []
        for comm in ["KSK", "MTB", "KAN"]:
            losses = combined[comm] if vtype == "combined" else avg_losses[(comm, vtype)]
            data_to_plot.append(losses)
            labels.append(f"{comm}\n({COMMUNITIES[comm]})\nn={len(losses)}")
            box_colors.append(colors[comm])

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, c in zip(bp["boxes"], box_colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)

        means = [d.mean() for d in data_to_plot]
        ax.scatter(range(1, 4), means, color="black", marker="D", s=50, zorder=3)

        # Significance bracket: MTB vs KAN
        mtb_l = combined["MTB"] if vtype == "combined" else avg_losses[("MTB", vtype)]
        kan_l = combined["KAN"] if vtype == "combined" else avg_losses[("KAN", vtype)]
        _, p_val = stats.mannwhitneyu(mtb_l, kan_l, alternative="two-sided")
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

        y_max = max(d.max() for d in data_to_plot)
        y_step = (y_max - min(d.min() for d in data_to_plot)) * 0.08
        bracket_y = y_max + y_step
        ax.plot([2, 3], [bracket_y, bracket_y], "k-", linewidth=1.5)
        ax.plot([2, 2], [bracket_y - y_step*0.2, bracket_y], "k-", linewidth=1.5)
        ax.plot([3, 3], [bracket_y - y_step*0.2, bracket_y], "k-", linewidth=1.5)
        ax.text(2.5, bracket_y + y_step*0.1, f"MTB vs KAN: {sig} (p={p_val:.4f})",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("L1 Reconstruction Error")
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("KSK-Trained CycleGAN: Cycle Loss by Community\n"
                 "Hypothesis: KSK < MTB (neighbor) < KAN (distant)\n"
                 f"(averaged over {N_RUNS} independent runs)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ksk_focused_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {RESULTS_DIR / 'ksk_focused_boxplots.png'}")

    # 2. Violin plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_combined = [combined["KSK"], combined["MTB"], combined["KAN"]]
    parts = ax.violinplot(data_combined, positions=[1, 2, 3], showmeans=True,
                          showmedians=True, showextrema=False)
    for pc, comm in zip(parts["bodies"], ["KSK", "MTB", "KAN"]):
        pc.set_facecolor(colors[comm])
        pc.set_alpha(0.6)
    parts["cmeans"].set_color("black")
    parts["cmedians"].set_color("gray")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"KSK\n(Kasekela)\nn={len(combined['KSK'])}",
                        f"MTB\n(Mitumba)\nn={len(combined['MTB'])}",
                        f"KAN\n(Kanyawara)\nn={len(combined['KAN'])}"])
    ax.set_ylabel("Combined L1 Reconstruction Error")
    ax.set_title("KSK-Trained CycleGAN: Distribution of Cycle Loss\n"
                 f"(averaged over {N_RUNS} runs)", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    _, p_mtb_kan = stats.mannwhitneyu(combined["MTB"], combined["KAN"], alternative="two-sided")
    sig = "***" if p_mtb_kan < 0.001 else "**" if p_mtb_kan < 0.01 else "*" if p_mtb_kan < 0.05 else "n.s."
    y_max = max(d.max() for d in data_combined)
    y_step = (y_max - min(d.min() for d in data_combined)) * 0.06
    bracket_y = y_max + y_step
    ax.plot([2, 3], [bracket_y, bracket_y], "k-", linewidth=1.5)
    ax.plot([2, 2], [bracket_y - y_step*0.3, bracket_y], "k-", linewidth=1.5)
    ax.plot([3, 3], [bracket_y - y_step*0.3, bracket_y], "k-", linewidth=1.5)
    ax.text(2.5, bracket_y + y_step*0.1, f"MTB vs KAN: {sig} (p={p_mtb_kan:.4f})",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ksk_focused_violin.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {RESULTS_DIR / 'ksk_focused_violin.png'}")

    print("\nDone!")


if __name__ == "__main__":
    main()
