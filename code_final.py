"""
Chimpanzee Pant-Hoot Dialect Analysis via Cycle-Consistent Embedding Mapping
=============================================================================

Uses AVEX (Earth Species Project) to extract bioacoustic embeddings from
chimpanzee pant-hoot vocalizations, then trains a cycle-consistent mapping
between pant (buildup) and hoot (climax) embeddings on Kasekela data.

Dialect hypothesis: the Kasekela-trained cycle mapping should reconstruct
Kasekela calls better than Mitumba (neighboring) and Kanyawara (distant),
with reconstruction error increasing with geographic/social distance.
"""

import os
import re
import json
import warnings
from pathlib import Path
from collections import defaultdict
import io
import contextlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import librosa
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from tqdm import tqdm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path("/Users/natalia_mac/Downloads/Chimpanzee Pant-Hoots")
EMBEDDING_CACHE = BASE_DIR / "embeddings_cache.npz"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

AVEX_MODEL_NAME = "esp_aves2_sl_beats_bio"  # BEATs fine-tuned on bioacoustics
SAMPLE_RATE = 16000
EMBEDDING_DIM = 768

# CycleGAN training hyperparameters
HIDDEN_DIM = 512
N_HIDDEN_LAYERS = 4
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 1e-6
N_EPOCHS = 1000
CYCLE_WEIGHT = 20.0       # weight for cycle-consistency loss (doubled)
IDENTITY_WEIGHT = 2.0     # weight for identity loss (reduced to allow tighter mapping)
ADVERSARIAL_WEIGHT = 1.0  # weight for adversarial loss
K_FOLDS = 5
BATCH_SIZE = 8            # smaller batches for better gradient signal
PATIENCE = 100            # more patience for slower LR

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Community ordering (by expected dialect distance from Kasekela)
COMMUNITIES = {"KSK": "Kasekela", "MTB": "Mitumba", "KAN": "Kanyawara"}


# ---------------------------------------------------------------------------
# Phase 0: Data Inventory
# ---------------------------------------------------------------------------
def inventory_files():
    """Parse all .wav files and return structured metadata."""
    records = []
    dirs = {
        ("KSK", "pant"): BASE_DIR / "KSK buildups",
        ("KSK", "hoot"): BASE_DIR / "KSK climaxes",
        ("MTB", "pant"): BASE_DIR / "MTB buildups",
        ("MTB", "hoot"): BASE_DIR / "MTB climaxes",
        ("KAN", "pant"): BASE_DIR / "KAN buildups",
        ("KAN", "hoot"): BASE_DIR / "KAN climaxes",
    }
    for (community, vtype), dirpath in dirs.items():
        if not dirpath.exists():
            print(f"  WARNING: {dirpath} not found")
            continue
        wavs = sorted(dirpath.glob("*.wav"))
        for wav in wavs:
            records.append({
                "path": str(wav),
                "community": community,
                "vtype": vtype,  # 'pant' or 'hoot'
                "filename": wav.stem,
            })
    print(f"Inventoried {len(records)} .wav files:")
    for comm in ["KSK", "MTB", "KAN"]:
        pants = sum(1 for r in records if r["community"] == comm and r["vtype"] == "pant")
        hoots = sum(1 for r in records if r["community"] == comm and r["vtype"] == "hoot")
        print(f"  {comm} ({COMMUNITIES[comm]}): {pants} pants, {hoots} hoots")
    return records


# ---------------------------------------------------------------------------
# Phase 1: Embedding Extraction
# ---------------------------------------------------------------------------
def extract_embeddings(records, force=False):
    """Extract AVEX embeddings for all .wav files, with caching."""
    if EMBEDDING_CACHE.exists() and not force:
        print(f"Loading cached embeddings from {EMBEDDING_CACHE}")
        data = np.load(EMBEDDING_CACHE, allow_pickle=True)
        embeddings = data["embeddings"]
        metadata = json.loads(str(data["metadata"]))
        print(f"  Loaded {len(embeddings)} embeddings of dim {embeddings.shape[1]}")
        return embeddings, metadata

    print(f"Extracting embeddings using AVEX model: {AVEX_MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    from avex import load_model

    model = load_model(AVEX_MODEL_NAME, return_features_only=True, device=DEVICE)
    model.eval()

    embeddings = []
    metadata = []

    for rec in tqdm(records, desc="Extracting embeddings"):
        audio, sr = librosa.load(rec["path"], sr=SAMPLE_RATE)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            emb = model(audio_tensor)  # (1, T, 768)
            pooled = emb.mean(dim=1).squeeze(0).cpu().numpy()  # (768,)

        embeddings.append(pooled)
        metadata.append(rec)

    embeddings = np.array(embeddings)
    print(f"  Extracted {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

    # Cache to disk
    np.savez(EMBEDDING_CACHE,
             embeddings=embeddings,
             metadata=json.dumps(metadata))
    print(f"  Cached to {EMBEDDING_CACHE}")

    return embeddings, metadata


# ---------------------------------------------------------------------------
# Phase 2: CycleGAN Architecture
# ---------------------------------------------------------------------------
class Generator(nn.Module):
    """MLP generator for mapping between embedding domains."""
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
                nn.Dropout(0.05),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        # Residual connection: output = input + learned_transform
        self.residual = (input_dim == output_dim)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            return x + out
        return out


class Discriminator(nn.Module):
    """MLP discriminator to distinguish real vs. generated embeddings."""
    def __init__(self, input_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class CycleGAN:
    """Cycle-consistent GAN in embedding space."""

    def __init__(self, device=DEVICE):
        self.device = device

        # Generators
        self.G_p2h = Generator().to(device)  # pant -> hoot
        self.G_h2p = Generator().to(device)  # hoot -> pant

        # Discriminators
        self.D_h = Discriminator().to(device)  # discriminate real/fake hoots
        self.D_p = Discriminator().to(device)  # discriminate real/fake pants

        # Optimizers
        self.opt_G = optim.AdamW(
            list(self.G_p2h.parameters()) + list(self.G_h2p.parameters()),
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.999)
        )
        self.opt_D = optim.AdamW(
            list(self.D_h.parameters()) + list(self.D_p.parameters()),
            
            lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.5, 0.999)
        )

        # Learning rate schedulers
        self.sched_G = optim.lr_scheduler.CosineAnnealingLR(self.opt_G, T_max=N_EPOCHS)
        self.sched_D = optim.lr_scheduler.CosineAnnealingLR(self.opt_D, T_max=N_EPOCHS)

        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def train_epoch(self, pant_loader, hoot_loader):
        """Train one epoch, returns dict of losses."""
        self.G_p2h.train(); self.G_h2p.train()
        self.D_h.train(); self.D_p.train()

        epoch_losses = defaultdict(float)
        n_batches = 0

        for (pants,), (hoots,) in zip(pant_loader, hoot_loader):
            pants = pants.to(self.device)
            hoots = hoots.to(self.device)

            # Match batch sizes (they may differ)
            min_bs = min(pants.size(0), hoots.size(0))
            pants, hoots = pants[:min_bs], hoots[:min_bs]

            # ----- Train Discriminators -----
            self.opt_D.zero_grad()

            fake_hoots = self.G_p2h(pants).detach()
            fake_pants = self.G_h2p(hoots).detach()

            # Real labels = 1, Fake labels = 0 (with label smoothing)
            real_label = torch.ones(min_bs, 1, device=self.device) * 0.9
            fake_label = torch.zeros(min_bs, 1, device=self.device) + 0.1

            loss_D_h = (self.mse(self.D_h(hoots), real_label) +
                        self.mse(self.D_h(fake_hoots), fake_label)) * 0.5
            loss_D_p = (self.mse(self.D_p(pants), real_label) +
                        self.mse(self.D_p(fake_pants), fake_label)) * 0.5

            loss_D = loss_D_h + loss_D_p
            loss_D.backward()
            self.opt_D.step()

            # ----- Train Generators -----
            self.opt_G.zero_grad()

            # Forward cycle: pant -> hoot -> pant
            fake_hoots = self.G_p2h(pants)
            cycle_pants = self.G_h2p(fake_hoots)

            # Backward cycle: hoot -> pant -> hoot
            fake_pants = self.G_h2p(hoots)
            cycle_hoots = self.G_p2h(fake_pants)

            # Adversarial losses (fool the discriminators)
            loss_adv = (self.mse(self.D_h(fake_hoots), real_label) +
                        self.mse(self.D_p(fake_pants), real_label)) * ADVERSARIAL_WEIGHT

            # Cycle consistency losses
            loss_cycle = (self.l1(cycle_pants, pants) +
                          self.l1(cycle_hoots, hoots)) * CYCLE_WEIGHT

            # Identity losses (regularization)
            loss_identity = (self.l1(self.G_h2p(pants), pants) +
                             self.l1(self.G_p2h(hoots), hoots)) * IDENTITY_WEIGHT

            loss_G = loss_adv + loss_cycle + loss_identity
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.G_p2h.parameters()) + list(self.G_h2p.parameters()),
                max_norm=1.0
            )
            self.opt_G.step()

            epoch_losses["D"] += loss_D.item()
            epoch_losses["G_adv"] += loss_adv.item()
            epoch_losses["G_cycle"] += loss_cycle.item()
            epoch_losses["G_identity"] += loss_identity.item()
            epoch_losses["G_total"] += loss_G.item()
            n_batches += 1

        self.sched_G.step()
        self.sched_D.step()

        return {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    @torch.no_grad()
    def compute_cycle_loss(self, embeddings, vtype):
        """Compute cycle reconstruction loss for a set of embeddings.

        Args:
            embeddings: (N, 768) numpy array
            vtype: 'pant' or 'hoot'

        Returns:
            per-sample L1 cycle losses (N,)
        """
        self.G_p2h.eval(); self.G_h2p.eval()
        emb = torch.tensor(embeddings, dtype=torch.float32).to(self.device)

        if vtype == "pant":
            # pant -> synthetic hoot -> synthetic pant
            synthetic_hoot = self.G_p2h(emb)
            reconstructed = self.G_h2p(synthetic_hoot)
        else:
            # hoot -> synthetic pant -> synthetic hoot
            synthetic_pant = self.G_h2p(emb)
            reconstructed = self.G_p2h(synthetic_pant)

        # Per-sample L1 loss
        losses = (emb - reconstructed).abs().mean(dim=1).cpu().numpy()
        return losses


# ---------------------------------------------------------------------------
# Phase 2 (cont): Training Pipeline
# ---------------------------------------------------------------------------
def get_community_embeddings(embeddings, metadata, community, vtype):
    """Filter embeddings by community and vocalization type."""
    mask = np.array([(m["community"] == community and m["vtype"] == vtype)
                     for m in metadata])
    return embeddings[mask]


def train_cyclegan_kfold(embeddings, metadata):
    """Train CycleGAN on Kasekela with k-fold cross-validation."""
    ksk_pants = get_community_embeddings(embeddings, metadata, "KSK", "pant")
    ksk_hoots = get_community_embeddings(embeddings, metadata, "KSK", "hoot")

    print(f"\nTraining CycleGAN on Kasekela data:")
    print(f"  Pants (buildups):  {len(ksk_pants)}")
    print(f"  Hoots (climaxes):  {len(ksk_hoots)}")
    print(f"  Device: {DEVICE}")
    print(f"  {K_FOLDS}-fold cross-validation")

    fold_results = []
    best_val_loss = float("inf")
    best_model_state = None

    kf_p = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    kf_h = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    pant_splits = list(kf_p.split(ksk_pants))
    hoot_splits = list(kf_h.split(ksk_hoots))

    for fold, ((p_train_idx, p_val_idx), (h_train_idx, h_val_idx)) in enumerate(
        zip(pant_splits, hoot_splits)
    ):
        print(f"\n  --- Fold {fold + 1}/{K_FOLDS} ---")

        # Split and scale
        p_train = ksk_pants[p_train_idx]
        p_val = ksk_pants[p_val_idx]
        h_train = ksk_hoots[h_train_idx]
        h_val = ksk_hoots[h_val_idx]

        scaler_p_fold = StandardScaler().fit(p_train)
        scaler_h_fold = StandardScaler().fit(h_train)

        p_train_s = scaler_p_fold.transform(p_train)
        p_val_s = scaler_p_fold.transform(p_val)
        h_train_s = scaler_h_fold.transform(h_train)
        h_val_s = scaler_h_fold.transform(h_val)

        # DataLoaders
        pant_ds = TensorDataset(torch.tensor(p_train_s, dtype=torch.float32))
        hoot_ds = TensorDataset(torch.tensor(h_train_s, dtype=torch.float32))
        pant_loader = DataLoader(pant_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        hoot_loader = DataLoader(hoot_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        # Initialize model
        cyclegan = CycleGAN(device=DEVICE)

        best_fold_loss = float("inf")
        patience_counter = 0

        for epoch in range(N_EPOCHS):
            losses = cyclegan.train_epoch(pant_loader, hoot_loader)

            # Validation: compute cycle loss on held-out data
            val_pant_loss = cyclegan.compute_cycle_loss(p_val_s, "pant").mean()
            val_hoot_loss = cyclegan.compute_cycle_loss(h_val_s, "hoot").mean()
            val_loss = (val_pant_loss + val_hoot_loss) / 2

            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1:4d} | G_cycle: {losses['G_cycle']:.4f} | "
                      f"Val cycle: {val_loss:.4f}")

            if val_loss < best_fold_loss:
                best_fold_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

        fold_results.append({
            "fold": fold,
            "best_val_loss": best_fold_loss,
            "epochs_trained": epoch + 1,
        })
        print(f"    Best val cycle loss: {best_fold_loss:.4f}")

    print(f"\n  Mean CV val loss: {np.mean([f['best_val_loss'] for f in fold_results]):.4f}")

    # Retrain final model on ALL Kasekela data
    print("\n  Retraining final model on all Kasekela data...")
    scaler_p_final = StandardScaler().fit(ksk_pants)
    scaler_h_final = StandardScaler().fit(ksk_hoots)

    p_all_s = scaler_p_final.transform(ksk_pants)
    h_all_s = scaler_h_final.transform(ksk_hoots)

    pant_ds = TensorDataset(torch.tensor(p_all_s, dtype=torch.float32))
    hoot_ds = TensorDataset(torch.tensor(h_all_s, dtype=torch.float32))
    pant_loader = DataLoader(pant_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    hoot_loader = DataLoader(hoot_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    final_model = CycleGAN(device=DEVICE)

    for epoch in range(N_EPOCHS):
        losses = final_model.train_epoch(pant_loader, hoot_loader)
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1:4d} | G_cycle: {losses['G_cycle']:.4f}")

    return final_model, scaler_p_final, scaler_h_final, fold_results


# ---------------------------------------------------------------------------
# Phase 3: Cross-Community Dialect Test
# ---------------------------------------------------------------------------
def dialect_test(model, scaler_p, scaler_h, embeddings, metadata):
    """Apply Kasekela-trained model to all communities and measure cycle loss."""
    results = {}

    for community in ["KSK", "MTB", "KAN"]:
        pants = get_community_embeddings(embeddings, metadata, community, "pant")
        hoots = get_community_embeddings(embeddings, metadata, community, "hoot")

        # Scale using Kasekela-fitted scalers
        pants_s = scaler_p.transform(pants)
        hoots_s = scaler_h.transform(hoots)

        pant_cycle_losses = model.compute_cycle_loss(pants_s, "pant")
        hoot_cycle_losses = model.compute_cycle_loss(hoots_s, "hoot")

        results[community] = {
            "pant_cycle_losses": pant_cycle_losses,
            "hoot_cycle_losses": hoot_cycle_losses,
            "pant_mean": pant_cycle_losses.mean(),
            "pant_std": pant_cycle_losses.std(),
            "hoot_mean": hoot_cycle_losses.mean(),
            "hoot_std": hoot_cycle_losses.std(),
            "combined_mean": np.concatenate([pant_cycle_losses, hoot_cycle_losses]).mean(),
            "n_pants": len(pants),
            "n_hoots": len(hoots),
        }

    return results


# ---------------------------------------------------------------------------
# Phase 4: Statistical Validation
# ---------------------------------------------------------------------------
def statistical_tests(results):
    """Run permutation tests and bootstrap CIs."""
    stats_results = {}

    for vtype in ["pant", "hoot"]:
        key = f"{vtype}_cycle_losses"

        for comm_a, comm_b in [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]:
            losses_a = results[comm_a][key]
            losses_b = results[comm_b][key]

            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(losses_a, losses_b, alternative="two-sided")

            # Effect size (rank-biserial correlation)
            n1, n2 = len(losses_a), len(losses_b)
            effect_size = 1 - (2 * u_stat) / (n1 * n2)

            # Bootstrap CI for difference in means
            n_bootstrap = 10000
            diffs = []
            rng = np.random.default_rng(42)
            for _ in range(n_bootstrap):
                boot_a = rng.choice(losses_a, size=len(losses_a), replace=True)
                boot_b = rng.choice(losses_b, size=len(losses_b), replace=True)
                diffs.append(boot_b.mean() - boot_a.mean())
            diffs = np.array(diffs)
            ci_lower, ci_upper = np.percentile(diffs, [2.5, 97.5])

            test_key = f"{vtype}_{comm_a}_vs_{comm_b}"
            stats_results[test_key] = {
                "U_statistic": u_stat,
                "p_value": p_value,
                "effect_size": effect_size,
                "mean_diff": losses_b.mean() - losses_a.mean(),
                "ci_95": (ci_lower, ci_upper),
            }

    # Permutation test for the ordering hypothesis: KSK < MTB < KAN
    for vtype in ["pant", "hoot"]:
        key = f"{vtype}_cycle_losses"
        ksk = results["KSK"][key].mean()
        mtb = results["MTB"][key].mean()
        kan = results["KAN"][key].mean()

        observed = (mtb - ksk) + (kan - mtb)

        all_losses = np.concatenate([
            results["KSK"][key], results["MTB"][key], results["KAN"][key]
        ])
        n_ksk = len(results["KSK"][key])
        n_mtb = len(results["MTB"][key])

        n_perm = 10000
        perm_stats = []
        rng = np.random.default_rng(42)
        for _ in range(n_perm):
            shuffled = rng.permutation(all_losses)
            perm_ksk = shuffled[:n_ksk].mean()
            perm_mtb = shuffled[n_ksk:n_ksk + n_mtb].mean()
            perm_kan = shuffled[n_ksk + n_mtb:].mean()
            perm_stats.append((perm_mtb - perm_ksk) + (perm_kan - perm_mtb))
        perm_stats = np.array(perm_stats)
        p_ordering = (perm_stats >= observed).mean()

        stats_results[f"{vtype}_ordering_KSK<MTB<KAN"] = {
            "observed_spread": observed,
            "p_value": p_ordering,
            "means": {"KSK": ksk, "MTB": mtb, "KAN": kan},
        }

    return stats_results


# ---------------------------------------------------------------------------
# Phase 4 (cont): Visualization
# ---------------------------------------------------------------------------
def plot_results(results, stats_results):
    """Generate all plots."""

    # 1. Boxplot of cycle reconstruction errors by community
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (vtype, title) in enumerate([
        ("pant", "Pant Cycle Loss\n(pant -> synth. hoot -> synth. pant)"),
        ("hoot", "Hoot Cycle Loss\n(hoot -> synth. pant -> synth. hoot)"),
        ("combined", "Combined Cycle Loss"),
    ]):
        ax = axes[idx]
        data_to_plot = []
        labels = []
        colors = ["#2ecc71", "#f39c12", "#e74c3c"]

        for comm, color in zip(["KSK", "MTB", "KAN"], colors):
            if vtype == "combined":
                losses = np.concatenate([
                    results[comm]["pant_cycle_losses"],
                    results[comm]["hoot_cycle_losses"],
                ])
            else:
                losses = results[comm][f"{vtype}_cycle_losses"]
            data_to_plot.append(losses)
            labels.append(f"{comm}\n({COMMUNITIES[comm]})\nn={len(losses)}")

        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        widths=0.6, showfliers=True,
                        flierprops=dict(marker="o", markersize=3, alpha=0.5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(title, fontsize=11)
        ax.set_ylabel("L1 Reconstruction Error")
        ax.grid(axis="y", alpha=0.3)

        # Add mean markers
        means = [d.mean() for d in data_to_plot]
        ax.scatter(range(1, 4), means, color="black", marker="D", s=40, zorder=3, label="Mean")

    plt.suptitle("Dialect Analysis: Cycle Reconstruction Error by Community\n"
                 "(Kasekela-trained mapping applied to all communities)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cycle_loss_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'cycle_loss_boxplots.png'}")

    # 2. Bar chart comparing pant vs hoot dialect signal
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    width = 0.35

    pant_means = [results[c]["pant_mean"] for c in ["KSK", "MTB", "KAN"]]
    hoot_means = [results[c]["hoot_mean"] for c in ["KSK", "MTB", "KAN"]]
    pant_stds = [results[c]["pant_std"] for c in ["KSK", "MTB", "KAN"]]
    hoot_stds = [results[c]["hoot_std"] for c in ["KSK", "MTB", "KAN"]]

    ax.bar(x - width/2, pant_means, width, yerr=pant_stds, label="Pant cycle",
           color="#3498db", alpha=0.7, capsize=4)
    ax.bar(x + width/2, hoot_means, width, yerr=hoot_stds, label="Hoot cycle",
           color="#e74c3c", alpha=0.7, capsize=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{c}\n({COMMUNITIES[c]})" for c in ["KSK", "MTB", "KAN"]])
    ax.set_ylabel("Mean L1 Reconstruction Error")
    ax.set_title("Pant vs. Hoot Cycle Loss by Community\n"
                 "(Which subtype carries more dialect signal?)",
                 fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pant_vs_hoot_dialect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'pant_vs_hoot_dialect.png'}")


def plot_umap(embeddings, metadata):
    """UMAP visualization of embedding space colored by community and type."""
    import umap

    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    emb_2d = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Color by community
    ax = axes[0]
    comm_colors = {"KSK": "#2ecc71", "MTB": "#f39c12", "KAN": "#e74c3c"}
    for comm, color in comm_colors.items():
        mask = np.array([m["community"] == comm for m in metadata])
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=COMMUNITIES[comm],
                   alpha=0.6, s=20)
    ax.set_title("UMAP by Community", fontweight="bold")
    ax.legend()
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    # Color by vocalization type
    ax = axes[1]
    vtype_colors = {"pant": "#3498db", "hoot": "#e74c3c"}
    for vtype, color in vtype_colors.items():
        mask = np.array([m["vtype"] == vtype for m in metadata])
        label = "Pant (buildup)" if vtype == "pant" else "Hoot (climax)"
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color, label=label,
                   alpha=0.6, s=20)
    ax.set_title("UMAP by Vocalization Type", fontweight="bold")
    ax.legend()
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")

    plt.suptitle("AVEX Embedding Space (UMAP Projection)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "umap_embeddings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'umap_embeddings.png'}")


def print_report(results, stats_results):
    """Print and save a comprehensive results report."""
    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    p("=" * 70)
    p("CHIMPANZEE PANT-HOOT DIALECT ANALYSIS RESULTS")
    p("=" * 70)
    p(f"Model: AVEX ({AVEX_MODEL_NAME})")
    p(f"Method: Cycle-consistent embedding mapping (CycleGAN in embedding space)")
    p(f"Training community: Kasekela (KSK)")
    p(f"Test communities: Mitumba (MTB, neighboring), Kanyawara (KAN, distant)")
    p()

    p("--- Cycle Reconstruction Error (Mean +/- SD) ---")
    p()
    p(f"{'Community':<20} {'Pant Cycle':<25} {'Hoot Cycle':<25} {'Combined':<20}")
    p("-" * 70)
    for comm in ["KSK", "MTB", "KAN"]:
        r = results[comm]
        combined = np.concatenate([r["pant_cycle_losses"], r["hoot_cycle_losses"]])
        p(f"{comm} ({COMMUNITIES[comm]:<12}) "
          f"{r['pant_mean']:.4f} +/- {r['pant_std']:.4f}     "
          f"{r['hoot_mean']:.4f} +/- {r['hoot_std']:.4f}     "
          f"{combined.mean():.4f} +/- {combined.std():.4f}")

    p()
    p("--- Hypothesis: Dialect Distance Ordering (KSK < MTB < KAN) ---")
    p()
    for vtype in ["pant", "hoot"]:
        key = f"{vtype}_ordering_KSK<MTB<KAN"
        r = stats_results[key]
        ordering_holds = (r["means"]["KSK"] < r["means"]["MTB"] < r["means"]["KAN"])
        status = "SUPPORTED" if ordering_holds else "NOT SUPPORTED"
        p(f"  {vtype.upper()} cycle: KSK={r['means']['KSK']:.4f}, "
          f"MTB={r['means']['MTB']:.4f}, KAN={r['means']['KAN']:.4f}")
        p(f"    Ordering KSK < MTB < KAN: {status}")
        p(f"    Permutation test p-value: {r['p_value']:.4f}")

    p()
    p("--- Pairwise Statistical Tests ---")
    p()
    for test_key, r in sorted(stats_results.items()):
        if "ordering" in test_key:
            continue
        p(f"  {test_key}:")
        p(f"    Mann-Whitney U = {r['U_statistic']:.1f}, p = {r['p_value']:.4f}")
        p(f"    Effect size (rank-biserial) = {r['effect_size']:.3f}")
        p(f"    Mean diff = {r['mean_diff']:.4f}, "
          f"95% CI = [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]")

    p()
    p("--- Which Vocalization Type Carries More Dialect Information? ---")
    p()
    pant_spread = results["KAN"]["pant_mean"] - results["KSK"]["pant_mean"]
    hoot_spread = results["KAN"]["hoot_mean"] - results["KSK"]["hoot_mean"]
    if pant_spread > hoot_spread:
        winner = "PANTS (buildups)"
    elif hoot_spread > pant_spread:
        winner = "HOOTS (climaxes)"
    else:
        winner = "EQUAL"
    p(f"  Pant cycle spread (KAN - KSK): {pant_spread:.4f}")
    p(f"  Hoot cycle spread (KAN - KSK): {hoot_spread:.4f}")
    p(f"  -> {winner} show more dialect differentiation")
    p()
    p("=" * 70)

    # Save report
    report_path = RESULTS_DIR / "dialect_report.txt"
    report_path.write_text("\n".join(lines))
    print(f"Report saved to {report_path}")


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("CHIMPANZEE PANT-HOOT DIALECT ANALYSIS")
    print("Cycle-Consistent Embedding Mapping with AVEX")
    print("=" * 70)

    # Phase 0: Inventory
    print("\n[Phase 0] Data Inventory")
    records = inventory_files()

    # Phase 1: Extract embeddings
    print("\n[Phase 1] Embedding Extraction")
    embeddings, metadata = extract_embeddings(records)

    # UMAP visualization of raw embeddings
    print("\n[Phase 1b] Embedding Visualization")
    plot_umap(embeddings, metadata)

    # Phase 2: Train CycleGAN on Kasekela
    print("\n[Phase 2] CycleGAN Training (Kasekela)")
    model, scaler_p, scaler_h, fold_results = train_cyclegan_kfold(embeddings, metadata)

    # Phase 3: Cross-community dialect test
    print("\n[Phase 3] Cross-Community Dialect Test")
    results = dialect_test(model, scaler_p, scaler_h, embeddings, metadata)

    # Phase 4: Statistical validation
    print("\n[Phase 4] Statistical Validation")
    stats_results = statistical_tests(results)

    # Visualization and report
    print("\n[Phase 4b] Generating Plots")
    plot_results(results, stats_results)
    print_report(results, stats_results)

    # Save raw results
    np.savez(RESULTS_DIR / "raw_results.npz",
             **{f"{c}_{v}": results[c][f"{v}_cycle_losses"]
                for c in ["KSK", "MTB", "KAN"] for v in ["pant", "hoot"]})
    print(f"\nRaw results saved to {RESULTS_DIR / 'raw_results.npz'}")
    print("\nDone!")


if __name__ == "__main__":
    main()

