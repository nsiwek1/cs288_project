"""
Direct embedding space comparison across communities.
Multiple distance metrics to test whether MTB is closer to KSK than KAN.

Methods:
1. Centroid distance (Euclidean & cosine)
2. Maximum Mean Discrepancy (MMD) — distributional distance
3. Pairwise sample distances (all-pairs between communities)
4. Linear classifier separability (how hard is it to tell communities apart?)
5. PERMANOVA — multivariate permutation test on full embedding vectors
"""

import json
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")

BASE_DIR = Path("/Users/natalia_mac/Downloads/Chimpanzee Pant-Hoots")
EMBEDDING_CACHE = BASE_DIR / "embeddings_cache.npz"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

COMMUNITIES = {"KSK": "Kasekela", "MTB": "Mitumba", "KAN": "Kanyawara"}


def load_embeddings():
    data = np.load(EMBEDDING_CACHE, allow_pickle=True)
    embeddings = data["embeddings"]
    metadata = json.loads(str(data["metadata"]))
    return embeddings, metadata


def get_emb(embeddings, metadata, community, vtype=None):
    if vtype:
        mask = np.array([(m["community"] == community and m["vtype"] == vtype)
                         for m in metadata])
    else:
        mask = np.array([m["community"] == community for m in metadata])
    return embeddings[mask]


# ---------------------------------------------------------------------------
# Method 1: Centroid distances
# ---------------------------------------------------------------------------
def centroid_distances(embeddings, metadata):
    print("\n" + "=" * 70)
    print("METHOD 1: CENTROID DISTANCES")
    print("=" * 70)

    results = {}
    for vtype in ["pant", "hoot", None]:
        vtype_label = vtype if vtype else "all"
        centroids = {}
        for comm in ["KSK", "MTB", "KAN"]:
            emb = get_emb(embeddings, metadata, comm, vtype)
            centroids[comm] = emb.mean(axis=0)

        # Pairwise distances
        print(f"\n  {vtype_label.upper()} embeddings:")
        for metric in ["euclidean", "cosine"]:
            print(f"    {metric.capitalize()} distance:")
            for a, b in [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]:
                if metric == "euclidean":
                    d = np.linalg.norm(centroids[a] - centroids[b])
                else:
                    d = 1 - np.dot(centroids[a], centroids[b]) / (
                        np.linalg.norm(centroids[a]) * np.linalg.norm(centroids[b]))
                print(f"      {a} <-> {b}: {d:.6f}")
                results[(vtype_label, metric, a, b)] = d

        # Bootstrap CI for the difference (KSK-MTB) vs (KSK-KAN)
        print(f"\n    Bootstrap test: is KSK-MTB < KSK-KAN?")
        for metric in ["euclidean", "cosine"]:
            rng = np.random.default_rng(42)
            ksk_emb = get_emb(embeddings, metadata, "KSK", vtype)
            mtb_emb = get_emb(embeddings, metadata, "MTB", vtype)
            kan_emb = get_emb(embeddings, metadata, "KAN", vtype)

            diffs = []
            for _ in range(10000):
                ksk_boot = ksk_emb[rng.choice(len(ksk_emb), len(ksk_emb), replace=True)]
                mtb_boot = mtb_emb[rng.choice(len(mtb_emb), len(mtb_emb), replace=True)]
                kan_boot = kan_emb[rng.choice(len(kan_emb), len(kan_emb), replace=True)]

                c_ksk = ksk_boot.mean(axis=0)
                c_mtb = mtb_boot.mean(axis=0)
                c_kan = kan_boot.mean(axis=0)

                if metric == "euclidean":
                    d_ksk_mtb = np.linalg.norm(c_ksk - c_mtb)
                    d_ksk_kan = np.linalg.norm(c_ksk - c_kan)
                else:
                    d_ksk_mtb = 1 - np.dot(c_ksk, c_mtb) / (np.linalg.norm(c_ksk) * np.linalg.norm(c_mtb))
                    d_ksk_kan = 1 - np.dot(c_ksk, c_kan) / (np.linalg.norm(c_ksk) * np.linalg.norm(c_kan))
                diffs.append(d_ksk_kan - d_ksk_mtb)

            diffs = np.array(diffs)
            p_closer = (diffs <= 0).mean()  # proportion where KAN is NOT farther
            ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
            print(f"      {metric}: KSK-KAN minus KSK-MTB = {np.mean(diffs):.6f}, "
                  f"95%CI=[{ci_lo:.6f}, {ci_hi:.6f}], p(KAN not farther)={p_closer:.4f}")

    return results


# ---------------------------------------------------------------------------
# Method 2: Maximum Mean Discrepancy (MMD)
# ---------------------------------------------------------------------------
def compute_mmd(X, Y, gamma=None):
    """Compute MMD^2 with RBF kernel."""
    if gamma is None:
        # Median heuristic
        all_data = np.vstack([X, Y])
        dists = pdist(all_data, metric="euclidean")
        gamma = 1.0 / (2 * np.median(dists) ** 2)

    XX = np.exp(-gamma * cdist(X, X, metric="sqeuclidean"))
    YY = np.exp(-gamma * cdist(Y, Y, metric="sqeuclidean"))
    XY = np.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))

    n, m = len(X), len(Y)
    mmd2 = (XX.sum() / (n * n) - 2 * XY.sum() / (n * m) + YY.sum() / (m * m))
    return mmd2


def mmd_analysis(embeddings, metadata):
    print("\n" + "=" * 70)
    print("METHOD 2: MAXIMUM MEAN DISCREPANCY (MMD)")
    print("=" * 70)

    for vtype in ["pant", "hoot", None]:
        vtype_label = vtype if vtype else "all"
        comms_emb = {}
        for comm in ["KSK", "MTB", "KAN"]:
            comms_emb[comm] = get_emb(embeddings, metadata, comm, vtype)

        # Compute gamma from all data
        all_data = np.vstack(list(comms_emb.values()))
        dists = pdist(all_data, metric="euclidean")
        gamma = 1.0 / (2 * np.median(dists) ** 2)

        print(f"\n  {vtype_label.upper()} embeddings (RBF kernel, median heuristic gamma={gamma:.4f}):")

        mmd_values = {}
        for a, b in [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]:
            mmd2 = compute_mmd(comms_emb[a], comms_emb[b], gamma)
            mmd_values[(a, b)] = mmd2
            print(f"    {a} <-> {b}: MMD^2 = {mmd2:.6f}")

        # Permutation test for MMD
        print(f"\n    Permutation test (10000 permutations):")
        for a, b in [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]:
            observed = mmd_values[(a, b)]
            combined = np.vstack([comms_emb[a], comms_emb[b]])
            n_a = len(comms_emb[a])
            rng = np.random.default_rng(42)
            count = 0
            for _ in range(10000):
                perm = rng.permutation(len(combined))
                perm_a = combined[perm[:n_a]]
                perm_b = combined[perm[n_a:]]
                perm_mmd = compute_mmd(perm_a, perm_b, gamma)
                if perm_mmd >= observed:
                    count += 1
            p_val = count / 10000
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."
            print(f"      {a} <-> {b}: p = {p_val:.4f} {sig}")

        # Key comparison
        d_ksk_mtb = mmd_values[("KSK", "MTB")]
        d_ksk_kan = mmd_values[("KSK", "KAN")]
        d_mtb_kan = mmd_values[("MTB", "KAN")]
        print(f"\n    Distance ordering:")
        print(f"      KSK<->MTB = {d_ksk_mtb:.6f}")
        print(f"      KSK<->KAN = {d_ksk_kan:.6f}")
        print(f"      MTB<->KAN = {d_mtb_kan:.6f}")
        closest = min([(d_ksk_mtb, "KSK-MTB"), (d_ksk_kan, "KSK-KAN"), (d_mtb_kan, "MTB-KAN")])
        farthest = max([(d_ksk_mtb, "KSK-MTB"), (d_ksk_kan, "KSK-KAN"), (d_mtb_kan, "MTB-KAN")])
        print(f"      Closest pair: {closest[1]} ({closest[0]:.6f})")
        print(f"      Farthest pair: {farthest[1]} ({farthest[0]:.6f})")


# ---------------------------------------------------------------------------
# Method 3: Pairwise sample distances
# ---------------------------------------------------------------------------
def pairwise_sample_distances(embeddings, metadata):
    print("\n" + "=" * 70)
    print("METHOD 3: PAIRWISE SAMPLE DISTANCES")
    print("=" * 70)

    results = {}
    for vtype in ["pant", "hoot", None]:
        vtype_label = vtype if vtype else "all"
        comms_emb = {}
        for comm in ["KSK", "MTB", "KAN"]:
            comms_emb[comm] = get_emb(embeddings, metadata, comm, vtype)

        print(f"\n  {vtype_label.upper()} embeddings:")

        # Compute all pairwise distances between communities
        for a, b in [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]:
            dists = cdist(comms_emb[a], comms_emb[b], metric="euclidean").ravel()
            results[(vtype_label, a, b)] = dists
            print(f"    {a} <-> {b}: mean={dists.mean():.4f}, "
                  f"median={np.median(dists):.4f}, std={dists.std():.4f}, "
                  f"n_pairs={len(dists)}")

        # Within-community distances for reference
        for comm in ["KSK", "MTB", "KAN"]:
            within = pdist(comms_emb[comm], metric="euclidean")
            print(f"    {comm} within: mean={within.mean():.4f}, "
                  f"median={np.median(within):.4f}")

        # Statistical test: KSK-MTB vs KSK-KAN distances
        d_ksk_mtb = results[(vtype_label, "KSK", "MTB")]
        d_ksk_kan = results[(vtype_label, "KSK", "KAN")]
        d_mtb_kan = results[(vtype_label, "MTB", "KAN")]

        u, p = stats.mannwhitneyu(d_ksk_mtb, d_ksk_kan, alternative="two-sided")
        es = 1 - (2 * u) / (len(d_ksk_mtb) * len(d_ksk_kan))
        print(f"\n    KSK-MTB vs KSK-KAN pairwise distances:")
        print(f"      Mann-Whitney U={u:.0f}, p={p:.6f}, effect={es:.3f}")
        print(f"      KSK-MTB mean={d_ksk_mtb.mean():.4f}, KSK-KAN mean={d_ksk_kan.mean():.4f}")
        print(f"      -> KSK is {'CLOSER to MTB' if d_ksk_mtb.mean() < d_ksk_kan.mean() else 'CLOSER to KAN'}")

        u2, p2 = stats.mannwhitneyu(d_ksk_mtb, d_mtb_kan, alternative="two-sided")
        es2 = 1 - (2 * u2) / (len(d_ksk_mtb) * len(d_mtb_kan))
        print(f"\n    KSK-MTB vs MTB-KAN pairwise distances:")
        print(f"      Mann-Whitney U={u2:.0f}, p={p2:.6f}, effect={es2:.3f}")
        print(f"      KSK-MTB mean={d_ksk_mtb.mean():.4f}, MTB-KAN mean={d_mtb_kan.mean():.4f}")

    return results


# ---------------------------------------------------------------------------
# Method 4: Classifier separability
# ---------------------------------------------------------------------------
def classifier_separability(embeddings, metadata):
    print("\n" + "=" * 70)
    print("METHOD 4: CLASSIFIER SEPARABILITY")
    print("(How well can a linear classifier tell communities apart?)")
    print("=" * 70)

    for vtype in ["pant", "hoot", None]:
        vtype_label = vtype if vtype else "all"
        print(f"\n  {vtype_label.upper()} embeddings:")

        # Pairwise classification accuracy
        for a, b in [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]:
            emb_a = get_emb(embeddings, metadata, a, vtype)
            emb_b = get_emb(embeddings, metadata, b, vtype)
            X = np.vstack([emb_a, emb_b])
            y = np.array([0] * len(emb_a) + [1] * len(emb_b))

            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)

            # SVM with 5-fold CV
            min_class = min(len(emb_a), len(emb_b))
            n_splits = min(5, min_class)
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            svm = LinearSVC(max_iter=5000, random_state=42)
            scores = cross_val_score(svm, X_s, y, cv=cv, scoring="accuracy")

            # LDA
            lda = LinearDiscriminantAnalysis()
            lda_scores = cross_val_score(lda, X_s, y, cv=cv, scoring="accuracy")

            print(f"    {a} vs {b} (n={len(emb_a)}+{len(emb_b)}):")
            print(f"      SVM accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")
            print(f"      LDA accuracy: {lda_scores.mean():.3f} +/- {lda_scores.std():.3f}")

        # 3-way classification
        X_all = []
        y_all = []
        for i, comm in enumerate(["KSK", "MTB", "KAN"]):
            emb = get_emb(embeddings, metadata, comm, vtype)
            X_all.append(emb)
            y_all.extend([i] * len(emb))
        X_all = np.vstack(X_all)
        y_all = np.array(y_all)
        X_s = StandardScaler().fit_transform(X_all)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        svm = LinearSVC(max_iter=5000, random_state=42)
        scores = cross_val_score(svm, X_s, y_all, cv=cv, scoring="accuracy")
        print(f"    3-way (KSK vs MTB vs KAN):")
        print(f"      SVM accuracy: {scores.mean():.3f} +/- {scores.std():.3f} (chance=0.333)")


# ---------------------------------------------------------------------------
# Method 5: PERMANOVA
# ---------------------------------------------------------------------------
def permanova(embeddings, metadata):
    print("\n" + "=" * 70)
    print("METHOD 5: PERMANOVA")
    print("(Multivariate permutation ANOVA on distance matrices)")
    print("=" * 70)

    for vtype in ["pant", "hoot", None]:
        vtype_label = vtype if vtype else "all"
        print(f"\n  {vtype_label.upper()} embeddings:")

        comms_emb = {}
        for comm in ["KSK", "MTB", "KAN"]:
            comms_emb[comm] = get_emb(embeddings, metadata, comm, vtype)

        # Pairwise PERMANOVA
        for a, b in [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]:
            X = np.vstack([comms_emb[a], comms_emb[b]])
            groups = np.array([0] * len(comms_emb[a]) + [1] * len(comms_emb[b]))
            n_a, n_b = len(comms_emb[a]), len(comms_emb[b])
            n_total = n_a + n_b

            # Distance matrix
            D = pairwise_distances(X, metric="euclidean")
            D2 = D ** 2

            # Centering matrix
            A = -0.5 * D2
            row_means = A.mean(axis=1, keepdims=True)
            col_means = A.mean(axis=0, keepdims=True)
            grand_mean = A.mean()
            G = A - row_means - col_means + grand_mean

            # SS_between and SS_total
            ss_total = np.trace(G)

            # SS_within
            mask_a = groups == 0
            mask_b = groups == 1
            ss_within = (np.trace(G[np.ix_(mask_a, mask_a)]) -
                        G[np.ix_(mask_a, mask_a)].sum() / n_a +
                        np.trace(G[np.ix_(mask_b, mask_b)]) -
                        G[np.ix_(mask_b, mask_b)].sum() / n_b)

            # Use simpler formulation: pseudo-F
            # SS_between = sum of between-group squared distances / n
            between_dists = D2[np.ix_(mask_a, mask_b)]
            within_a = D2[np.ix_(mask_a, mask_a)]
            within_b = D2[np.ix_(mask_b, mask_b)]

            ss_between = between_dists.sum() / n_total
            ss_within_a = within_a.sum() / (2 * n_a)
            ss_within_b = within_b.sum() / (2 * n_b)
            ss_within = ss_within_a + ss_within_b

            pseudo_f = (ss_between / 1) / (ss_within / (n_total - 2))

            # Permutation test
            rng = np.random.default_rng(42)
            n_perm = 10000
            count = 0
            for _ in range(n_perm):
                perm_groups = rng.permutation(groups)
                p_mask_a = perm_groups == 0
                p_mask_b = perm_groups == 1
                p_between = D2[np.ix_(p_mask_a, p_mask_b)].sum() / n_total
                p_within_a = D2[np.ix_(p_mask_a, p_mask_a)].sum() / (2 * n_a)
                p_within_b = D2[np.ix_(p_mask_b, p_mask_b)].sum() / (2 * n_b)
                p_within = p_within_a + p_within_b
                p_f = (p_between / 1) / (p_within / (n_total - 2))
                if p_f >= pseudo_f:
                    count += 1
            p_val = count / n_perm
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "n.s."

            # Effect size (R^2)
            r2 = ss_between / (ss_between + ss_within)

            print(f"    {a} vs {b}: pseudo-F={pseudo_f:.2f}, R^2={r2:.4f}, p={p_val:.4f} {sig}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_distance_matrix(embeddings, metadata):
    """Plot community distance matrices as heatmaps."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    comms = ["KSK", "MTB", "KAN"]

    for idx, (vtype, title) in enumerate([
        ("pant", "Pant Embeddings"), ("hoot", "Hoot Embeddings"), (None, "All Embeddings")
    ]):
        ax = axes[idx]
        matrix = np.zeros((3, 3))
        for i, a in enumerate(comms):
            for j, b in enumerate(comms):
                if i == j:
                    emb = get_emb(embeddings, metadata, a, vtype)
                    matrix[i, j] = pdist(emb, metric="euclidean").mean()
                else:
                    emb_a = get_emb(embeddings, metadata, a, vtype)
                    emb_b = get_emb(embeddings, metadata, b, vtype)
                    matrix[i, j] = cdist(emb_a, emb_b, metric="euclidean").mean()

        im = ax.imshow(matrix, cmap="YlOrRd", aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"{c}\n({COMMUNITIES[c]})" for c in comms])
        ax.set_yticklabels([f"{c}\n({COMMUNITIES[c]})" for c in comms])
        for i in range(3):
            for j in range(3):
                color = "white" if matrix[i, j] > matrix.mean() else "black"
                label = "within" if i == j else ""
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                        color=color, fontsize=11, fontweight="bold")
        ax.set_title(title, fontweight="bold")
        plt.colorbar(im, ax=ax, label="Mean Euclidean Distance")

    plt.suptitle("Mean Pairwise Distances Between Communities\n"
                 "(diagonal = within-community, off-diagonal = between-community)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "embedding_distance_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'embedding_distance_heatmaps.png'}")


def plot_pairwise_distributions(pairwise_results):
    """Plot distributions of pairwise distances."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"KSK-MTB": "#2ecc71", "KSK-KAN": "#e74c3c", "MTB-KAN": "#3498db"}

    for idx, (vtype, title) in enumerate([("pant", "Pant"), ("hoot", "Hoot"), ("all", "All")]):
        ax = axes[idx]
        for (a, b), color_key in [(("KSK", "MTB"), "KSK-MTB"),
                                   (("KSK", "KAN"), "KSK-KAN"),
                                   (("MTB", "KAN"), "MTB-KAN")]:
            dists = pairwise_results[(vtype, a, b)]
            ax.hist(dists, bins=50, alpha=0.5, label=f"{a}-{b} (mean={dists.mean():.3f})",
                    color=colors[color_key], density=True)

        ax.set_xlabel("Euclidean Distance")
        ax.set_ylabel("Density")
        ax.set_title(f"{title} Embeddings", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Distribution of Pairwise Distances Between Communities",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pairwise_distance_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'pairwise_distance_distributions.png'}")


def plot_summary(embeddings, metadata):
    """Summary bar chart of all distance metrics."""
    comms = ["KSK", "MTB", "KAN"]
    pairs = [("KSK", "MTB"), ("KSK", "KAN"), ("MTB", "KAN")]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    pair_labels = ["KSK-MTB\n(neighbors)", "KSK-KAN\n(distant)", "MTB-KAN"]
    colors_list = ["#2ecc71", "#e74c3c", "#3498db"]

    for idx, (vtype, title) in enumerate([(None, "All Embeddings"), ("pant", "Pants Only"), ("hoot", "Hoots Only")]):
        ax = axes[idx]

        # Centroid euclidean
        centroid_dists = []
        for a, b in pairs:
            c_a = get_emb(embeddings, metadata, a, vtype).mean(axis=0)
            c_b = get_emb(embeddings, metadata, b, vtype).mean(axis=0)
            centroid_dists.append(np.linalg.norm(c_a - c_b))

        # MMD
        all_data = np.vstack([get_emb(embeddings, metadata, c, vtype) for c in comms])
        gamma = 1.0 / (2 * np.median(pdist(all_data)) ** 2)
        mmd_dists = []
        for a, b in pairs:
            mmd_dists.append(compute_mmd(
                get_emb(embeddings, metadata, a, vtype),
                get_emb(embeddings, metadata, b, vtype), gamma))

        # Mean pairwise
        pw_dists = []
        for a, b in pairs:
            d = cdist(get_emb(embeddings, metadata, a, vtype),
                      get_emb(embeddings, metadata, b, vtype)).mean()
            pw_dists.append(d)

        # Normalize each metric to [0,1] for comparison
        x = np.arange(3)
        width = 0.25

        def norm(arr):
            arr = np.array(arr)
            mn, mx = arr.min(), arr.max()
            if mx == mn:
                return np.ones_like(arr) * 0.5
            return (arr - mn) / (mx - mn)

        ax.bar(x - width, norm(centroid_dists), width, label="Centroid (Eucl.)",
               color="#3498db", alpha=0.7)
        ax.bar(x, norm(mmd_dists), width, label="MMD",
               color="#e74c3c", alpha=0.7)
        ax.bar(x + width, norm(pw_dists), width, label="Mean pairwise",
               color="#2ecc71", alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels)
        ax.set_ylabel("Normalized Distance (0-1)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Community Distance Comparison Across Metrics\n"
                 "(Is KSK-MTB closer than KSK-KAN?)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "distance_metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {RESULTS_DIR / 'distance_metric_comparison.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("DIRECT EMBEDDING DISTANCE ANALYSIS")
    print("Is KSK closer to MTB than to KAN in embedding space?")
    print("=" * 70)

    embeddings, metadata = load_embeddings()
    print(f"Loaded {len(embeddings)} embeddings of dim {embeddings.shape[1]}")

    centroid_distances(embeddings, metadata)
    mmd_analysis(embeddings, metadata)
    pw_results = pairwise_sample_distances(embeddings, metadata)
    classifier_separability(embeddings, metadata)
    permanova(embeddings, metadata)

    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)
    plot_distance_matrix(embeddings, metadata)
    plot_pairwise_distributions(pw_results)
    plot_summary(embeddings, metadata)

    print("\nDone!")


if __name__ == "__main__":
    main()
