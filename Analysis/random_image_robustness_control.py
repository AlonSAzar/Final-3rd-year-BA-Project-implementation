
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim

# ==========================
# CONFIGURATION
# ==========================

L_values = [10, 50, 100, 250, 500]
T_values = [10, 50, 100, 250, 500]

num_trials = 100          # Random image pairs per (L,T)
seed = 0
rng = np.random.default_rng(seed)

# ==========================
# METRICS
# ==========================

def compute_ncc(a, b):
    a = a.astype(float).ravel()
    b = b.astype(float).ravel()

    a -= a.mean()
    b -= b.mean()

    denom = np.sqrt(np.sum(a*a) * np.sum(b*b))
    if denom == 0:
        return 0.0
    return np.sum(a*b) / denom


def compute_ssim(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    win = min(7, min(img1.shape))
    if win % 2 == 0:
        win -= 1
    if win < 3:
        win = 3

    return ssim(img1, img2, data_range=1.0, win_size=win)


def compute_nmi(a, b):
    return normalized_mutual_info_score(a.ravel(), b.ravel())


# ==========================
# DAMAGE CONE
# ==========================

def randomize_damage_cone(image, flipped_seed_bit, rng):
    """
    Radius-1 ECA damage cone with periodic boundaries.
    image shape = (T+1, L)
    """
    mutant = image.copy()

    rows, L = mutant.shape

    for t in range(rows):
        for dx in range(-t, t + 1):
            x = (flipped_seed_bit + dx) % L
            mutant[t, x] = rng.integers(0, 2)

    return mutant


# ==========================
# MAIN EXPERIMENT
# ==========================

ncc_heat = np.zeros((len(T_values), len(L_values)))
ssim_heat = np.zeros_like(ncc_heat)
nmi_heat = np.zeros_like(ncc_heat)

print("Running random-image damage-cone control...")

for ti, T in enumerate(tqdm(T_values)):
    for li, L in enumerate(L_values):

        ncc_scores = []
        ssim_scores = []
        nmi_scores = []

        for _ in range(num_trials):

            img = rng.integers(0, 2, size=(T + 1, L), dtype=np.uint8)

            flipped_seed = rng.integers(L)

            mutant = randomize_damage_cone(img, flipped_seed, rng)

            ncc_scores.append(compute_ncc(img, mutant))
            ssim_scores.append(compute_ssim(img, mutant))
            nmi_scores.append(compute_nmi(img, mutant))

        ncc_heat[ti, li] = np.mean(ncc_scores)
        ssim_heat[ti, li] = np.mean(ssim_scores)
        nmi_heat[ti, li] = np.mean(nmi_scores)

        print(
            f"L={L:3d} T={T:3d} "
            f"NCC={ncc_heat[ti,li]:.3f} "
            f"SSIM={ssim_heat[ti,li]:.3f} "
            f"NMI={nmi_heat[ti,li]:.3f}"
        )

os.makedirs("Saved Figures", exist_ok=True)

def plot_heatmap(data, title, filename):
    plt.figure(figsize=(9,7))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=L_values,
        yticklabels=T_values
    )
    plt.xlabel("Grid Size (L)")
    plt.ylabel("Time Steps (T)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join("Saved Figures", filename))
    plt.close()

plot_heatmap(ncc_heat,
             "Random Control - NCC",
             "RandomControl_NCC.png")

plot_heatmap(ssim_heat,
             "Random Control - SSIM",
             "RandomControl_SSIM.png")

plot_heatmap(nmi_heat,
             "Random Control - Normalized MI",
             "RandomControl_NMI.png")

print("Finished.")