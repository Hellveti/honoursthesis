import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── IQR helpers for folded (absolute-value) distributions ──
# |N(0, σ²)|:  CDF = 2Φ(x/σ) − 1  →  Qp = σ · Φ⁻¹((1+p)/2)
def folded_normal_iqr(sigma):
    q1 = sigma * norm.ppf(0.625)   # p = 0.25
    q3 = sigma * norm.ppf(0.875)   # p = 0.75
    return q3 - q1

# |C(0, γ)|:  CDF = (2/π) arctan(x/γ)  →  Qp = γ tan(pπ/2)
def folded_cauchy_iqr(gamma):
    q1 = gamma * np.tan(0.25 * np.pi / 2)
    q3 = gamma * np.tan(0.75 * np.pi / 2)
    return q3 - q1

# ── Data from notebook ──
# Gaussian–Gaussian mixtures (σ₁ = 1 fixed, σ₂ varies)
# gg_data = [
#     {"sigma2": 2, "mean_steps": 36.70},
#     {"sigma2": 3, "mean_steps": 27.37},
#     {"sigma2": 4, "mean_steps": 22.44},
#     {"sigma2": 5, "mean_steps": 19.88},
# ]

# # Gaussian–Cauchy mixtures (γ = 1 fixed, Gaussian σ varies)
# gc_data = [
#     {"sigma": 1.0,      "gamma": 1, "mean_steps": 33.86},
#     {"sigma": 1.482602, "gamma": 1, "mean_steps": 33.33},
#     {"sigma": 2.4045,   "gamma": 1, "mean_steps": 33.36},
# ]

gg_data = [
    {"sigma2": 2, "mean_steps": 96.79},
    {"sigma2": 3, "mean_steps": 38.03},
    {"sigma2": 4, "mean_steps": 29.82},
    {"sigma2": 5, "mean_steps": 28.46},
]


gc_data = [
    {"sigma": 1.0,      "gamma": 1, "mean_steps": 42.01},
    {"sigma": 2.4045,   "gamma": 1, "mean_steps": 42.03},
]


# IQR *difference* between the two mixture components
SIGMA1 = 1  # fixed small-step Gaussian in all mixtures
gg_iqr   = [folded_normal_iqr(d["sigma2"]) - folded_normal_iqr(SIGMA1) for d in gg_data]
gg_steps = [d["mean_steps"] for d in gg_data]

gc_iqr   = [folded_cauchy_iqr(d["gamma"]) - folded_normal_iqr(d["sigma"]) for d in gc_data]
gc_steps = [d["mean_steps"] for d in gc_data]

# ── Line of best fit (all points combined) ──
all_iqr   = np.array(gg_iqr + gc_iqr)
all_steps = np.array(gg_steps + gc_steps)
coeffs = np.polyfit(all_iqr, all_steps, 1)
fit_x = np.linspace(all_iqr.min() - 0.1, all_iqr.max() + 0.1, 100)
fit_y = np.polyval(coeffs, fit_x)

# ── Plot ──
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(fit_x, fit_y, '-', color='grey', lw=1.2, alpha=0.7,
        label=rf'Best fit: $y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}$')
ax.plot(gg_iqr, gg_steps, 'x',  color="blue", markersize=3,
        label='Gaussian–Gaussian Mixture')
ax.plot(gc_iqr, gc_steps, 'x', color="red", markersize=3,
        label='Cauchy-Gaussian Mixture')

# Annotate GG points with σ₂ values
for d, x, y in zip(gg_data, gg_iqr, gg_steps):
    ax.annotate(rf'$\sigma_2={d["sigma2"]}$', (x, y),
                textcoords='offset points', xytext=(8, 6), fontsize=9)

# Annotate GC points with σ values
for d, x, y in zip(gc_data, gc_iqr, gc_steps):
    ax.annotate(rf'$\sigma_G={d["sigma"]:.2f},\;\gamma={d["gamma"]}$', (x, y),
                textcoords='offset points', xytext=(8, 6), fontsize=8)

ax.set_xlabel(r'Difference in IQR between mixture components', fontsize=12)
ax.set_ylabel('Mean steps to target', fontsize=12)
ax.set_title('Search efficiency', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig.tight_layout()
plt.show()
plt.close()

# Print computed ΔIQR for reference
print("\nComputed ΔIQRs:")
print(f"  |N(0,σ²)| IQR coefficient: {folded_normal_iqr(1):.4f} × σ")
print(f"  |C(0,γ)|  IQR coefficient: {folded_cauchy_iqr(1):.4f} × γ")
for d, iqr in zip(gg_data, gg_iqr):
    print(f"  GG σ₁=1, σ₂={d['sigma2']}:  ΔIQR = {iqr:.3f}")
for d, iqr in zip(gc_data, gc_iqr):
    print(f"  GC σ={d['sigma']:.4f}, γ={d['gamma']}:  ΔIQR = {iqr:.3f}")