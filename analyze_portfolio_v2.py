import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os, glob, re

EXP_DIR = 'experiments/dpmorl_portfolio_v2/DPMORL.Portfolio.LossNormLamda_0.01'
OUT_DIR = EXP_DIR

# ── 1. Load all test results ──────────────────────────────────────────────────
files = sorted(glob.glob(f'{EXP_DIR}/test_returns_policy_*.npz'))
policies, r0_means, r1_means, r0_stds, r1_stds = [], [], [], [], []

for f in files:
    label = re.search(r'test_returns_policy_(.+)\.npz', f).group(1)
    r = np.load(f)['test_returns']   # shape (100, 2)
    r0, r1 = r[:, 0], r[:, 1]
    policies.append(label)
    r0_means.append(r0.mean())
    r1_means.append(r1.mean())
    r0_stds.append(r0.std())
    r1_stds.append(r1.std())

r0_means = np.array(r0_means)
r1_means = np.array(r1_means)
r0_stds  = np.array(r0_stds)
r1_stds  = np.array(r1_stds)

# ── 2. Metrics ────────────────────────────────────────────────────────────────
sharpe = r0_means / (r0_stds + 1e-8)
corr_matrix = np.corrcoef(r0_means, r1_means)
corr_r0_r1  = corr_matrix[0, 1]

print("=" * 60)
print(f"{'Policy':<22} {'Return':>8} {'±':>6} {'Risk':>8} {'±':>6} {'Sharpe':>7}")
print("-" * 60)
for i, p in enumerate(policies):
    print(f"{p:<22} {r0_means[i]:>8.2f} {r0_stds[i]:>6.2f} {r1_means[i]:>8.2f} {r1_stds[i]:>6.2f} {sharpe[i]:>7.3f}")

print("=" * 60)
print(f"\n[Diversity]")
print(f"  return[0] range: {r0_means.min():.2f} ~ {r0_means.max():.2f}  (spread={r0_means.max()-r0_means.min():.2f})")
print(f"  return[1] range: {r1_means.min():.2f} ~ {r1_means.max():.2f}  (spread={r1_means.max()-r1_means.min():.2f})")
print(f"  corr(return, risk): {corr_r0_r1:.4f}")
print(f"  Sharpe range: {sharpe.min():.3f} ~ {sharpe.max():.3f}")

# ── 3. Plot 1: Policy scatter (return vs risk) ────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
colors = cm.plasma(np.linspace(0.1, 0.9, len(policies)))
for i, p in enumerate(policies):
    ax.scatter(r0_means[i], r1_means[i], color=colors[i], s=120, zorder=3)
    ax.annotate(p.replace('pretrain-', 'pt').replace('program-', 'pg'),
                (r0_means[i], r1_means[i]),
                textcoords='offset points', xytext=(5, 4), fontsize=7)
ax.set_xlabel('Cumulative Return (↑)')
ax.set_ylabel('Cumulative -Risk (↑ = less risk)')
ax.set_title(f'dpmorl_portfolio_v2: Policy Diversity (n={len(policies)})\ncorr={corr_r0_r1:.3f}')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/test_policy_scatter.png', dpi=150)
print(f"\nSaved: test_policy_scatter.png")

# ── 4. Plot 2: Sharpe bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
idx = np.argsort(sharpe)[::-1]
ax.bar(range(len(policies)), sharpe[idx], color=cm.viridis(np.linspace(0.2, 0.8, len(policies))))
ax.set_xticks(range(len(policies)))
ax.set_xticklabels([policies[i].replace('pretrain-', 'pt').replace('program-', 'pg') for i in idx], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Sharpe Ratio')
ax.set_title('Sharpe Ratio by Policy (sorted)')
ax.axhline(sharpe.mean(), color='red', linestyle='--', label=f'mean={sharpe.mean():.3f}')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUT_DIR}/test_policy_sharpe.png', dpi=150)
print(f"Saved: test_policy_sharpe.png")

# ── 5. Plot 3: Return distribution box ───────────────────────────────────────
all_r0, all_r1, labels = [], [], []
for f in files:
    label = re.search(r'test_returns_policy_(.+)\.npz', f).group(1)
    r = np.load(f)['test_returns']
    all_r0.append(r[:, 0])
    all_r1.append(r[:, 1])
    labels.append(label.replace('pretrain-', 'pt').replace('program-', 'pg'))

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
axes[0].boxplot(all_r0, labels=labels, vert=True)
axes[0].set_title('Return[0] distribution per policy')
axes[0].set_ylabel('Cumulative Return')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3, axis='y')

axes[1].boxplot(all_r1, labels=labels, vert=True)
axes[1].set_title('Return[1] (-Risk) distribution per policy')
axes[1].set_ylabel('Cumulative -Risk')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUT_DIR}/test_policy_boxplot.png', dpi=150)
print(f"Saved: test_policy_boxplot.png")

# ── 6. Compare with old experiment ───────────────────────────────────────────
old_files = sorted(glob.glob('experiments/dpmorl_portfolio_a/DPMORL.Portfolio.LossNormLamda_0.1/test_returns_policy_*.npz'))
if old_files:
    old_r0, old_r1 = [], []
    for f in old_files:
        r = np.load(f)['test_returns']
        old_r0.append(r[:, 0].mean())
        old_r1.append(r[:, 1].mean())
    old_r0, old_r1 = np.array(old_r0), np.array(old_r1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(old_r0, old_r1, c='blue', s=80, label=f'v1 (n={len(old_r0)})', alpha=0.7)
    axes[0].scatter(r0_means, r1_means, c='red', s=80, label=f'v2 (n={len(r0_means)})', alpha=0.7)
    axes[0].set_xlabel('Return[0]')
    axes[0].set_ylabel('Return[1] (-Risk)')
    axes[0].set_title('v1 vs v2: Policy Diversity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(old_r0, old_r1, c='blue', s=80, label=f'v1 spread=({old_r0.max()-old_r0.min():.2f}, {old_r1.max()-old_r1.min():.2f})', alpha=0.7)
    axes[1].scatter(r0_means, r1_means, c='red', s=80, label=f'v2 spread=({r0_means.max()-r0_means.min():.2f}, {r1_means.max()-r1_means.min():.2f})', alpha=0.7)
    axes[1].set_xlabel('Return[0]')
    axes[1].set_ylabel('Return[1] (-Risk)')
    axes[1].set_title('v1 vs v2: Spread Comparison')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/test_v1_vs_v2_comparison.png', dpi=150)
    print(f"Saved: test_v1_vs_v2_comparison.png")

plt.close('all')
print("\nDone.")
