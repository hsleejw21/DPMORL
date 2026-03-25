"""
Visualization for dpmorl_portfolio_v3.
Replicates DPMORL paper Figure 5/6 style:
  - Left panel:  utility function contour in zt space
  - Right panel: return distribution of trained policy (scatter of 100 test episodes)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob, os, sys, re

EXP_NAME   = 'dpmorl_portfolio_v3'
EXP_DIR    = f'experiments/{EXP_NAME}'
LAMDA      = '0.01'

# The 3 utility functions we want to highlight (linear utility function indices)
# Utility_Function_Linear weights:
#   0:[0.9,0.1], 1:[0.8,0.2], 2:[0.7,0.3], 3:[0.5,0.5], 4:[0.3,0.7], 5:[0.2,0.8],
#   6:[0.1,0.9], 7:[0.98,0.02], 8:[0.02,0.98], 9:[0.95,0.05], 10:[0.05,0.95],
#   11:[1.0,0.0], 12:[0.0,1.0]
HIGHLIGHT = {
    'program-11': {'weights': [1.0, 0.0], 'label': 'UF-A: Pure Return\n[w0=1.0, w1=0.0]',  'color': '#e74c3c'},
    'program-3':  {'weights': [0.5, 0.5], 'label': 'UF-B: Balanced\n[w0=0.5, w1=0.5]',     'color': '#2ecc71'},
    'program-12': {'weights': [0.0, 1.0], 'label': 'UF-C: Pure Safety\n[w0=0.0, w1=1.0]',  'color': '#3498db'},
}

def find_exp_dir():
    candidates = glob.glob(f'{EXP_DIR}/DPMORL.Portfolio.LossNormLamda_{LAMDA}')
    if candidates:
        return candidates[0]
    # fallback: find any subdirectory
    subdirs = [d for d in glob.glob(f'{EXP_DIR}/*/') if os.path.isdir(d)]
    if subdirs:
        return subdirs[0].rstrip('/')
    raise FileNotFoundError(f"No experiment directory found under {EXP_DIR}")

def load_test_returns(utility_dir, policy_name):
    path = f'{utility_dir}/test_returns_policy_{policy_name}.npz'
    if not os.path.exists(path):
        return None
    return np.load(path)['test_returns']  # shape (N, 2)

def linear_utility(z0, z1, w0, w1, z0_range, z1_range):
    """Normalized linear utility for contour plotting."""
    norm_z0 = (z0 - z0_range[0]) / (z0_range[1] - z0_range[0] + 1e-8)
    norm_z1 = (z1 - z1_range[0]) / (z1_range[1] - z1_range[0] + 1e-8)
    return w0 * norm_z0 + w1 * norm_z1

def plot_figure5_style(utility_dir, out_path):
    """Main figure: 3 columns × 2 rows (utility contour + return scatter)."""
    # Load all available test returns to determine zt range
    all_r0, all_r1 = [], []
    for name in HIGHLIGHT:
        r = load_test_returns(utility_dir, name)
        if r is not None:
            all_r0.extend(r[:, 0].tolist())
            all_r1.extend(r[:, 1].tolist())

    if not all_r0:
        print("No test results found. Run eval first.")
        return

    # zt display range (with small padding)
    pad = 0.05
    r0_lo, r0_hi = min(all_r0), max(all_r0)
    r1_lo, r1_hi = min(all_r1), max(all_r1)
    r0_pad = (r0_hi - r0_lo) * pad
    r1_pad = (r1_hi - r1_lo) * pad
    z0_range = (r0_lo - r0_pad, r0_hi + r0_pad)
    z1_range = (r1_lo - r1_pad, r1_hi + r1_pad)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle('DPMORL Portfolio v3: Utility Functions & Return Distributions', fontsize=14, y=1.01)

    z0g = np.linspace(*z0_range, 200)
    z1g = np.linspace(*z1_range, 200)
    Z0, Z1 = np.meshgrid(z0g, z1g)

    for col, (name, info) in enumerate(HIGHLIGHT.items()):
        w0, w1 = info['weights']
        color  = info['color']
        label  = info['label']
        r = load_test_returns(utility_dir, name)

        # ── Top row: utility function contour ─────────────────────────────
        ax_top = axes[0, col]
        U = linear_utility(Z0, Z1, w0, w1, z0_range, z1_range)
        cf = ax_top.contourf(Z0, Z1, U, levels=20, cmap='YlOrRd', alpha=0.85)
        plt.colorbar(cf, ax=ax_top, shrink=0.8)
        ax_top.set_title(f'Utility Function\n{label}', fontsize=10)
        ax_top.set_xlabel('Cumulative Return (↑)')
        ax_top.set_ylabel('Cumulative -Drawdown (↑)')
        # Mark optimal direction
        ax_top.annotate('', xy=(z0_range[1]*0.85, z1_range[1]*0.85),
                        xytext=(z0_range[0]*0.5, z1_range[0]*0.5),
                        arrowprops=dict(arrowstyle='->', color='white', lw=2))

        # ── Bottom row: return distribution scatter ────────────────────────
        ax_bot = axes[1, col]
        if r is not None:
            # Background: light contour of utility
            ax_bot.contourf(Z0, Z1, U, levels=10, cmap='YlOrRd', alpha=0.25)
            ax_bot.scatter(r[:, 0], r[:, 1], c=color, s=30, alpha=0.7,
                           edgecolors='white', linewidth=0.3, label=f'n={len(r)}')
            # Mark mean
            ax_bot.scatter(r[:, 0].mean(), r[:, 1].mean(), c='black', s=120,
                           marker='*', zorder=5, label=f'mean=({r[:,0].mean():.1f}, {r[:,1].mean():.1f})')
            ax_bot.legend(fontsize=8)
        else:
            ax_bot.text(0.5, 0.5, f'No data\nfor {name}',
                        ha='center', va='center', transform=ax_bot.transAxes, color='gray')

        ax_bot.set_xlim(z0_range)
        ax_bot.set_ylim(z1_range)
        ax_bot.set_title(f'Return Distribution\n{name}', fontsize=10)
        ax_bot.set_xlabel('Cumulative Return (↑)')
        ax_bot.set_ylabel('Cumulative -Drawdown (↑)')
        ax_bot.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")

def plot_diversity_scatter(utility_dir, out_path):
    """All policies: diversity in (return, -drawdown) space."""
    all_files = sorted(glob.glob(f'{utility_dir}/test_returns_policy_*.npz'))
    if not all_files:
        print("No test results found.")
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = cm.plasma(np.linspace(0.1, 0.9, len(all_files)))
    all_r0_means, all_r1_means = [], []

    for i, f in enumerate(all_files):
        name = re.search(r'test_returns_policy_(.+)\.npz', f).group(1)
        r = np.load(f)['test_returns']
        r0m, r1m = r[:, 0].mean(), r[:, 1].mean()
        all_r0_means.append(r0m); all_r1_means.append(r1m)

        c = HIGHLIGHT.get(name, {}).get('color', colors[i])
        size = 200 if name in HIGHLIGHT else 60
        ax.scatter(r0m, r1m, color=c, s=size, zorder=3,
                   edgecolors='black' if name in HIGHLIGHT else 'none', linewidth=1.5)
        ax.annotate(name.replace('program-', 'p'), (r0m, r1m),
                    textcoords='offset points', xytext=(5, 4), fontsize=7)

    # Highlight the 3 key ones
    for name, info in HIGHLIGHT.items():
        r = load_test_returns(utility_dir, name)
        if r is not None:
            ax.scatter(r[:, 0].mean(), r[:, 1].mean(),
                       color=info['color'], s=250, zorder=4,
                       edgecolors='black', linewidth=2, label=info['label'].split('\n')[0])

    r0a = np.array(all_r0_means); r1a = np.array(all_r1_means)
    ax.set_xlabel('Mean Cumulative Return (↑)', fontsize=12)
    ax.set_ylabel('Mean Cumulative -Drawdown (↑ = less risk)', fontsize=12)
    ax.set_title(f'Policy Diversity — {EXP_NAME}\n'
                 f'return spread={r0a.max()-r0a.min():.2f}, '
                 f'drawdown spread={r1a.max()-r1a.min():.2f}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

def print_metrics(utility_dir):
    """Print key metrics for each policy."""
    files = sorted(glob.glob(f'{utility_dir}/test_returns_policy_*.npz'))
    if not files:
        print("No test results found.")
        return

    print(f"\n{'Policy':<22} {'Return':>8} {'±':>6} {'Drawdown':>10} {'±':>6} {'Sharpe':>7}")
    print("-" * 65)
    all_r0, all_r1 = [], []
    for f in files:
        name = re.search(r'test_returns_policy_(.+)\.npz', f).group(1)
        r = np.load(f)['test_returns']
        r0, r1 = r[:, 0], r[:, 1]
        sharpe = r0.mean() / (r0.std() + 1e-8)
        all_r0.append(r0.mean()); all_r1.append(r1.mean())
        marker = ' ◄' if name in HIGHLIGHT else ''
        print(f"{name:<22} {r0.mean():>8.2f} {r0.std():>6.2f} {r1.mean():>10.2f} {r1.std():>6.2f} {sharpe:>7.3f}{marker}")

    r0a = np.array(all_r0); r1a = np.array(all_r1)
    print(f"\nDiversity — return spread: {r0a.max()-r0a.min():.2f}, "
          f"drawdown spread: {r1a.max()-r1a.min():.2f}")


if __name__ == '__main__':
    try:
        utility_dir = find_exp_dir()
    except FileNotFoundError as e:
        print(e); sys.exit(1)

    print(f"Using experiment dir: {utility_dir}")
    print_metrics(utility_dir)
    plot_figure5_style(utility_dir, f'{utility_dir}/figure5_style.png')
    plot_diversity_scatter(utility_dir, f'{utility_dir}/diversity_scatter.png')
    print("\nDone.")
