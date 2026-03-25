"""
Calibration script for Portfolio normalization data.
Runs random policy episodes to measure actual zt = [sum(reward[0]), sum(reward[1])]
ranges, then updates normalization_data/data.pickle with Portfolio entry.

Reward design (v4a):
  reward[0] = log return per step  (Expected Return proxy)
  reward[1] = -max(0, -r) / alpha  (CVaR per-step contribution, alpha=0.1)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'third_party', 'gymfolio', 'src'))

import numpy as np
import pickle

from MORL_stablebaselines3.envs.portfolio.mo_portfolio_env import MOFinancePortfolioEnv

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--reward_type', type=str, default='cvar_perstep',
                    choices=['cvar_perstep', 'sharpe_cvar', 'rolling_cvar', 'variance'])
parser.add_argument('--n_episodes', type=int, default=500)
cal_args = parser.parse_args()

N_EPISODES = cal_args.n_episodes
DATA_DIR   = 'data/portfolio_expanded/train'
RISK_SCALE = 1.0  # measure raw ranges first, then decide risk_scale

print(f"Running {N_EPISODES} random-policy episodes to calibrate zt ranges...")
print(f"reward_type: {cal_args.reward_type}")
env = MOFinancePortfolioEnv(
    data_dir=DATA_DIR,
    rebalance_every=5,
    max_trajectory_len=252,
    observation_frame_lookback=16,
    risk_scale=RISK_SCALE,
    reward_type=cal_args.reward_type,
)

zt0_list, zt1_list = [], []
for ep in range(N_EPISODES):
    obs, _ = env.reset()
    zt0, zt1 = 0.0, 0.0
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        zt0 += reward[0]
        zt1 += reward[1]
        done = terminated or truncated
    zt0_list.append(zt0)
    zt1_list.append(zt1)
    if (ep + 1) % 50 == 0:
        print(f"  {ep+1}/{N_EPISODES} done...")

zt0 = np.array(zt0_list)
zt1 = np.array(zt1_list)

print(f"\n=== Calibration Results (risk_scale={RISK_SCALE}) ===")
print(f"zt[0] (return):      min={zt0.min():.3f}, max={zt0.max():.3f}, range={zt0.max()-zt0.min():.3f}")
print(f"zt[1] (-CVaR contrib): min={zt1.min():.3f}, max={zt1.max():.3f}, range={zt1.max()-zt1.min():.3f}")

range0 = zt0.max() - zt0.min()
range1 = zt1.max() - zt1.min()
ratio  = range0 / (range1 + 1e-8)
print(f"\nrange ratio (zt0/zt1): {ratio:.3f}")
print(f"→ Recommended risk_scale to equalize: {1/ratio:.3f}")

# Determine risk_scale so ranges match
# We'll use a slight margin (5th/95th percentile for robustness)
zt0_min = float(np.percentile(zt0, 1))
zt0_max = float(np.percentile(zt0, 99))
zt1_min = float(np.percentile(zt1, 1))
zt1_max = float(np.percentile(zt1, 99))

print(f"\n=== Robust range (1st-99th percentile) ===")
print(f"zt[0]: [{zt0_min:.3f}, {zt0_max:.3f}]")
print(f"zt[1]: [{zt1_min:.3f}, {zt1_max:.3f}]")

recommended_risk_scale = (zt0_max - zt0_min) / ((zt1_max - zt1_min) + 1e-8)
print(f"\nRecommended risk_scale: {recommended_risk_scale:.3f}")

# Update normalization_data/data.pickle
norm_path = 'normalization_data/data.pickle'
with open(norm_path, 'rb') as f:
    norm_data = pickle.load(f)

# Use raw (risk_scale=1) ranges; risk_scale will be applied at reward time
norm_data['Portfolio'] = {
    'min': [np.array([zt0_min, zt1_min])],
    'max': [np.array([zt0_max, zt1_max])],
}
with open(norm_path, 'wb') as f:
    pickle.dump(norm_data, f)

print(f"\n✓ Updated normalization_data/data.pickle with Portfolio entry:")
print(f"  min = {norm_data['Portfolio']['min']}")
print(f"  max = {norm_data['Portfolio']['max']}")
print(f"\nDone. Use --portfolio_risk_scale {recommended_risk_scale:.2f} in training command.")
