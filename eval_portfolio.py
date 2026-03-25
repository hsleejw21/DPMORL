"""eval_portfolio.py — Evaluate trained portfolio policies and save 2-D episode returns.

Mirrors the v_a evaluation format:
  test_returns_policy_{name}.npz  →  key 'test_returns', shape (N_episodes, 2)
  where [:,0] = cumulative reward[0] and [:,1] = cumulative reward[1] over episode.

risk_scale=1.0 is used by default so raw, unscaled rewards are recorded across all
experiments, making cross-experiment comparison (v_a, v2, v3, v4a, v4b) fair.

Usage:
    python eval_portfolio.py --exp_name dpmorl_portfolio_v3  --reward_type cvar_perstep
    python eval_portfolio.py --exp_name dpmorl_portfolio_v4a --reward_type cvar_perstep
    python eval_portfolio.py --exp_name dpmorl_portfolio_v4b --reward_type sharpe_cvar
    python eval_portfolio.py --exp_name dpmorl_portfolio_a   --reward_type variance   --lamda 0.1

    # Use training data (same as v_a/v2 evaluation):
    python eval_portfolio.py --exp_name dpmorl_portfolio_v3 --reward_type cvar_perstep \\
        --data_dir data/portfolio_expanded/train
"""

import argparse
import glob
import os
import sys

import numpy as np
from stable_baselines3 import PPO

from MORL_stablebaselines3.envs.portfolio.mo_portfolio_env import MOFinancePortfolioEnv
from MORL_stablebaselines3.envs.wrappers.utility_env_wrapper import ObsInfoWrapper
from utils import DummyVecEnv


def make_eval_env(data_dir, reward_type, risk_scale, n_envs):
    """Build a DummyVecEnv that tracks raw 2-D episode returns via ObsInfoWrapper."""
    def _make_one():
        env = MOFinancePortfolioEnv(
            data_dir=data_dir,
            rebalance_every=5,
            max_trajectory_len=252,
            observation_frame_lookback=16,
            risk_scale=risk_scale,
            reward_type=reward_type,
        )
        return ObsInfoWrapper(env, reward_dim=2, reward_dim_indices=[0, 1])
    return DummyVecEnv([_make_one for _ in range(n_envs)], reward_dim=2)


def evaluate_policy(model, env, n_episodes):
    """Run policy for n_episodes; return array shape (n_episodes, 2) of cumulative rewards."""
    episode_returns = []
    obs = env.reset()
    while len(episode_returns) < n_episodes:
        action, _ = model.predict(obs, deterministic=False)
        obs, _, done, infos = env.step(action)
        for info in infos:
            if "episode" in info:
                episode_returns.append(info["episode"]["r"])
                if len(episode_returns) % 10 == 0:
                    print(f"    progress: {len(episode_returns)}/{n_episodes}")
    return np.array(episode_returns[:n_episodes])


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained portfolio policies (v_a format)")
    parser.add_argument("--exp_name", required=True,
                        help="e.g. dpmorl_portfolio_v3")
    parser.add_argument("--reward_type", default="cvar_perstep",
                        choices=["cvar_perstep", "sharpe_cvar", "rolling_cvar", "variance"],
                        help="Must match the reward type used during training")
    parser.add_argument("--data_dir", default="data/portfolio_expanded/train",
                        help="Portfolio data directory for evaluation (default: train, same as v_a/v2)")
    parser.add_argument("--risk_scale", type=float, default=1.0,
                        help="Risk scaling factor (default: 1.0 for raw comparable returns)")
    parser.add_argument("--n_episodes", type=int, default=100,
                        help="Number of test episodes per policy")
    parser.add_argument("--n_envs", type=int, default=10,
                        help="Number of parallel envs for evaluation")
    parser.add_argument("--lamda", type=float, default=0.01,
                        help="Lambda used during training (determines experiment subdir name)")
    args = parser.parse_args()

    exp_dir = f"experiments/{args.exp_name}/DPMORL.Portfolio.LossNormLamda_{args.lamda}"
    if not os.path.isdir(exp_dir):
        sys.exit(f"Experiment directory not found: {exp_dir}")

    policy_files = sorted(glob.glob(f"{exp_dir}/policy-*.zip"))
    if not policy_files:
        sys.exit(f"No policy .zip files found in {exp_dir}")

    print(f"Experiment  : {args.exp_name}")
    print(f"Policies    : {len(policy_files)} found")
    print(f"Data dir    : {args.data_dir}")
    print(f"Reward type : {args.reward_type}  risk_scale={args.risk_scale}")
    print(f"Episodes    : {args.n_episodes} per policy")
    print()

    env = make_eval_env(args.data_dir, args.reward_type, args.risk_scale, args.n_envs)

    for policy_file in policy_files:
        policy_name = os.path.basename(policy_file).replace("policy-", "").replace(".zip", "")
        out_path = os.path.join(exp_dir, f"test_returns_policy_{policy_name}.npz")
        if os.path.exists(out_path):
            print(f"  [{policy_name}] already evaluated — skipping.")
            continue

        print(f"  Evaluating {policy_name}...")
        model = PPO.load(policy_file, device="cpu")
        returns = evaluate_policy(model, env, args.n_episodes)
        np.savez_compressed(out_path, test_returns=returns)
        print(f"  Saved: {out_path}  shape={returns.shape}"
              f"  r0={returns[:, 0].mean():.3f}±{returns[:, 0].std():.3f}"
              f"  r1={returns[:, 1].mean():.3f}±{returns[:, 1].std():.3f}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
