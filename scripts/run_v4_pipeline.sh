#!/usr/bin/env bash
# Sequential pipeline: v4b (rolling_cvar) → v4c (variance)
# Run AFTER v4a finishes. v4a uses cvar_perstep (default).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

COMMON_ARGS="
  --env Portfolio
  --linear_utility True
  --max_num_policies 13
  --total_timesteps 2000000
  --num_envs 20
  --portfolio_lookback 16
  --portfolio_rebalance_every 5
  --portfolio_max_trajectory_len 252
  --lamda 0.01
"

# ── v4b: rolling CVaR ────────────────────────────────────────────────────────
echo "=========================================="
echo " Starting v4b: Return + Rolling CVaR"
echo "=========================================="
# rolling_cvar needs its own calibration (zt[1] range is different)
# For now use same risk_scale=0.49 as v4a; can re-calibrate if needed
python main_policy.py \
  $COMMON_ARGS \
  --exp_name dpmorl_portfolio_v4b \
  --portfolio_risk_scale 0.49 \
  --portfolio_reward_type rolling_cvar \
  2>&1 | tee outputs/portfolio_v4b.log

echo "v4b done."

# ── v4c: variance ────────────────────────────────────────────────────────────
echo "=========================================="
echo " Starting v4c: Return + Variance"
echo "=========================================="
# variance reward: zt[1] = -sum(r_t^2), range likely much smaller
# calibrate first, then train
python calibrate_portfolio_normalization.py \
  --reward_type variance \
  2>&1 | tee outputs/calibration_v4c.log

RISK_SCALE_V4C=$(grep "Use --portfolio_risk_scale" outputs/calibration_v4c.log | awk '{print $NF}')
echo "v4c risk_scale = $RISK_SCALE_V4C"

python main_policy.py \
  $COMMON_ARGS \
  --exp_name dpmorl_portfolio_v4c \
  --portfolio_risk_scale "${RISK_SCALE_V4C:-1.0}" \
  --portfolio_reward_type variance \
  2>&1 | tee outputs/portfolio_v4c.log

echo "v4c done."
echo "All v4 experiments complete."
