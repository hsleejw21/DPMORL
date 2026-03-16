#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$ROOT_DIR"

# You can override these with environment variables.
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-300000}"
NUM_ENVS="${NUM_ENVS:-8}"
MAX_POLICIES="${MAX_POLICIES:-16}"
EXP_NAME="${EXP_NAME:-dpmorl_portfolio_expanded}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/portfolio_expanded/train}"
LAMDA="${LAMDA:-0.1}"
PRETRAINED_ONLY="${PRETRAINED_ONLY:-False}"
NUM_PRETRAINED_TO_USE="${NUM_PRETRAINED_TO_USE:-0}"

PYTHONPATH="$PYTHONPATH" conda run -n dpmorl python -u "$ROOT_DIR/main_policy.py" \
  --env Portfolio \
  --reward_two_dim \
  --lamda "$LAMDA" \
  --exp_name "$EXP_NAME" \
  --max_num_policies "$MAX_POLICIES" \
  --pretrained_only "$PRETRAINED_ONLY" \
  --num_pretrained_to_use "$NUM_PRETRAINED_TO_USE" \
  --total_timesteps "$TOTAL_TIMESTEPS" \
  --num_envs "$NUM_ENVS" \
  --portfolio_data_dir "$DATA_DIR" \
  --portfolio_rebalance_every 5 \
  --portfolio_max_trajectory_len 252 \
  --portfolio_lookback 20 \
  --portfolio_risk_scale 1.0
