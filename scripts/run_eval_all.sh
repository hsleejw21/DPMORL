#!/bin/bash
# run_eval_all.sh — Evaluate all trained portfolio policies in v_a format.
#
# Evaluates v3, v4a, v4b policies that were trained but never evaluated.
# Results saved to each experiment's directory as test_returns_policy_*.npz
# (shape 100x2), same format as dpmorl_portfolio_a and dpmorl_portfolio_v2.
#
# risk_scale=1.0 for all experiments → raw unscaled 2-D returns, comparable across versions.
# data_dir=train  → same dataset as v_a / v2 evaluations.
#
# Usage:
#   cd /path/to/DPMORL && bash scripts/run_eval_all.sh 2>&1 | tee outputs/eval_all.log

set -e
cd "$(dirname "$0")/.."

echo "=========================================="
echo " Portfolio policy evaluation (v_a format)"
echo "=========================================="

# ── v3: linear utilities, cvar_perstep, 5 policies ──────────────────────────
echo ""
echo "[v3] cvar_perstep | 5 policies"
python eval_portfolio.py \
    --exp_name dpmorl_portfolio_v3 \
    --reward_type cvar_perstep \
    --data_dir data/portfolio_expanded/train \
    --risk_scale 1.0 \
    --n_episodes 100 \
    --lamda 0.01

# ── v4a: linear utilities, cvar_perstep, 6 policies ─────────────────────────
echo ""
echo "[v4a] cvar_perstep | 6 policies"
python eval_portfolio.py \
    --exp_name dpmorl_portfolio_v4a \
    --reward_type cvar_perstep \
    --data_dir data/portfolio_expanded/train \
    --risk_scale 1.0 \
    --n_episodes 100 \
    --lamda 0.01

# ── v4b: linear utilities, sharpe_cvar, 4 policies ──────────────────────────
echo ""
echo "[v4b] sharpe_cvar | 4 policies"
python eval_portfolio.py \
    --exp_name dpmorl_portfolio_v4b \
    --reward_type sharpe_cvar \
    --data_dir data/portfolio_expanded/train \
    --risk_scale 1.0 \
    --n_episodes 100 \
    --lamda 0.01

echo ""
echo "=========================================="
echo " All evaluations complete."
echo "=========================================="
