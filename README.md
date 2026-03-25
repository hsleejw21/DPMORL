# DPMORL

Implementations for our paper [*Distributional Pareto-Optimal Multi-Objective Reinforcement Learning*](https://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html) at NeurIPS 2023. 

## Summary

**DPMORL** (Distributional Pareto-Optimal Multi-Objective Reinforcement Learning) is a framework published at NeurIPS 2023 for learning a diverse set of policies that collectively approximate the Pareto front in multi-objective reinforcement learning (MORL). Rather than collapsing multiple objectives into a single scalar up front, DPMORL trains one policy per utility function and reasons about the full *distribution* of episode returns — capturing risk, variance, and tail behaviour — not just expected value.

---

### Problem Setting

Real-world sequential decision-making often involves conflicting objectives (e.g., speed vs. safety, reward vs. energy). Classical RL requires the designer to fix a scalar trade-off before training. MORL instead discovers a *Pareto front* — a set of policies where no single policy dominates another across all objectives — and hands that front to the decision-maker. DPMORL extends this to the distributional setting: it is not just the mean return that matters, but the entire return distribution, enabling risk-sensitive policy selection.

---

### Algorithm Overview

DPMORL proceeds in three phases:

**Phase 1 — Utility Function Generation (`main_generate_utility.py`)**  
A library of diverse, monotone utility functions is built before any policy training:
- A small set of *programmed* utilities (`Utility_Function_Programmed`) are defined analytically (weighted sums, softplus compositions, etc.).
- A large set of *parameterised* utilities (`Utility_Function_Parameterized`) are trained as monotone MLPs. Each is trained to be *diverse* from all existing utilities using a repulsion loss in function-value and angular-derivative space (`logsumexp`-based contrastive loss).
- All weights are clamped to `[0, max_weight]` after every gradient step, enforcing monotonicity: utility can only increase when any reward dimension improves.
- 33 pre-trained utility checkpoints for 2-dimensional reward are provided in `utility-model-selected/dim-2/`.

**Phase 2 — Policy Training (`main_policy.py`)**  
Each utility function induces a scalar reward signal; one PPO agent is trained per utility:
- The environment's raw vector reward `r ∈ ℝᵈ` is accumulated into a running return `zₜ`.
- The scalar reward given to PPO at step `t` is `U(zₜ) − U(zₜ₋₁)` — the *marginal utility* of the new transition.
- This reward shaping makes the PPO objective exactly equivalent to maximising the expected final utility `E[U(Z_T)]`.
- `λ`-regularisation blends the learned utility with a linear scalarisation `λ · mean(r)` to maintain well-behaved gradients.
- Policies are saved as `.zip` files (Stable-Baselines3 format) under `experiments/<exp_name>/`.

**Phase 3 — Evaluation & Analysis**  
- `run_test.sh` calls `main_policy.py --test_only` to roll out each saved policy for 100 episodes and records the raw multi-objective return vectors.
- `plot_utility_returns.py` visualises the 2D return distributions of all policies as scatter plots.
- `stats.py` computes four evaluation metrics against baseline algorithms: **Expected Utility (EU)**, **CVaR**, **Constraint Satisfaction**, and **Variance Objective**.

---

### Key Components

#### Utility Functions (`MORL_stablebaselines3/utility_function/`)

| Class | File | Description |
|---|---|---|
| `Utility_Function_Parameterized` | `utility_function_parameterized.py` | 4-layer monotone MLP. Weights clamped to `[0, max_weight]`. Input normalised via `BatchNorm1d`. λ-blend with linear scalarisation. |
| `Utility_Function_Programmed` | `utility_function_programmed.py` | Analytic utilities: uniform mean + per-dimension emphasis weights. |
| `Utility_Function_Linear` | `utility_function_programmed.py` | 13 fixed linear weight vectors covering the full simplex (for ablation / baselines). |
| `Utility_Function_Diverse_Goal` | `utility_function_programmed.py` | Six hand-crafted non-linear utilities for the DiverseGoal environment (sigmoid-based). |

#### Environment Wrappers (`MORL_stablebaselines3/envs/wrappers/`)

| Class | File | Description |
|---|---|---|
| `ObsInfoWrapper` | `utility_env_wrapper.py` | Gymnasium→Gym bridge. Accumulates per-step rewards into episode return `zₜ`. Reports `episode.r` (vector) in `info`. |
| `MultiEnv_UtilityFunction` | `utility_env_wrapper.py` | `VecEnvWrapper` that converts vector rewards to scalar utility via `U(zₜ)−U(zₜ₋₁)`. Optionally augments observations with normalised cumulative return. |
| `DummyVecEnv` | `utils.py` | Multi-objective-aware vectorised environment that stores `buf_rews` with shape `(num_envs, reward_dim)`. |

#### Custom Environment (`DIPG/diverse_goal_env.py`)

`DiverseGoalEnv` — a 2D continuous grid with four goal regions, each having a different stochastic reward distribution (Gaussian with different means/covariances). Designed to test distributional MORL: policies must learn to reach specific goal regions to satisfy specific utility functions.

---

### Supported Environments

| Domain | Environment | MO-Gymnasium ID |
|---|---|---|
| Classic Control | MountainCar | `mo-mountaincarcontinuous-v0` |
| Grid Worlds | DeepSeaTreasure | `deep-sea-treasure-v0` |
| | FruitTree | `fruit-tree-v0` |
| | FourRoom | `four-room-v0` |
| | BreakableBottles | `breakable-bottles-v0` |
| | FishWood | `fishwood-v0` |
| | ResourceGathering | `resource-gathering-v0` |
| | DiverseGoal | custom (`DIPG/`) |
| Continuous Control | HalfCheetah | `mo-halfcheetah-v4` |
| | Hopper | `mo-hopper-v4` |
| | Reacher | `mo-reacher-v4` |
| | ReacherBullet | `mo-reacher-v0` |
| Other | Highway | `mo-highway-v0` |
| | LunarLander | `mo-lunar-lander-v2` |
| | SuperMarioBros | `mo-supermario-v0` |
| | WaterReservoir | `water-reservoir-v0` |
| | Minecart | `minecart-v0` |

---

### Evaluation Metrics (`stats.py`)

DPMORL is compared against **GPI-LS**, **GPI-PD**, **OLS**, and **PGMORL** using:

| Metric | Description |
|---|---|
| **Expected Utility (EU)** | Average best-policy utility under 101 uniformly-spaced linear weight vectors. |
| **CVaR** | Conditional Value at Risk (α=0.05) of the scalarised return distribution across weight vectors — measures tail-risk performance. |
| **Constraint Satisfaction** | Fraction of randomly sampled linear return constraints that are satisfiable by at least one policy. |
| **Variance Objective** | Mean–variance trade-off under random weight vectors combining mean return and return standard deviation. |

---

### Key Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--lamda` | `0.01` | λ blend between learned utility and linear scalarisation |
| `--env` | `MountainCar` | Environment name (see table above) |
| `--exp_name` | `dpmorl` | Experiment directory name under `experiments/` |
| `--reward_two_dim` | `False` | Restrict to first two reward dimensions |
| `--reward_dim_indices` | `''` | Explicit reward dimension indices (e.g. `[0,1,2]`) |
| `--total_timesteps` | `1e7` | PPO training budget per policy |
| `--iters` | `50` | Utility function training iterations |
| `--max_num_policies` | `20` | Maximum number of policies to train |
| `--num_envs` | `20` | Parallel environments for vectorised PPO rollout |
| `--utility_epochs` | `200` | Epochs per utility function training round |
| `--gpu` | `all` | GPU selection (auto-selects most free GPU) |

---

### Repository Structure

```
DPMORL/
├── main_generate_utility.py          # Generate & save diverse utility function models
├── main_policy.py                    # Train / evaluate a set of Pareto-optimal policies
├── plot_utility_returns.py           # Visualise 2D return distribution scatter plots
├── stats.py                          # Compute EU, CVaR, constraint satisfaction, variance metrics
├── utils.py                          # Multi-objective DummyVecEnv (stores buf_rews as reward_dim vector)
├── env.txt                           # List of environments used in the paper
├── run_policy_parallel.sh            # Train all environments in parallel (nohup)
├── run_test.sh                       # Evaluate all trained policies
├── requirements.txt                  # Python dependencies
├── MORL_stablebaselines3/
│   ├── common/
│   │   ├── argument_parser.py        # Shared argument parser utilities
│   │   ├── base_runner.py            # Base training-loop runner
│   │   ├── mpi_adam.py               # MPI-compatible Adam optimiser
│   │   ├── mpi_adam_optimizer.py     # MPI Adam optimiser (extended)
│   │   ├── tf_utils.py               # TensorFlow utility helpers
│   │   └── utils.py                  # Miscellaneous training utilities
│   ├── envs/
│   │   ├── utils.py                  # Array type alias & shared env utilities
│   │   ├── gridworlds/
│   │   │   ├── gridworld_base.py         # Base grid cell / object definitions
│   │   │   ├── mo_gridworld_base.py      # Multi-objective gridworld base class
│   │   │   ├── mo_deep_sea_treasure_env.py  # Custom DeepSeaTreasure grid (10×10 map)
│   │   │   ├── mo_gathering_env.py       # Resource-gathering grid environment
│   │   │   └── mo_traffic_env.py         # Multi-objective traffic grid environment
│   │   ├── mountain_car/
│   │   │   ├── mo_mountain_car.py        # Multi-objective mountain car extension
│   │   │   ├── mountain_car.py           # Base mountain car implementation
│   │   │   └── test.py                   # Mountain car environment tests
│   │   ├── pendula/
│   │   │   ├── double_pendulum.py        # MuJoCo double-pendulum (safety-constrained)
│   │   │   ├── single_pendulum.py        # MuJoCo single-pendulum (safety-constrained)
│   │   │   └── test.py                   # Pendulum environment tests
│   │   ├── reacher/
│   │   │   ├── reacher.py                # Custom Reacher environment
│   │   │   └── test.py                   # Reacher environment tests
│   │   ├── safety_gym/
│   │   │   └── augmented_sg_envs.py      # Safety-gym environment augmentations
│   │   └── wrappers/
│   │       ├── utility_env_wrapper.py    # ObsInfoWrapper + MultiEnv_UtilityFunction (core wrappers)
│   │       ├── morl_env_wrapper.py       # Generic MORL env decorator (class-based)
│   │       ├── morl_env.py               # morl_env class decorator (Utility_Function mixin)
│   │       ├── morl_env_torch.py         # morl_env_torch decorator (PyTorch utility)
│   │       ├── original_multi_rewards_env_torch.py  # Original multi-reward env decorator
│   │       ├── scalar_reward_wrapper.py  # Scalar reward scalarisation wrapper
│   │       ├── safe_env.py               # SafeEnv base class (adds safety cost to info)
│   │       └── saute_env.py              # SAUTE safety augmentation class decorator
│   └── utility_function/
│       ├── utility_function_parameterized.py  # Monotone neural-network utility (4-layer MLP)
│       └── utility_function_programmed.py     # Analytic, linear & DiverseGoal utility functions
├── DIPG/
│   └── diverse_goal_env.py           # Custom 2D grid with 4 stochastic goal regions
├── utility-model-selected/
│   └── dim-2/                        # 33 pre-trained utility function checkpoints (2D reward)
├── utility-plot-selected/
│   └── dim-2/                        # Contour plot visualisations of pre-trained utilities
└── normalization_data/
    └── data.pickle                   # Per-environment min/max return normalisation statistics
```

## Requirements

To install the environment (except for ReacherBullet), run: 

```
conda create -n dpmorl python=3.8
conda activate dpmorl
pip install -r requirements.txt
```

## Training Policies with DPMORL

### Generating Utility Functions

Before training policies, DPMORL requires first generate utility functions. To generate utility functions, run:

```
python main_generate_utility.py
```

The generated utility functions are saved in `utility-model/dim-2`, and the visualization are saved in `utility-plot/dim-2`. 

You can run 

```
python main_generate_utility.py --reward_shape 3 --num_utility_function 100
```

for configuring the reward dimensions and utility function number for generated utility functions.  

We have provided part of our generated utility functions in `utility-model-selected` and `utility-plot-selected`. 

### Training Policies

To policies by DPMORL in the paper, run this command:

```
python -u main_policy.py --lamda=0.1 --env [env] --reward_two_dim --exp_name [exp_name]
```

The environment name is in `env.txt`. 

Configuration `--reward_two_dim` makes DPMORL run on the first two dimensions of reward functions. To run DPMORL on other dimensions of reward (e.g. 0, 1, 2 dimension), you can change `--reward_two_dim` to `--reward_dim_indices=[0,1,2]`. 

You can also run `. run_policy_parallel.sh` to run DPMORL in all environments in parallel. 

### Evaluate the policies

After training finished, you should evaluate the policies learned by DPMORL by running `. run_test.sh`. 

## Visualize the return distributions of learned policies

To visualize the return distributions of policies learned by DPMORL, run `python plot_utility_returns.py [exp_name]`. The visualization results will be located in the `experiments/[exp_name]` directory. `test_final_*.png` will visualize the return distributions of all learned policies by DPMORL. 

## Compute the evaluation metric

Run `stats.py` to compute all the evaluation metrics for DPMORL and other baseline methods. The implemenetation includes constraint satisfaction and variance objective. 

## Portfolio Environment (GymFolio + FinRL) Integration

This repository includes a portfolio optimization adapter (`Portfolio`) that plugs GymFolio into the DPMORL loop.

### What was added

Code:

- `MORL_stablebaselines3/envs/portfolio/mo_portfolio_env.py`
  - wraps GymFolio environment as a 2-objective MORL environment
  - objective 1: portfolio return reward
  - objective 2: negative volatility proxy (`-risk_penalty`)
- `main_policy.py`
  - `--env Portfolio` support
  - portfolio CLI options:
    - `--portfolio_data_dir`
    - `--portfolio_rebalance_every`
    - `--portfolio_max_trajectory_len`
    - `--portfolio_lookback`
    - `--portfolio_risk_scale`
  - pretrained-only workflow:
    - `--pretrained_only`
    - `--num_pretrained_to_use`
  - resume/skip logic for already-trained policy checkpoints
- `scripts/prepare_portfolio_data.py`
  - downloads FinRL-style data and converts to GymFolio input (`df_ohlc.pkl`, `df_observations.pkl`)
- helper scripts:
  - `scripts/prepare_portfolio_expanded.sh`
  - `scripts/run_portfolio_expanded_train.sh`

### Utility files: `.pt` vs `.png`

- `utility-model-selected/dim-2/utility-*.pt` are the actual utility models used for training policies.
- `utility-plot-selected/dim-2/utility-*.png` are visualization-only contours of those utility functions.

### Install optional dependencies

```
pip install git+https://github.com/hsleejw21/gymfolio
pip install git+https://github.com/hsleejw21/FinRL
```

### Data preparation

Base example:

```
python scripts/prepare_portfolio_data.py \
	--start_date 2010-01-01 \
	--end_date 2024-01-01 \
	--tickers AAPL,MSFT,GOOG,AMZN,META \
	--output_dir data/portfolio
```

Expanded split preset:

```
./scripts/prepare_portfolio_expanded.sh
```

It creates `data/portfolio_expanded/{train,val,test,full}`.

### Reproducible experiment run used in this repository

#### Train (Option A)

- `exp_name=dpmorl_portfolio_a`
- `total_timesteps=1e6`
- `max_num_policies=20`
- `pretrained_only=True`
- `num_pretrained_to_use=20`
- train data: `data/portfolio_expanded/train`

Example:

```
python -u main_policy.py \
	--env Portfolio \
	--reward_two_dim \
	--lamda 0.1 \
	--exp_name dpmorl_portfolio_a \
	--max_num_policies 20 \
	--pretrained_only True \
	--num_pretrained_to_use 20 \
	--total_timesteps 1000000 \
	--num_envs 10 \
	--portfolio_data_dir data/portfolio_expanded/train \
	--portfolio_rebalance_every 5 \
	--portfolio_max_trajectory_len 252 \
	--portfolio_lookback 20 \
	--portfolio_risk_scale 1.0
```

#### Test/evaluation

Important: the number of tickers must match between train and test (observation shape consistency).

In this run, train used 20 tickers after alignment, so evaluation used:

- `data/portfolio_expanded/test_aligned`

with:

```
python -u main_policy.py \
	--env Portfolio \
	--reward_two_dim \
	--lamda 0.1 \
	--exp_name dpmorl_portfolio_a \
	--max_num_policies 20 \
	--pretrained_only True \
	--num_pretrained_to_use 20 \
	--test_only True \
	--num_test_episodes 100 \
	--portfolio_data_dir data/portfolio_expanded/test_aligned \
	--portfolio_rebalance_every 5 \
	--portfolio_max_trajectory_len 252 \
	--portfolio_lookback 20 \
	--portfolio_risk_scale 1.0
```

### Result files and how to read them

All outputs are under:

- `experiments/dpmorl_portfolio_a/DPMORL.Portfolio.LossNormLamda_0.1/`

Key files:

- `policy-pretrain-*.zip`: trained policies (20 total)
- `MORL_Portfolio_PPO_policypretrain-*_seed0_0.npz`: training episode vector returns
- `test_returns_policy_pretrain-*.npz`: test episode vector returns (100 episodes each)
- `test_final_batch_1.png`, `test_final_batch_10.png`: policy distribution on test set
- `Portfolio_final_batch_1.png`, `Portfolio_final_batch_10.png`: final training return distribution

Interpretation:

- x-axis (`Return 1`): return objective (higher is better)
- y-axis (`Return 2`): `-risk` objective (higher is better; means lower risk)

### Visualization

```
python plot_utility_returns.py dpmorl_portfolio_a
```

This generates final scatter summaries (`test_final_*`, `Portfolio_final_*`) and per-policy test scatter plots.
