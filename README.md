# DPMORL

Implementations for our paper [*Distributional Pareto-Optimal Multi-Objective Reinforcement Learning*](https://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html) at NeurIPS 2023. 

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
