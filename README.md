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

This repository now includes a portfolio optimization environment adapter (`Portfolio`) that plugs GymFolio into the same DPMORL training loop.

### 1) Install optional dependencies

```
pip install git+https://github.com/hsleejw21/gymfolio
pip install git+https://github.com/hsleejw21/FinRL
```

### 2) Prepare portfolio data via FinRL downloader

The script below downloads Yahoo Finance data, computes FinRL features, and converts them to GymFolio input format:

```
python scripts/prepare_portfolio_data.py \
	--start_date 2010-01-01 \
	--end_date 2024-01-01 \
	--tickers AAPL,MSFT,GOOG,AMZN,META \
	--output_dir data/portfolio
```

This creates:

- `data/portfolio/df_ohlc.pkl`
- `data/portfolio/df_observations.pkl`
- `data/portfolio/finrl_raw.csv`

### 3) Train DPMORL on portfolio environment

```
python -u main_policy.py \
	--env Portfolio \
	--reward_two_dim \
	--exp_name dpmorl_portfolio \
	--lamda 0.1 \
	--portfolio_data_dir data/portfolio \
	--portfolio_rebalance_every 5 \
	--portfolio_max_trajectory_len 252 \
	--portfolio_lookback 16 \
	--portfolio_risk_scale 1.0
```

Reward dimensions for `Portfolio` are:

- objective 1: portfolio return reward from GymFolio
- objective 2: negative volatility proxy (risk penalty)

### 4) Expanded experiment preset (more assets + split periods)

For a larger experiment setup, use these helper scripts:

```
./scripts/prepare_portfolio_expanded.sh
./scripts/run_portfolio_expanded_train.sh
```

`prepare_portfolio_expanded.sh` builds 4 datasets under `data/portfolio_expanded`:

- `train`: 2010-01-01 ~ 2019-12-31
- `val`: 2020-01-01 ~ 2021-12-31
- `test`: 2022-01-01 ~ 2024-12-31
- `full`: 2010-01-01 ~ 2024-12-31

By default it uses 30 large-cap US tickers and trains with:

- `portfolio_max_trajectory_len=252`
- `portfolio_lookback=20`
- `portfolio_rebalance_every=5`
- `total_timesteps=300000` (override with `TOTAL_TIMESTEPS=...`)
- `max_num_policies=16` (override with `MAX_POLICIES=...`)
