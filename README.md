# DPMORL

Implementations for our paper [*Distributional Pareto-Optimal Multi-Objective Reinforcement Learning*](https://papers.nips.cc/paper_files/paper/2023/hash/32285dd184dbfc33cb2d1f0db53c23c5-Abstract-Conference.html) at NeurIPS 2023. 

## Summary

**DPMORL** (Distributional Pareto-Optimal Multi-Objective Reinforcement Learning) is a framework for training a set of policies that collectively approximate the Pareto front in multi-objective reinforcement learning (MORL) settings. Rather than optimising a single scalar reward, DPMORL finds a diverse set of policies, each excelling under a different trade-off between objectives, while simultaneously reasoning about the *distribution* of returns rather than just their expectation.

### Problem Setting

Many real-world decision-making problems involve several conflicting objectives (e.g., speed vs. energy efficiency, reward vs. safety). Classical RL collapses these into a single scalar, which requires the designer to fix trade-off weights up front. MORL instead produces a *Pareto-optimal* set of policies — a frontier where improving one objective necessarily worsens another — allowing a decision-maker to pick the most suitable policy after training.

### Key Ideas

| Concept | Description |
|---|---|
| **Distributional MORL** | Policies are evaluated using the full *distribution* of cumulative returns (not just the mean), capturing risk and variance. |
| **Learned Utility Functions** | Nonlinear monotone neural networks map a multi-dimensional return vector to a scalar utility. Multiple diverse utility functions are generated before training to cover different preference regions of the Pareto front. |
| **Monotone MLP** | The utility network enforces monotonicity by keeping all weights non-negative, guaranteeing that higher rewards on any objective never decrease the utility score. |
| **Policy-Utility Co-optimisation** | Each policy is paired with one utility function and trained via PPO (from Stable-Baselines3). Policies are trained sequentially, and each new policy is encouraged to produce a return distribution *different* from those already found (diversification). |
| **λ-regularised Utility** | A hyperparameter `λ` blends the learned nonlinear utility with a simple linear scalarisation, maintaining gradient quality during early training. |

### Training Pipeline

```
1. Generate utility functions     →  python main_generate_utility.py
2. Train policies                 →  python -u main_policy.py --lamda=0.1 --env <env> --reward_two_dim --exp_name <name>
3. Evaluate trained policies      →  . run_test.sh
4. Visualise return distributions →  python plot_utility_returns.py <exp_name>
5. Compute evaluation metrics     →  python stats.py
```

### Supported Environments

DPMORL has been tested on the following multi-objective environments:

| Domain | Environments |
|---|---|
| Classic Control | MountainCar |
| Grid Worlds | DeepSeaTreasure, FruitTree, FourRoom, BreakableBottles, FishWood, ResourceGathering, DiverseGoal |
| Continuous Control (MuJoCo) | HalfCheetah, Hopper, Reacher, ReacherBullet |
| Other | Highway, LunarLander, SuperMarioBros, WaterReservoir, Minecart |

### Repository Structure

```
DPMORL/
├── main_generate_utility.py       # Step 1 – generate & save utility function models
├── main_policy.py                 # Step 2 – train a set of Pareto-optimal policies
├── plot_utility_returns.py        # Step 4 – visualise return distributions
├── stats.py                       # Step 5 – compute evaluation metrics
├── run_policy_parallel.sh         # Helper to train all environments in parallel
├── run_test.sh                    # Helper to evaluate all trained policies
├── MORL_stablebaselines3/
│   ├── common/                    # PPO algorithm and training loop extensions
│   ├── envs/                      # Environment wrappers & custom environments
│   └── utility_function/
│       ├── utility_function_parameterized.py  # Monotone neural-network utility
│       └── utility_function_programmed.py     # Hand-crafted & linear utility functions
├── DIPG/                          # DiverseGoal environment implementation
├── utility-model-selected/        # Pre-generated utility function checkpoints
├── utility-plot-selected/         # Visualisations of the pre-generated utilities
└── normalization_data/            # Per-environment reward normalisation statistics
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
