import os
import sys
from typing import Callable, List, Optional, Sequence, Tuple

import gymnasium
import numpy as np
import pandas as pd


def _try_import_gymfolio_env():
    """Import gymfolio PortfolioOptimizationEnv with flexible paths."""
    import importlib

    candidates = [
        "gymfolio.envs.base",
        "envs.base",  # when using editable source checkout with src on PYTHONPATH
    ]

    # Fallback: vendored gymfolio source in this repository
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, "..", "..", ".."))
    vendored_src = os.path.join(repo_root, "third_party", "gymfolio", "src")
    if os.path.isdir(vendored_src) and vendored_src not in sys.path:
        sys.path.insert(0, vendored_src)
    last_error = None
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "PortfolioOptimizationEnv")
        except Exception as exc:  # pragma: no cover - best effort import probing
            last_error = exc
    raise ImportError(
        "Failed to import gymfolio PortfolioOptimizationEnv. Install gymfolio first: "
        "pip install git+https://github.com/hsleejw21/gymfolio"
    ) from last_error


def _features_to_multiindex(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    """Convert long FinRL dataframe to gymfolio MultiIndex wide observation dataframe."""
    blocks = []
    for feature in feature_cols:
        pivot = df.pivot(index="date", columns="tic", values=feature).sort_index(axis=1)
        # Desired layout for gymfolio: columns as (Ticker, Feature)
        pivot.columns = pd.MultiIndex.from_tuples([(tic, feature) for tic in pivot.columns])
        blocks.append(pivot)
    wide = pd.concat(blocks, axis=1)
    wide = wide.sort_index(axis=1)
    wide.index = pd.to_datetime(wide.index)
    return wide.dropna()


def build_gymfolio_data_from_finrl(
    df_finrl: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convert FinRL-style long dataframe into gymfolio OHLC and observation dataframes.

    Expected FinRL columns: date, tic, open, high, low, close (+ optional indicators)
    Returns:
        df_ohlc: MultiIndex columns (tic, Open/High/Low/Close)
        df_observations: MultiIndex columns (tic, feature)
    """
    required = {"date", "tic", "open", "high", "low", "close"}
    missing = required.difference(df_finrl.columns)
    if missing:
        raise ValueError(f"FinRL dataframe missing required columns: {sorted(missing)}")

    df = df_finrl.copy()
    df["date"] = pd.to_datetime(df["date"])

    # OHLC in gymfolio format: MultiIndex (ticker, Field)
    ohlc_fields = [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close")]
    ohlc_blocks = []
    for src, dst in ohlc_fields:
        pivot = df.pivot(index="date", columns="tic", values=src).sort_index(axis=1)
        pivot.columns = pd.MultiIndex.from_tuples([(tic, dst) for tic in pivot.columns])
        ohlc_blocks.append(pivot)
    df_ohlc = pd.concat(ohlc_blocks, axis=1).sort_index(axis=1)
    df_ohlc.index = pd.to_datetime(df_ohlc.index)

    if feature_cols is None:
        # Default to close + common FinRL indicators when available.
        candidate = [
            "close",
            "volume",
            "macd",
            "rsi_30",
            "cci_30",
            "dx_30",
            "close_30_sma",
            "close_60_sma",
            "vix",
            "turbulence",
        ]
        feature_cols = [col for col in candidate if col in df.columns]
        if not feature_cols:
            feature_cols = ["close"]

    df_observations = _features_to_multiindex(df, feature_cols)

    # Align index intersection and remove NaNs introduced by indicators.
    common_index = df_ohlc.index.intersection(df_observations.index)
    df_ohlc = df_ohlc.loc[common_index].dropna()
    df_observations = df_observations.loc[df_ohlc.index].dropna()

    # Ensure strict index alignment
    common_index = df_ohlc.index.intersection(df_observations.index)
    df_ohlc = df_ohlc.loc[common_index]
    df_observations = df_observations.loc[common_index]

    if len(df_ohlc) == 0:
        raise ValueError("Converted portfolio data is empty after alignment/dropna.")

    return df_ohlc, df_observations


class MOFinancePortfolioEnv(gymnasium.Env):
    """
    Adapter that wraps gymfolio PortfolioOptimizationEnv into DPMORL-compatible
    multi-objective environment with 2D reward:

      reward_type='cvar_perstep'  (v4a)
        reward[0] = log_return_t
        reward[1] = -max(0, -r_t) / alpha   (CVaR per-step, VaR≈0 approx.)

      reward_type='sharpe_cvar'  (v4b)
        reward[0] = Differential Sharpe contribution (Moody et al. 1998)
                    dS_t = [B_{t-1}*r_t - 0.5*A_{t-1}*r_t^2] / (B_{t-1}-A_{t-1}^2)^1.5
                    where A,B are EMA of r and r^2.  Penalises variance in reward[0]
                    itself → genuine trade-off with CVaR in reward[1].
        reward[1] = -max(0, -r_t) / alpha   (same CVaR contribution as v4a)

      reward_type='rolling_cvar'
        reward[0] = log_return_t
        reward[1] = -CVaR over rolling buffer of episode returns so far.

      reward_type='variance'
        reward[0] = log_return_t
        reward[1] = -r_t^2   (Markowitz-style variance contribution)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_dir: str = "data/portfolio_expanded/train",
        ohlc_file: str = "df_ohlc.pkl",
        obs_file: str = "df_observations.pkl",
        rebalance_every: int = 5,
        max_trajectory_len: int = 252,
        observation_frame_lookback: int = 16,
        continuous_weights: bool = True,
        allow_short_positions: bool = False,
        slippage: float = 5e-4,
        transaction_costs: float = 2e-4,
        render_mode: str = "tile",
        agent_type: str = "continuous",
        risk_scale: float = 1.0,
        reward_type: str = "cvar_perstep",
    ):
        super().__init__()

        PortfolioOptimizationEnv = _try_import_gymfolio_env()

        ohlc_path = os.path.join(data_dir, ohlc_file)
        obs_path = os.path.join(data_dir, obs_file)
        if not os.path.exists(ohlc_path) or not os.path.exists(obs_path):
            raise FileNotFoundError(
                f"Portfolio data not found. Expected: {ohlc_path} and {obs_path}. "
                "Generate data first with scripts/prepare_portfolio_data.py"
            )

        df_ohlc = pd.read_pickle(ohlc_path)
        df_observations = pd.read_pickle(obs_path)

        self.base_env = PortfolioOptimizationEnv(
            df_ohlc=df_ohlc,
            df_observations=df_observations,
            rebalance_every=rebalance_every,
            slippage=slippage,
            transaction_costs=transaction_costs,
            continuous_weights=continuous_weights,
            allow_short_positions=allow_short_positions,
            max_trajectory_len=max_trajectory_len,
            observation_frame_lookback=observation_frame_lookback,
            render_mode=render_mode,
            agent_type=agent_type,
            convert_to_terminated_truncated=True,
        )

        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.reward_dim = 2
        self.risk_scale = risk_scale
        self.reward_type: str = reward_type
        self.cvar_alpha: float = 0.1    # CVaR tail probability (worst 10% steps)
        self._return_buf: list = []     # rolling buffer for rolling_cvar
        # Differential Sharpe EMA state (sharpe_cvar)
        self._sharpe_eta: float = 0.1  # EMA decay rate
        self._ema_r: float = 0.0       # EMA of returns
        self._ema_r2: float = 1e-6     # EMA of squared returns (init >0 to avoid /0)

    def _to_numpy_obs(self, obs):
        if hasattr(obs, "detach"):
            obs = obs.detach().cpu().numpy()
        elif not isinstance(obs, np.ndarray):
            obs = np.asarray(obs)
        return obs.astype(np.float32, copy=False)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        obs, info = self.base_env.reset(seed=seed if seed is not None else 5106, options=options)
        self._return_buf = []
        self._ema_r = 0.0
        self._ema_r2 = 1e-6
        return self._to_numpy_obs(obs), info

    def _compute_rewards(self, r: float):
        """
        Returns (r0, r1) pair based on reward_type.
        r0 = return signal, r1 = risk signal (before risk_scale).
        """
        if self.reward_type == "cvar_perstep":
            # reward[0]: log return
            # reward[1]: CVaR per-step contribution (VaR≈0 approx.)
            r1 = -max(0.0, -r) / self.cvar_alpha
            return r, r1

        elif self.reward_type == "sharpe_cvar":
            # reward[0]: Differential Sharpe contribution (Moody et al. 1998)
            #   dS_t = [B_{t-1}*r_t - 0.5*A_{t-1}*r_t^2] / (B_{t-1} - A_{t-1}^2)^1.5
            #   where A = EMA(r), B = EMA(r^2), eta=0.1
            #   Agents earning high return via volatile assets get penalised in r0
            #   because variance enters the denominator → genuine trade-off with CVaR.
            A = self._ema_r
            B = self._ema_r2
            var_est = max(B - A ** 2, 1e-8)
            d_sharpe = (B * r - 0.5 * A * r ** 2) / (var_est ** 1.5)
            # Clip to avoid extreme values at episode start
            d_sharpe = float(np.clip(d_sharpe, -100.0, 100.0))
            # Update EMAs for next step
            self._ema_r  = (1 - self._sharpe_eta) * A + self._sharpe_eta * r
            self._ema_r2 = (1 - self._sharpe_eta) * B + self._sharpe_eta * r ** 2
            # reward[1]: same CVaR contribution as v4a
            r1 = -max(0.0, -r) / self.cvar_alpha
            return d_sharpe, r1

        elif self.reward_type == "rolling_cvar":
            # reward[0]: log return
            # reward[1]: full CVaR over episode returns so far
            self._return_buf.append(r)
            buf = np.array(self._return_buf)
            if len(buf) < 2:
                r1 = -max(0.0, -r) / self.cvar_alpha
            else:
                var = float(np.percentile(buf, self.cvar_alpha * 100))
                shortfall = buf[buf <= var]
                cvar = float(shortfall.mean()) if len(shortfall) > 0 else var
                r1 = -abs(cvar)
            return r, r1

        elif self.reward_type == "variance":
            # reward[0]: log return
            # reward[1]: per-step variance contribution (Markowitz)
            return r, -(r ** 2)

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

    def step(self, action):
        obs, reward_scalar, truncated, terminated, info = self.base_env.step(action)

        r = float(reward_scalar)
        r0, r1 = self._compute_rewards(r)

        reward_vec = np.array([r0, self.risk_scale * r1], dtype=np.float32)

        info = dict(info)
        info["reward_return"] = r
        info["reward_r0"] = r0
        info["reward_r1"] = float(r1)

        return self._to_numpy_obs(obs), reward_vec, bool(terminated), bool(truncated), info

    def render(self):
        return self.base_env.render()

    def close(self):
        return self.base_env.close()


def make_portfolio_env_fn(
    data_dir: str,
    rebalance_every: int,
    max_trajectory_len: int,
    observation_frame_lookback: int,
    risk_scale: float,
    reward_type: str = "cvar_perstep",
) -> Callable[[], MOFinancePortfolioEnv]:
    """Factory to build a zero-arg constructor compatible with main_policy.make_env."""

    def _ctor():
        return MOFinancePortfolioEnv(
            data_dir=data_dir,
            rebalance_every=rebalance_every,
            max_trajectory_len=max_trajectory_len,
            observation_frame_lookback=observation_frame_lookback,
            risk_scale=risk_scale,
            reward_type=reward_type,
        )

    return _ctor
