import argparse
import io
import os

import pandas as pd
from stockstats import StockDataFrame as Sdf
import yfinance as yf

from MORL_stablebaselines3.envs.portfolio.mo_portfolio_env import build_gymfolio_data_from_finrl


def _download_yahoo_finrl_style(start_date: str, end_date: str, ticker_list):
    data_df = pd.DataFrame()
    num_failures = 0

    session = None
    try:
        from curl_cffi import requests as curl_requests

        session = curl_requests.Session()
        session.verify = False
    except Exception:
        session = None

    for tic in ticker_list:
        download_kwargs = dict(start=start_date, end=end_date, auto_adjust=False, progress=False)
        if session is not None:
            download_kwargs["session"] = session
        temp_df = yf.download(tic, **download_kwargs)

        if temp_df.empty:
            stooq_symbol = tic if "." in tic else f"{tic}.US"
            try:
                url = f"https://stooq.com/q/d/l/?s={stooq_symbol.lower()}&i=d"
                if session is not None:
                    response = session.get(url)
                    response.raise_for_status()
                    stooq_df = pd.read_csv(io.StringIO(response.text))
                else:
                    stooq_df = pd.read_csv(url)
                stooq_df["Date"] = pd.to_datetime(stooq_df["Date"])
                stooq_df = stooq_df[(stooq_df["Date"] >= pd.Timestamp(start_date)) & (stooq_df["Date"] < pd.Timestamp(end_date))]
                stooq_df = stooq_df.sort_values("Date").set_index("Date")
                temp_df = stooq_df.rename(columns={"Open": "Open", "High": "High", "Low": "Low", "Close": "Close", "Volume": "Volume"})
                temp_df["Adj Close"] = temp_df["Close"]
            except Exception as exc:
                print(f"Stooq fallback failed for {tic}: {exc}")
                temp_df = pd.DataFrame()

        temp_df["tic"] = tic
        if len(temp_df) > 0:
            data_df = pd.concat([data_df, temp_df], axis=0)
        else:
            num_failures += 1

    if num_failures == len(ticker_list):
        raise ValueError("no data is fetched.")

    data_df = data_df.reset_index()
    data_df.columns = ["date", "open", "high", "low", "close", "adjcp", "volume", "tic"]
    data_df["close"] = data_df["adjcp"]
    data_df = data_df.drop(labels="adjcp", axis=1)
    data_df["day"] = data_df["date"].dt.dayofweek
    data_df["date"] = data_df["date"].apply(lambda x: x.strftime("%Y-%m-%d"))
    data_df = data_df.dropna().sort_values(by=["date", "tic"]).reset_index(drop=True)

    return data_df


def _feature_engineer_lite(df: pd.DataFrame) -> pd.DataFrame:
    """Minimal FeatureEngineer equivalent for common indicators used in FinRL tutorials."""
    df = df.copy().sort_values(["date", "tic"], ignore_index=True)
    df.index = df.date.factorize()[0]

    merged_closes = df.pivot_table(index="date", columns="tic", values="close")
    merged_closes = merged_closes.dropna(axis=1)
    tics = merged_closes.columns
    df = df[df.tic.isin(tics)].copy()

    indicators = ["macd", "rsi_30", "cci_30", "dx_30"]
    df_by_tic = df.sort_values(by=["tic", "date"])
    stock = Sdf.retype(df_by_tic.copy())
    unique_ticker = stock.tic.unique()

    for indicator in indicators:
        indicator_df = pd.DataFrame()
        for tic in unique_ticker:
            try:
                temp_indicator = stock[stock.tic == tic][indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator["tic"] = tic
                temp_indicator["date"] = df_by_tic[df_by_tic.tic == tic]["date"].to_list()
                indicator_df = pd.concat([indicator_df, temp_indicator], axis=0, ignore_index=True)
            except Exception:
                continue

        if not indicator_df.empty:
            df = df.merge(indicator_df[["tic", "date", indicator]], on=["tic", "date"], how="left")

    close_by_tic = df.sort_values(["tic", "date"]) 
    close_by_tic["close_30_sma"] = close_by_tic.groupby("tic")["close"].transform(
        lambda s: s.rolling(window=30, min_periods=1).mean()
    )
    close_by_tic["close_60_sma"] = close_by_tic.groupby("tic")["close"].transform(
        lambda s: s.rolling(window=60, min_periods=1).mean()
    )
    df = close_by_tic.sort_values(["date", "tic"]).reset_index(drop=True)

    return df.ffill().bfill()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare gymfolio-ready portfolio data using FinRL downloader")
    parser.add_argument("--start_date", type=str, default="2010-01-01")
    parser.add_argument("--end_date", type=str, default="2024-01-01")
    parser.add_argument("--tickers", type=str, default="AAPL,MSFT,GOOG,AMZN,META")
    parser.add_argument("--output_dir", type=str, default="data/portfolio")
    return parser.parse_args()


def main():
    args = parse_args()
    tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]

    print(f"Downloading data for {tickers} from {args.start_date} to {args.end_date}...")
    df = _download_yahoo_finrl_style(args.start_date, args.end_date, tickers)

    print("Running feature engineering (FinRL-compatible subset)...")
    df = _feature_engineer_lite(df)

    # Keep only columns needed/likely useful for observations.
    keep_cols = [
        col
        for col in [
            "date",
            "tic",
            "open",
            "high",
            "low",
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
        if col in df.columns
    ]
    df = df[keep_cols].copy()

    print("Converting to gymfolio format...")
    df_ohlc, df_observations = build_gymfolio_data_from_finrl(df)

    os.makedirs(args.output_dir, exist_ok=True)
    ohlc_path = os.path.join(args.output_dir, "df_ohlc.pkl")
    obs_path = os.path.join(args.output_dir, "df_observations.pkl")
    raw_path = os.path.join(args.output_dir, "finrl_raw.csv")

    df.to_csv(raw_path, index=False)
    df_ohlc.to_pickle(ohlc_path)
    df_observations.to_pickle(obs_path)

    print("Saved:")
    print(f"- {raw_path}")
    print(f"- {ohlc_path} (shape={df_ohlc.shape})")
    print(f"- {obs_path} (shape={df_observations.shape})")


if __name__ == "__main__":
    main()
