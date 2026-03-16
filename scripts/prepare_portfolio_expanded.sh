#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHONPATH="$ROOT_DIR"

# 30 large-cap US tickers (edit freely)
TICKERS="AAPL,MSFT,GOOG,AMZN,META,NVDA,TSLA,BRK-B,UNH,JNJ,JPM,V,XOM,WMT,PG,MA,HD,CVX,ABBV,KO,PEP,MRK,BAC,PFE,AVGO,COST,TMO,DIS,ADBE,CRM"

mkdir -p "$ROOT_DIR/data/portfolio_expanded/train" \
         "$ROOT_DIR/data/portfolio_expanded/val" \
         "$ROOT_DIR/data/portfolio_expanded/test" \
         "$ROOT_DIR/data/portfolio_expanded/full"

echo "[1/4] train split: 2010-01-01 ~ 2019-12-31"
PYTHONPATH="$PYTHONPATH" conda run -n dpmorl python "$ROOT_DIR/scripts/prepare_portfolio_data.py" \
  --start_date 2010-01-01 \
  --end_date 2019-12-31 \
  --tickers "$TICKERS" \
  --output_dir "$ROOT_DIR/data/portfolio_expanded/train"

echo "[2/4] val split: 2020-01-01 ~ 2021-12-31"
PYTHONPATH="$PYTHONPATH" conda run -n dpmorl python "$ROOT_DIR/scripts/prepare_portfolio_data.py" \
  --start_date 2020-01-01 \
  --end_date 2021-12-31 \
  --tickers "$TICKERS" \
  --output_dir "$ROOT_DIR/data/portfolio_expanded/val"

echo "[3/4] test split: 2022-01-01 ~ 2024-12-31"
PYTHONPATH="$PYTHONPATH" conda run -n dpmorl python "$ROOT_DIR/scripts/prepare_portfolio_data.py" \
  --start_date 2022-01-01 \
  --end_date 2024-12-31 \
  --tickers "$TICKERS" \
  --output_dir "$ROOT_DIR/data/portfolio_expanded/test"

echo "[4/4] full set: 2010-01-01 ~ 2024-12-31"
PYTHONPATH="$PYTHONPATH" conda run -n dpmorl python "$ROOT_DIR/scripts/prepare_portfolio_data.py" \
  --start_date 2010-01-01 \
  --end_date 2024-12-31 \
  --tickers "$TICKERS" \
  --output_dir "$ROOT_DIR/data/portfolio_expanded/full"

echo "Done. Data saved under $ROOT_DIR/data/portfolio_expanded"
