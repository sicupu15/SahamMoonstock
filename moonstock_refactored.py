# -*- coding: utf-8 -*-
"""
Moonstock Daily Scanner — Production Ready
Fixed all critical issues + optimizations
"""

from __future__ import annotations
import pandas as pd, numpy as np, datetime as dt, json
import yfinance as yf
from pathlib import Path
import argparse, os, sys, logging, threading, queue, time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    HAS_TK = True
except Exception:
    HAS_TK = False

# ====== Configuration ======
@dataclass
class Config:
    RVOL_LEN: int = 15
    RVOL_THRESH: float = 1.6
    CALM_MIN_PCT: float = 0.5
    CALM_MAX_PCT: float = 3.5
    BASE_WINDOW: int = 4
    MIN_CONSEC: int = 2
    QA_NEAR_MIN: float = 0.5
    QA_NEAR_MAX: float = 3.0
    QA_MAX_GAP_PCT: float = 3.0
    SR_LOOK: int = 5
    SR_MIN_RVOL: float = 1.6
    MOM_MAX_GAP: float = 4.0
    PROF_LOOKBACK: int = 7
    RALLY_THR_PCT: float = 10.0
    NEAR_BASE_THR: float = 3.0
    LIQ_MIN_VALUE_IDR: float = 200_000_000
    LABEL_TARGET_PCT: float = 0.10
    LABEL_H_MAX: int = 20
    
    def update(self, params: Dict):
        for k, v in params.items():
            if hasattr(self, k):
                setattr(self, k, v)

DEFAULT_CONFIG = Config()

# ====== Logging ======
def setup_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"moonstock_{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ====== Cross-platform Paths ======
def get_default_dirs() -> Tuple[Path, Path]:
    home = Path.home()
    onedrive = home / "OneDrive" / "Moonstock" / "Scanner & Data Builder"
    base = onedrive if onedrive.exists() else home / "Documents" / "Moonstock_Exports" / "Screener Harian"
    base.mkdir(parents=True, exist_ok=True)
    
    sym_candidates = [base.parent / "symbols.csv", base / "symbols.csv", Path.cwd() / "symbols.csv"]
    sym_path = next((p for p in sym_candidates if p.exists()), base / "symbols.csv")
    
    return base, sym_path

DEFAULT_OUTPUT_DIR, DEFAULT_SYMBOLS = get_default_dirs()
TODAY_JKT = dt.datetime.now(dt.timezone(dt.timedelta(hours=7))).date()

# ====== Safe Math (Optimized) ======
def safe_divide_scalar(num: float, denom: float, fill=np.nan) -> float:
    """Optimized scalar division"""
    if denom == 0 or np.isnan(denom) or np.isnan(num):
        return fill
    result = num / denom
    return fill if np.isinf(result) else result

def safe_divide(num: pd.Series, denom: pd.Series, fill=np.nan) -> pd.Series:
    """Safe vectorized division"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num / denom.replace(0, np.nan)
        result = result.replace([np.inf, -np.inf], np.nan)
    return result.fillna(fill)

# ====== Validation ======
def validate_dataframe(df: pd.DataFrame, required_cols: List[str], min_rows: int = 1) -> bool:
    """Validate DataFrame quality"""
    if df.empty or len(df) < min_rows:
        return False
    if not all(col in df.columns for col in required_cols):
        return False
    # Check for bad data
    if 'Close' in df.columns and (df['Close'] <= 0).any():
        return False
    return True

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate OHLCV data"""
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    
    # Remove bad data
    df = df.dropna(subset=["Close", "Volume"])
    df = df[df["Close"] > 0]
    df = df[df["Volume"] >= 0]
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    return df

# ====== Core Calculations (Optimized) ======
def calc_rvol(vol: pd.Series, lb: int) -> pd.Series:
    if len(vol) < lb:
        return pd.Series(0, index=vol.index)
    rolling_mean = vol.rolling(lb, min_periods=lb).mean()
    return safe_divide(vol, rolling_mean, fill=0)

def calc_calm_window(close: pd.Series, win: int) -> pd.Series:
    if len(close) < win:
        return pd.Series(0, index=close.index)
    shifted = close.shift(win)
    return safe_divide(close - shifted, shifted, fill=0) * 100.0

def calc_hvn_proxy(df: pd.DataFrame, lookback: int) -> Tuple[pd.Series, pd.Series]:
    v = df["Volume"].astype("float64")
    mid = (df["High"].astype("float64") + df["Low"].astype("float64")) / 2.0
    winVol = v.rolling(lookback, min_periods=1).max()
    hvn_mid = mid.where(v == winVol, np.nan).ffill()
    ma_window = min(5, max(2, lookback))
    hvn_ma = hvn_mid.rolling(ma_window, min_periods=1).mean()
    return hvn_mid, (hvn_mid > hvn_ma)

# ====== Fundamentals (with caching) ======
_FUNDAMENTALS_CACHE = {}

def fetch_fundamentals_batch(tickers: List[str], logger: logging.Logger) -> Dict[str, Tuple[float, float]]:
    """Batch fetch fundamentals with caching"""
    results = {}
    to_fetch = [t for t in tickers if t not in _FUNDAMENTALS_CACHE]
    
    logger.info(f"Fetching fundamentals for {len(to_fetch)} tickers...")
    
    for ticker in to_fetch:
        try:
            tk = yf.Ticker(ticker)
            fi = getattr(tk, "fast_info", None)
            if fi:
                mc = getattr(fi, "market_cap", None)
                sh = getattr(fi, "shares", None) or getattr(fi, "shares_outstanding", None)
                if mc is None and sh:
                    px = getattr(fi, "last_price", None)
                    if px:
                        mc = float(sh) * float(px)
                _FUNDAMENTALS_CACHE[ticker] = (
                    float(mc) if mc else np.nan,
                    float(sh) if sh else np.nan
                )
            else:
                _FUNDAMENTALS_CACHE[ticker] = (np.nan, np.nan)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            logger.warning(f"Fundamentals error for {ticker}: {e}")
            _FUNDAMENTALS_CACHE[ticker] = (np.nan, np.nan)
    
    return {t: _FUNDAMENTALS_CACHE.get(t, (np.nan, np.nan)) for t in tickers}

# ====== Main Computation (Optimized) ======
def compute_one(df_in: pd.DataFrame, config: Config, logger: logging.Logger) -> pd.DataFrame:
    """Compute features for one ticker"""
    df = clean_ohlcv(df_in)
    
    if not validate_dataframe(df, ["Open", "High", "Low", "Close", "Volume"], 
                              min_rows=config.BASE_WINDOW + config.RVOL_LEN):
        return pd.DataFrame()

    # Basic indicators
    df["rvol"] = calc_rvol(df["Volume"], config.RVOL_LEN)
    df["winChgPct"] = calc_calm_window(df["Close"], config.BASE_WINDOW)
    df["calmRise"] = df["winChgPct"].between(config.CALM_MIN_PCT, config.CALM_MAX_PCT)

    # Consecutive RVOL
    consec = []
    count = 0
    for meets in (df["rvol"] >= config.RVOL_THRESH):
        count = count + 1 if meets else 0
        consec.append(count)
    df["consecRvol"] = consec

    # VWAP components
    cumPV = (((df["High"] + df["Low"] + df["Close"]) / 3.0) * df["Volume"]).cumsum()
    cumV = df["Volume"].cumsum()

    # Base detection
    base_active = False
    base_start = None
    current_lo = current_hi = np.nan
    
    df["baseActive"] = False
    for col in ["lastFinalLeft", "lastFinalRight", "lastFinalLow", "lastFinalHigh", "lastFinalClose"]:
        df[col] = np.nan

    anchorPV = pd.Series(np.nan, index=df.index)
    anchorV = pd.Series(np.nan, index=df.index)

    for i in range(len(df)):
        cond = (df["rvol"].iloc[i] >= config.RVOL_THRESH) and bool(df["calmRise"].iloc[i])
        
        if not base_active:
            if (df["consecRvol"].iloc[i] >= config.MIN_CONSEC) and bool(df["calmRise"].iloc[i]):
                base_active = True
                base_start = max(0, i - config.MIN_CONSEC + 1)
                win = df.iloc[base_start:i+1]
                current_lo = float(win["Low"].min())
                current_hi = float(win["High"].max())
                anchorPV.iloc[i] = cumPV.iloc[base_start-1] if base_start > 0 else 0.0
                anchorV.iloc[i] = cumV.iloc[base_start-1] if base_start > 0 else 0.0
        else:
            if cond:
                current_lo = min(current_lo, float(df["Low"].iloc[i]))
                current_hi = max(current_hi, float(df["High"].iloc[i]))
            else:
                if i > 0:
                    df.iloc[i-1, df.columns.get_loc("lastFinalLeft")] = float(base_start)
                    df.iloc[i-1, df.columns.get_loc("lastFinalRight")] = float(i-1)
                    df.iloc[i-1, df.columns.get_loc("lastFinalLow")] = float(current_lo)
                    df.iloc[i-1, df.columns.get_loc("lastFinalHigh")] = float(current_hi)
                    df.iloc[i-1, df.columns.get_loc("lastFinalClose")] = float(df["Close"].iloc[i-1])
                base_active = False
                base_start = None
                current_lo = current_hi = np.nan
        
        df.iloc[i, df.columns.get_loc("baseActive")] = base_active
        if i > 0 and pd.isna(anchorPV.iloc[i]) and pd.notna(anchorPV.iloc[i-1]):
            anchorPV.iloc[i] = anchorPV.iloc[i-1]
            anchorV.iloc[i] = anchorV.iloc[i-1]
    
    # Forward fill base info
    base_cols = ["lastFinalLeft", "lastFinalRight", "lastFinalLow", "lastFinalHigh", "lastFinalClose"]
    df[base_cols] = df[base_cols].ffill()

    # AVWAP
    anchorPV = anchorPV.ffill().fillna(0)
    anchorV = anchorV.ffill().fillna(0)
    df["avwap"] = safe_divide(cumPV - anchorPV, cumV - anchorV)
    df["distAvwapPct"] = safe_divide(df["Close"] - df["avwap"], df["avwap"]) * 100.0

    # Run-up metrics (last row only)
    if not df.empty and pd.notna(df["lastFinalRight"].iloc[-1]):
        last_idx = len(df) - 1
        start_idx = int(df["lastFinalRight"].iloc[-1])
        start_idx = max(0, min(start_idx, last_idx))
        seg = df.iloc[start_idx:]
        if len(seg) > 0:
            hi = float(seg["High"].max())
            df.loc[df.index[-1], "hiSince"] = hi
            lc = df["lastFinalClose"].iloc[-1]
            if pd.notna(lc) and lc > 0:
                df.loc[df.index[-1], "runUpPct"] = safe_divide_scalar(hi - lc, lc) * 100.0
                df.loc[df.index[-1], "pullbackPct"] = safe_divide_scalar(df["Close"].iloc[-1] - hi, hi) * 100.0

    # Regime classification
    df["isNear"] = df["distAvwapPct"].abs().between(config.QA_NEAR_MIN, config.QA_NEAR_MAX)
    df["quietAcc"] = df["isNear"] & (df["rvol"] >= config.RVOL_THRESH) & df["calmRise"]
    
    prev_max = df["High"].shift(1).rolling(config.SR_LOOK, min_periods=config.SR_LOOK).max()
    df["breakMinor"] = df["High"] >= prev_max
    bodyPct = safe_divide(df["Close"] - df["Open"], df["Open"]) * 100.0
    df["swingReacc"] = (~df["quietAcc"]) & (df["distAvwapPct"] > 0) & df["breakMinor"] & (df["rvol"] >= config.SR_MIN_RVOL)
    df["momentum"] = (df["distAvwapPct"] > 0) & (df["rvol"] >= config.SR_MIN_RVOL * 1.2) & (bodyPct >= config.MOM_MAX_GAP)
    
    df["regime"] = "—"
    df.loc[df["quietAcc"], "regime"] = "Quiet Accumulation"
    df.loc[df["swingReacc"], "regime"] = "Swing Re-accumulation"
    df.loc[df["momentum"], "regime"] = "Momentum (Combo)"

    # Bar patterns
    spread = df["High"] - df["Low"]
    spreadMA = spread.rolling(10).mean()
    volMA = df["Volume"].rolling(10).mean()
    df["wideUp"] = (df["Close"] > df["Open"]) & (spread > spreadMA) & (df["Volume"] > volMA) & (df["Close"] > df["Low"] + spread * 0.66)
    df["upCloseLow"] = (df["Close"] > df["Open"]) & (df["Volume"] > volMA) & (df["Close"] < df["Low"] + spread * 0.5)

    # HVN
    hvn_mid, hvn_up = calc_hvn_proxy(df, config.PROF_LOOKBACK)
    df["hvn_mid"] = hvn_mid
    df["hvn_up"] = hvn_up
    df["value"] = df["Close"] * df["Volume"]
    
    return df

# ====== Features at t0 (Optimized) ======
def build_features_t0(df_raw: pd.DataFrame, idx: int, df_ihsg: Optional[pd.DataFrame], config: Config) -> Dict[str, float]:
    """Build features without leakage - optimized"""
    if idx < 60 or idx >= len(df_raw):
        return {}
    
    feats = {}
    close_t0 = df_raw["Close"].iloc[idx]
    if close_t0 <= 0:
        return {}
    
    # ATR
    tr = (df_raw["High"] - df_raw["Low"]).abs()
    atr14 = tr.rolling(14).mean().iloc[idx]
    feats["atr14_pct"] = safe_divide_scalar(atr14, close_t0)
    
    # Volatility
    ret = df_raw["Close"].pct_change()
    feats["volatility20"] = float(ret.rolling(20).std().iloc[idx])
    
    # Donchian
    high20 = df_raw["High"].rolling(20).max().iloc[idx]
    low20 = df_raw["Low"].rolling(20).min().iloc[idx]
    feats["donchian20_pos"] = safe_divide_scalar(close_t0 - low20, high20 - low20)
    
    # MA gaps
    ma5 = df_raw["Close"].rolling(5).mean().iloc[idx]
    ma20 = df_raw["Close"].rolling(20).mean().iloc[idx]
    feats["ma5_gap"] = safe_divide_scalar(close_t0 - ma5, close_t0)
    feats["ma20_gap"] = safe_divide_scalar(close_t0 - ma20, close_t0)
    
    # RVOL
    vol_t0 = df_raw["Volume"].iloc[idx]
    feats["rvol10"] = safe_divide_scalar(vol_t0, df_raw["Volume"].rolling(10).mean().iloc[idx])
    feats["rvol20"] = safe_divide_scalar(vol_t0, df_raw["Volume"].rolling(20).mean().iloc[idx])
    
    # Value metrics
    feats["med_value20"] = float((df_raw["Close"] * df_raw["Volume"]).rolling(20).median().iloc[idx])
    hl_spread = safe_divide(df_raw["High"] - df_raw["Low"], df_raw["Close"])
    feats["hl_spread20_pct"] = float(hl_spread.rolling(20).median().iloc[idx] * 100)
    
    # Returns
    close_series = df_raw["Close"]
    feats["ret_1d"] = float(ret.iloc[idx]) if idx > 0 else np.nan
    feats["ret_5d"] = safe_divide_scalar(close_series.iloc[idx] - close_series.iloc[idx-5], close_series.iloc[idx-5]) if idx >= 5 else np.nan
    feats["ret_20d"] = safe_divide_scalar(close_series.iloc[idx] - close_series.iloc[idx-20], close_series.iloc[idx-20]) if idx >= 20 else np.nan
    
    # RSI
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = safe_divide(gain, loss)
    feats["rsi_14"] = float((100 - (100 / (1 + rs))).iloc[idx])
    
    # Bollinger Bands
    ma20_bb = close_series.rolling(20).mean()
    std20 = close_series.rolling(20).std()
    bb_up = ma20_bb + (2 * std20)
    bb_low = ma20_bb - (2 * std20)
    feats["bb_pct_b"] = float(safe_divide(close_series - bb_low, bb_up - bb_low).iloc[idx])
    
    # IHSG context
    if df_ihsg is not None and not df_ihsg.empty and isinstance(df_ihsg.index, pd.DatetimeIndex):
        try:
            ihsg_sync = df_ihsg.reindex(df_raw.index, method='ffill')
            ihsg_close = ihsg_sync["Close"]
            ihsg_ret = ihsg_close.pct_change()
            ihsg_t0 = ihsg_close.iloc[idx]
            
            feats["ihsg_ret_20d"] = safe_divide_scalar(ihsg_close.iloc[idx] - ihsg_close.iloc[idx-20], ihsg_close.iloc[idx-20]) if idx >= 20 else np.nan
            feats["ihsg_volatility20"] = float(ihsg_ret.rolling(20).std().iloc[idx])
            ihsg_ma20 = ihsg_close.rolling(20).mean().iloc[idx]
            feats["ihsg_ma20_gap"] = safe_divide_scalar(ihsg_t0 - ihsg_ma20, ihsg_t0)
        except:
            feats.update({"ihsg_ret_20d": np.nan, "ihsg_volatility20": np.nan, "ihsg_ma20_gap": np.nan})
    else:
        feats.update({"ihsg_ret_20d": np.nan, "ihsg_volatility20": np.nan, "ihsg_ma20_gap": np.nan})
    
    return feats

# ====== Labels ======
def make_labels(df_adj: pd.DataFrame, t0_ts: pd.Timestamp, config: Config) -> Dict:
    """Create forward labels"""
    default = {"dir_label": np.nan, "ret_max_fwd": np.nan, "ret_min_fwd": np.nan, "days_to_hit": np.nan, "hit_side": "NONE"}
    
    try:
        if not isinstance(df_adj.index, pd.DatetimeIndex):
            return default
        
        if t0_ts not in df_adj.index:
            matches = np.where(df_adj.index.date == t0_ts.date())[0]
            if len(matches) == 0:
                return default
            t0_loc = matches[0]
        else:
            t0_loc = df_adj.index.get_loc(t0_ts)
        
        fwd = df_adj.iloc[t0_loc + 1 : t0_loc + 1 + config.LABEL_H_MAX]
        if fwd.empty:
            return default
        
        p0 = df_adj["Close"].iloc[t0_loc]
        if p0 <= 0:
            return default
        
        returns_high = fwd["High"].div(p0).sub(1.0)
        hit_idx = np.where(returns_high.values >= config.LABEL_TARGET_PCT)[0]
        
        if len(hit_idx) > 0:
            return {
                "dir_label": 1,
                "ret_max_fwd": float(fwd["High"].max() / p0 - 1.0),
                "ret_min_fwd": float(fwd["Low"].min() / p0 - 1.0),
                "days_to_hit": int(hit_idx[0] + 1),
                "hit_side": "UP"
            }
        else:
            return {
                "dir_label": 0,
                "ret_max_fwd": float(fwd["High"].max() / p0 - 1.0),
                "ret_min_fwd": float(fwd["Low"].min() / p0 - 1.0),
                "days_to_hit": np.nan,
                "hit_side": "NONE"
            }
    except Exception:
        return default

# ====== Helper Functions ======
def detect_base_close_events(df: pd.DataFrame) -> List[int]:
    if "baseActive" not in df.columns:
        return []
    ba = df["baseActive"].fillna(False).astype(bool).to_numpy()
    sh = np.roll(ba, 1)
    sh[0] = True
    ends = np.flatnonzero(sh & (~ba))
    return sorted(set([int(e-1) for e in ends if e-1 >= 0]))

def to_long_panel(df_multi: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if not isinstance(df_multi.columns, pd.MultiIndex):
        return pd.DataFrame()
    out = []
    for t in tickers:
        try:
            sub = df_multi.xs(t, axis=1, level=0).dropna(how="all")
            if not sub.empty:
                sub = sub.reset_index().rename(columns={"Date": "date", "index": "date"})
                sub["ticker"] = t
                out.append(sub)
        except:
            continue
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

# ====== Main Scanner (with Progress Callback) ======
def run_scan(
    symbols_csv: Optional[str] = None,
    output_base: Optional[str] = None,
    out_date: Optional[dt.date] = None,
    params: Optional[Dict] = None,
    progress_callback: Optional[callable] = None
) -> Tuple[str, str, str]:
    """Main scan with progress tracking"""
    
    config = Config()
    if params:
        try:
            config.update(params)
        except Exception as e:
            raise ValueError(f"Invalid parameters: {e}")
    
    base = Path(output_base) if output_base else DEFAULT_OUTPUT_DIR
    day = out_date if out_date else TODAY_JKT
    
    log_dir = base / "logs"
    logger = setup_logger(log_dir)
    logger.info(f"=== Starting Moonstock Scan for {day} ===")

    year_dir = base / f"{day:%Y}"
    month_dir = year_dir / f"{day:%B}"
    month_dir.mkdir(parents=True, exist_ok=True)

    # Load symbols
    symp = Path(symbols_csv) if symbols_csv else DEFAULT_SYMBOLS
    if not symp.exists():
        raise FileNotFoundError(f"symbols.csv not found: {symp}")
    
    tickers = pd.read_csv(symp)["ticker"].dropna().unique().tolist()
    logger.info(f"Loaded {len(tickers)} tickers")
    
    if progress_callback:
        progress_callback(0, f"Downloading data for {len(tickers)} tickers...")

    # Download data
    all_tickers = tickers + ["^JKSE"]
    try:
        data_raw = yf.download(tickers=all_tickers, period="750d", interval="1d",
                               group_by="ticker", auto_adjust=False, progress=False, threads=True)
        data_adj = yf.download(tickers=all_tickers, period="750d", interval="1d",
                               group_by="ticker", auto_adjust=True, progress=False, threads=True)
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise RuntimeError(f"Failed to download data: {e}")

    # Extract IHSG
    multi_raw = isinstance(data_raw.columns, pd.MultiIndex)
    multi_adj = isinstance(data_adj.columns, pd.MultiIndex)
    
    df_ihsg_raw = pd.DataFrame()
    df_ihsg_adj = pd.DataFrame()
    
    if multi_raw and "^JKSE" in data_raw.columns.get_level_values(0):
        df_ihsg_raw = data_raw.xs("^JKSE", axis=1, level=0).dropna(how="all")
        data_raw = data_raw.drop(columns="^JKSE", level=0, errors="ignore")
    
    if multi_adj and "^JKSE" in data_adj.columns.get_level_values(0):
        df_ihsg_adj = data_adj.xs("^JKSE", axis=1, level=0).dropna(how="all")
        data_adj = data_adj.drop(columns="^JKSE", level=0, errors="ignore")

    # Batch fetch fundamentals
    if progress_callback:
        progress_callback(5, "Fetching fundamentals...")
    fundamentals = fetch_fundamentals_batch(tickers, logger)

    # Process tickers
    rows_scan = []
    events_rows = []
    rows_features_today = []
    
    total_tickers = len(tickers)
    for idx, t in enumerate(tickers):
        if progress_callback:
            pct = 5 + int((idx / total_tickers) * 90)
            progress_callback(pct, f"Processing {t} ({idx+1}/{total_tickers})...")
        
        try:
            df_raw = data_raw.xs(t, axis=1, level=0).dropna(how="all") if multi_raw else data_raw.copy()
            if df_raw.empty:
                continue
            
            df_adj = data_adj.xs(t, axis=1, level=0).dropna(how="all") if multi_adj else data_adj.copy()
            
            df = compute_one(df_raw, config, logger)
            if df.empty:
                continue

            # Fundamentals
            mktcap, shares = fundamentals.get(t, (np.nan, np.nan))
            
            # Liquidity metrics
            val_series = df["Close"] * df["Volume"]
            adv20 = val_series.rolling(20).mean().iloc[-1] if len(val_series) >= 20 else np.nan
            adv60 = val_series.rolling(60).mean().iloc[-1] if len(val_series) >= 60 else np.nan
            medv20 = val_series.rolling(20).median().iloc[-1] if len(val_series) >= 20 else np.nan
            pct_liq = (val_series.tail(20) >= config.LIQ_MIN_VALUE_IDR).mean() * 100 if len(val_series) >= 20 else np.nan
            
            turnover20 = np.nan
            if pd.notna(shares) and shares > 0:
                avg_vol20 = df_raw["Volume"].rolling(20).mean().iloc[-1] if len(df_raw) >= 20 else np.nan
                turnover20 = safe_divide_scalar(avg_vol20, shares) * 100
            
            # Amihud illiquidity
            ret = df_raw["Close"].pct_change()
            amihud = (ret.tail(20).abs() / val_series.tail(20)).replace([np.inf, -np.inf], np.nan).mean()
            amihud20 = amihud * 1e9 if pd.notna(amihud) else np.nan
            
            # HL spread
            hl_spread = safe_divide(df_raw["High"] - df_raw["Low"], df_raw["Close"])
            hl_spread20 = hl_spread.rolling(20).median().iloc[-1] * 100 if len(hl_spread) >= 20 else np.nan

            # Current bar summary
            d = df.loc[df.index[-1]]
            
            # Execution notes
            if d["regime"] == "Quiet Accumulation":
                gap_abs = round(d["Close"] * config.QA_MAX_GAP_PCT / 100, 2)
                eksekusi = f"Entry: close hari sinyal / open H+1 jika gap ≤ {gap_abs}"
            elif d["regime"] == "Swing Re-accumulation":
                br = df["High"].shift(1).rolling(config.SR_LOOK).max().iloc[-1]
                eksekusi = f"Entry: break minor-high {config.SR_LOOK} bar ({br:.2f}) dgn RVOL ≥ {config.SR_MIN_RVOL}"
            elif d["regime"] == "Momentum (Combo)":
                eksekusi = f"Entry kecil; target cepat; hindari gap > {config.MOM_MAX_GAP}%"
            else:
                eksekusi = "—"

            # Base info
            if pd.notna(d.get("lastFinalRight")):
                end_idx = int(d["lastFinalRight"])
                end_idx = max(0, min(end_idx, len(df)-1))
                base_date = df.index[end_idx].date()
                base_close = df["Close"].iloc[end_idx]
                hi_since = d.get("hiSince", np.nan)
                peak_idx = np.nan
                if pd.notna(hi_since):
                    seg = df.iloc[end_idx:]
                    if len(seg) > 0:
                        peak_idx = int(np.argmax(seg["High"].values))
                delta_today = safe_divide_scalar(d["Close"] - base_close, base_close) * 100
            else:
                base_date = base_close = hi_since = peak_idx = delta_today = np.nan

            # Info string
            calm_val = d.get("winChgPct", np.nan)
            avwap_ok = pd.notna(d.get("avwap")) and d["Close"] > d["avwap"]
            avwap_status = "B:OK" if avwap_ok else ("B:No" if pd.notna(d.get("avwap")) else "B:na")
            bar_flag = "ACC" if d.get("wideUp", False) else ("SUP?" if d.get("upCloseLow", False) else "-")
            hvn_val = d.get("hvn_mid", np.nan)
            
            info_str = f"RVOL {d['rvol']:.2f} | Calm({config.BASE_WINDOW}): {calm_val:.2f}% | AVWAP {avwap_status} | Bar {bar_flag} | HVN {hvn_val:.2f}"

            base_label = "aktif" if d.get("baseActive", False) else ("final" if pd.notna(d.get("lastFinalRight")) else "—")
            rally_flag = (d.get("runUpPct", np.nan) >= config.RALLY_THR_PCT) if pd.notna(d.get("runUpPct")) else False

            # Scan row
            rows_scan.append({
                "Ticker": t,
                "Regime": d["regime"],
                "Eksekusi": eksekusi,
                "Near_AVWAP_%": round(d.get("distAvwapPct", np.nan), 2),
                "Base_terakhir": base_label,
                "Tanggal_base_terakhir": base_date,
                "Close_pada_tutup_base": round(base_close, 4) if pd.notna(base_close) else np.nan,
                "Δ%_CloseBase_ke_Today": round(delta_today, 2) if pd.notna(delta_today) else np.nan,
                "High_sejak_base_tutup": round(hi_since, 4) if pd.notna(hi_since) else np.nan,
                "RunUp_%_sejak_base": round(d.get("runUpPct", np.nan), 2),
                "Hari_menuju_puncak": peak_idx,
                "Pullback_%_dari_puncak": round(d.get("pullbackPct", np.nan), 2),
                "Close_hari_ini": round(d["Close"], 4),
                "Info": info_str,
                "RallyFlag10%": rally_flag,
                "MarketCap_IDR": round(mktcap, 0) if pd.notna(mktcap) else np.nan,
                "Shares_Outstanding": round(shares, 0) if pd.notna(shares) else np.nan,
                "Value_Today_IDR": round(val_series.iloc[-1], 0) if len(val_series) else np.nan,
                "ADV20_IDR": round(adv20, 0) if pd.notna(adv20) else np.nan,
                "ADV60_IDR": round(adv60, 0) if pd.notna(adv60) else np.nan,
                "MedValue20_IDR": round(medv20, 0) if pd.notna(medv20) else np.nan,
                "MinValue_IDR": round(config.LIQ_MIN_VALUE_IDR, 0),
                "%DaysValue≥Min_20D": round(pct_liq, 1) if pd.notna(pct_liq) else np.nan,
                "Turnover20_%": round(turnover20, 3) if pd.notna(turnover20) else np.nan,
                "Amihud20_(×1e-9)": round(amihud20, 3) if pd.notna(amihud20) else np.nan,
                "HL_Spread20_%": round(hl_spread20, 2) if pd.notna(hl_spread20) else np.nan,
            })

            # Historical events
            t0_indices = detect_base_close_events(df)
            for t0_idx in t0_indices:
                if t0_idx < 60:
                    continue
                
                t0_ts = df.index[t0_idx]
                row_t0 = df.iloc[t0_idx]
                
                feats = build_features_t0(df_raw, t0_idx, df_ihsg_raw, config)
                if not feats:
                    continue
                
                labels = make_labels(df_adj, pd.Timestamp(t0_ts), config)
                if pd.isna(labels["dir_label"]):
                    continue

                # Rally flag for this event
                run_up_t0 = 0.0
                base_left = row_t0.get("lastFinalLeft", -1)
                if pd.notna(base_left) and base_left > 0:
                    prev_bases = detect_base_close_events(df.iloc[:int(base_left)])
                    if prev_bases:
                        prev_idx = prev_bases[-1]
                        prev_close = df.iloc[prev_idx]["Close"]
                        seg_high = df.iloc[prev_idx:t0_idx+1]["High"].max()
                        run_up_t0 = safe_divide_scalar(seg_high - prev_close, prev_close) * 100

                val20_med = (df_raw["Close"] * df_raw["Volume"]).rolling(20).median().iloc[t0_idx]
                hl_sp20 = safe_divide(df_raw["High"] - df_raw["Low"], df_raw["Close"]).rolling(20).median().iloc[t0_idx] * 100

                events_rows.append({
                    "t0_date": t0_ts.date(),
                    "ticker": t,
                    "close_t0": df_raw["Close"].iloc[t0_idx],
                    "open_t0": df_raw["Open"].iloc[t0_idx],
                    "high_t0": df_raw["High"].iloc[t0_idx],
                    "low_t0": df_raw["Low"].iloc[t0_idx],
                    "volume_t0": df_raw["Volume"].iloc[t0_idx],
                    "avwap_t0": row_t0.get("avwap", np.nan),
                    "near_avwap_pct": row_t0.get("distAvwapPct", np.nan),
                    "rvol_15": row_t0.get("rvol", np.nan),
                    "regime_t0": row_t0.get("regime", "—"),
                    "rallyflag10": (run_up_t0 >= config.RALLY_THR_PCT),
                    "runUpPct_at_t0": run_up_t0,
                    **feats,
                    "med_value20": val20_med,
                    "hl_spread20_pct": hl_sp20,
                    "ADV20_IDR": adv20,
                    "ADV60_IDR": adv60,
                    "MedValue20_IDR": medv20,
                    "%DaysValue≥Min_20D": pct_liq,
                    "Turnover20_%": turnover20,
                    "Amihud20_(×1e-9)": amihud20,
                    **labels,
                })

            # Today's features
            idx_today = len(df) - 1
            if idx_today >= 60:
                feats_today = build_features_t0(df_raw, idx_today, df_ihsg_raw, config)
                if feats_today:
                    feats_today.update({
                        "ticker": t,
                        "date": df.index[idx_today].date(),
                        "close_today": df["Close"].iloc[idx_today],
                        "regime_today": d["regime"],
                        "near_avwap_pct": d.get("distAvwapPct", np.nan),
                        "rvol_15": d.get("rvol", np.nan),
                        "rallyflag10": rally_flag,
                        "ADV20_IDR": adv20,
                        "ADV60_IDR": adv60,
                        "MedValue20_IDR": medv20,
                        "%DaysValue≥Min_20D": pct_liq,
                        "Turnover20_%": turnover20,
                        "Amihud20_(×1e-9)": amihud20,
                    })
                    rows_features_today.append(feats_today)

        except Exception as e:
            logger.error(f"Error processing {t}: {e}", exc_info=True)
            rows_scan.append({"Ticker": t, "Regime": "ERROR", "Eksekusi": str(e)[:100]})

    # Sort scan results
    if progress_callback:
        progress_callback(95, "Saving results...")
    
    scan_df = pd.DataFrame(rows_scan)
    if not scan_df.empty and "Regime" in scan_df.columns:
        reg_cat = pd.Categorical(
            scan_df["Regime"],
            categories=["Momentum (Combo)", "Swing Re-accumulation", "Quiet Accumulation", "—", "ERROR"],
            ordered=True
        )
        scan_df["_order"] = reg_cat
        scan_df = scan_df.sort_values(["_order", "Ticker"]).drop(columns=["_order"])

    # Save outputs
    out_csv = month_dir / f"moonstock_scan_{day:%Y%m%d}.csv"
    out_xlsx = month_dir / f"moonstock_scan_{day:%Y%m%d}.xlsx"
    out_feat = month_dir / f"moonstock_features_today_{day:%Y%m%d}.csv"

    scan_df.to_csv(out_csv, index=False, encoding="utf-8")
    logger.info(f"CSV saved: {out_csv}")

    # Excel workbook
    events_df = pd.DataFrame(events_rows)
    features_df = pd.DataFrame(rows_features_today)
    panel_raw = to_long_panel(data_raw, tickers)
    panel_adj = to_long_panel(data_adj, tickers)

    try:
        with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
            scan_df.to_excel(writer, index=False, sheet_name="Scan")
            ws = writer.sheets["Scan"]
            ws.freeze_panes(1, 1)
            if len(scan_df) > 0:
                ws.autofilter(0, 0, len(scan_df), len(scan_df.columns) - 1)
            
            if not panel_raw.empty:
                panel_raw.to_excel(writer, index=False, sheet_name="ohlc_raw")
            if not panel_adj.empty:
                panel_adj.to_excel(writer, index=False, sheet_name="ohlc_adj")
            if not events_df.empty:
                events_df.to_excel(writer, index=False, sheet_name="events_training")
        
        logger.info(f"Excel saved: {out_xlsx}")
    except Exception as e:
        logger.error(f"Excel save failed: {e}")

    # Features today
    features_df.to_csv(out_feat, index=False, encoding="utf-8")
    logger.info(f"Features saved: {out_feat}")

    # Label spec
    label_spec = {"th_target": config.LABEL_TARGET_PCT, "H_max": config.LABEL_H_MAX}
    (out_xlsx.parent / "label_spec.json").write_text(json.dumps(label_spec, indent=2))

    if progress_callback:
        progress_callback(100, "Complete!")
    
    logger.info(f"=== Scan complete: {len(rows_scan)} tickers processed ===")
    return str(out_csv), str(out_xlsx), str(out_feat)

# ====== GUI with Progress Bar ======
class ProgressDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.title("Scanning...")
        self.top.geometry("400x120")
        self.top.transient(parent)
        self.top.grab_set()
        
        ttk.Label(self.top, text="Processing tickers...").pack(pady=10)
        
        self.progress = ttk.Progressbar(self.top, length=350, mode='determinate')
        self.progress.pack(pady=10)
        
        self.status_label = ttk.Label(self.top, text="Initializing...")
        self.status_label.pack(pady=5)
        
        self.cancelled = False
        
    def update(self, percent: int, message: str):
        self.progress['value'] = percent
        self.status_label.config(text=message)
        self.top.update()
    
    def close(self):
        self.top.destroy()

class CreateToolTip:
    """Fixed tooltip that properly hides on mouse leave"""
    def __init__(self, widget, text=''):
        self.widget = widget
        self.text = text
        self.tw = None
        self.id_after = None
        self.widget.bind("<Enter>", self.on_enter)
        self.widget.bind("<Leave>", self.on_leave)
        self.widget.bind("<ButtonPress>", self.on_leave)

    def on_enter(self, event=None):
        self.schedule()

    def on_leave(self, event=None):
        self.unschedule()
        self.hide()

    def schedule(self):
        self.unschedule()
        self.id_after = self.widget.after(500, self.show)

    def unschedule(self):
        if self.id_after:
            self.widget.after_cancel(self.id_after)
            self.id_after = None

    def show(self):
        if self.tw:
            return
        x = self.widget.winfo_rootx() + 25
        y = self.widget.winfo_rooty() + 20
        self.tw = tk.Toplevel(self.widget)
        self.tw.wm_overrideredirect(True)
        self.tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(self.tw, text=self.text, justify='left',
                        background="#ffffe0", relief='solid', borderwidth=1,
                        wraplength=350, font=("Arial", 8))
        label.pack(ipadx=2, ipady=2)
    
    def hide(self):
        if self.tw:
            self.tw.destroy()
            self.tw = None

TOOLTIP_TEXTS = {
    "RVOL_LEN": "RVOL Lookback Period (Default: 15)\n\nJumlah hari untuk menghitung rata-rata volume.\nRVOL = Volume Hari Ini ÷ Rata-rata Volume 15 Hari\n\nContoh: Jika volume hari ini 1.6x lipat dari rata-rata 15 hari terakhir, RVOL = 1.6",
    
    "RVOL_THRESH": "RVOL Threshold (Default: 1.6)\n\nBatas minimal RVOL untuk menandai volume 'tinggi'.\nHari dengan RVOL ≥ 1.6 dianggap memiliki volume signifikan.\n\nDigunakan untuk:\n• Memulai deteksi base\n• Menentukan regime Quiet Accumulation",
    
    "CALM_MIN_PCT": "Calm Rise Minimum % (Default: 0.5%)\n\nKenaikan harga MINIMAL dalam BASE_WINDOW hari agar dianggap 'calm'.\n\nContoh: Jika BASE_WINDOW=4, harga hari ini harus naik ≥0.5% dari 4 hari lalu.\n\nMencegah: Harga yang flat/stagnan",
    
    "CALM_MAX_PCT": "Calm Rise Maximum % (Default: 3.5%)\n\nKenaikan harga MAKSIMAL dalam BASE_WINDOW hari agar dianggap 'calm'.\n\nContoh: Jika BASE_WINDOW=4, harga hari ini tidak boleh naik >3.5% dari 4 hari lalu.\n\nMencegah: Harga yang sudah rally/melompat",
    
    "BASE_WINDOW": "Base Window Period (Default: 4)\n\nPanjang periode (hari) untuk mengukur kenaikan 'calm'.\n\nFormula:\nCalm Rise % = (Close Hari Ini - Close 4 Hari Lalu) ÷ Close 4 Hari Lalu × 100%\n\nHarus dalam range: CALM_MIN_PCT sampai CALM_MAX_PCT",
    
    "MIN_CONSEC": "Minimum Consecutive Days (Default: 2)\n\nJumlah hari BERTURUT-TURUT yang harus punya RVOL ≥ RVOL_THRESH untuk memulai base.\n\nContoh: Jika setting 2, butuh minimal 2 hari berturut dengan volume tinggi.",
    
    "QA_NEAR_MIN": "Quiet Accumulation Near Min % (Default: 0.5%)\n\nJarak MINIMAL harga dari AVWAP untuk masuk kategori 'Quiet Accumulation'.\n\nFormula:\nDistance = |Close - AVWAP| ÷ AVWAP × 100%\n\nHarus: 0.5% ≤ Distance ≤ 3.0%",
    
    "QA_NEAR_MAX": "Quiet Accumulation Near Max % (Default: 3.0%)\n\nJarak MAKSIMAL harga dari AVWAP untuk masuk kategori 'Quiet Accumulation'.\n\nJika jarak >3%, bukan Quiet Accumulation lagi (bisa jadi Swing Re-accumulation atau Momentum).",
    
    "SR_LOOK": "Swing Re-accumulation Lookback (Default: 5)\n\nPeriode untuk mencari 'minor-high' (high tertinggi dalam N hari terakhir).\n\nSwing Re-acc terjadi saat:\n• High hari ini ≥ Minor-High 5 hari terakhir\n• RVOL ≥ SR_MIN_RVOL\n• Harga di atas AVWAP",
    
    "RALLY_THR_PCT": "Rally Threshold % (Default: 10%)\n\nBatas RunUp % yang dianggap sudah 'rally'.\n\nFormula:\nRunUp % = (High Sejak Base Tutup - Close Base) ÷ Close Base × 100%\n\nJika RunUp ≥ 10%, flag 'rallyflag10' = True (sinyal basi/late)",
    
    "LIQ_MIN_VALUE_IDR": "Minimum Liquidity Value (Default: 200 juta)\n\nNilai transaksi harian minimum (Rupiah) yang dianggap 'likuid'.\n\nValue = Close × Volume\n\nDigunakan untuk hitung:\n%DaysValue≥Min_20D = Berapa % hari (dari 20 hari) yang value-nya ≥ 200 juta",
    
    "LABEL_TARGET_PCT": "Label Target % (Default: 10%)\n\nTarget kenaikan (profit target) untuk pelabelan training data.\n\nLabel = 1 (Success) jika:\nHigh dalam 20 hari ke depan ≥ Close hari ini × (1 + 0.10)\n\nLabel = 0 (Fail) jika tidak tercapai dalam 20 hari",
    
    "LABEL_H_MAX": "Label Horizon Maximum Days (Default: 20)\n\nBatas waktu (hari) untuk menunggu target tercapai.\n\nMelihat 20 hari ke depan setelah sinyal:\n• Jika target 10% tercapai dalam 20 hari → Label = 1\n• Jika tidak tercapai dalam 20 hari → Label = 0",
}

def launch_gui():
    if not HAS_TK:
        print("Tkinter not available. Use --cli mode.")
        return
    
    root = tk.Tk()
    root.title("Moonstock Scanner — Production Ready")
    
    frm = ttk.Frame(root, padding=15)
    frm.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # I/O Section
    io_frame = ttk.LabelFrame(frm, text="Input / Output", padding=10)
    io_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    io_frame.columnconfigure(1, weight=1)

    ttk.Label(io_frame, text="Symbols CSV:").grid(row=0, column=0, sticky="e", padx=5, pady=5)
    symbols_var = tk.StringVar(value=str(DEFAULT_SYMBOLS))
    ttk.Entry(io_frame, textvariable=symbols_var, width=50).grid(row=0, column=1, sticky="ew", padx=5)
    ttk.Button(io_frame, text="Browse", width=10,
               command=lambda: symbols_var.set(p) if (p := filedialog.askopenfilename(
                   filetypes=[("CSV", "*.csv")])) else None).grid(row=0, column=2, padx=5)

    ttk.Label(io_frame, text="Output Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
    out_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
    ttk.Entry(io_frame, textvariable=out_var, width=50).grid(row=1, column=1, sticky="ew", padx=5)
    ttk.Button(io_frame, text="Browse", width=10,
               command=lambda: out_var.set(p) if (p := filedialog.askdirectory()) else None).grid(row=1, column=2, padx=5)

    # Parameters Section
    param_frame = ttk.LabelFrame(frm, text="Parameters", padding=10)
    param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
    for i in range(4):
        param_frame.columnconfigure(i, weight=1)

    param_vars = {}
    config = Config()
    params_list = [k for k in config.__dataclass_fields__.keys()]
    
    for idx, key in enumerate(params_list):
        row = idx // 2
        col = (idx % 2) * 2
        
        lbl = ttk.Label(param_frame, text=f"{key}:")
        lbl.grid(row=row, column=col, sticky="e", padx=5, pady=3)
        
        var = tk.StringVar(value=str(getattr(config, key)))
        ent = ttk.Entry(param_frame, textvariable=var, width=12)
        ent.grid(row=row, column=col+1, sticky="w", padx=5, pady=3)
        
        param_vars[key] = var
        
        if key in TOOLTIP_TEXTS:
            CreateToolTip(lbl, TOOLTIP_TEXTS[key])
            CreateToolTip(ent, TOOLTIP_TEXTS[key])

    # Status
    status_var = tk.StringVar(value="Ready")
    ttk.Label(frm, textvariable=status_var, foreground="blue").grid(row=2, column=0, sticky="w", pady=5)

    # Buttons
    btn_frame = ttk.Frame(frm)
    btn_frame.grid(row=3, column=0, pady=10)

    scan_queue = queue.Queue()
    progress_dialog = None

    def progress_callback(percent, message):
        if progress_dialog:
            progress_dialog.update(percent, message)

    def thread_scan(params_dict, sym_path, out_path):
        try:
            csv_p, xlsx_p, feat_p = run_scan(sym_path, out_path, None, params_dict, progress_callback)
            scan_queue.put({"status": "ok", "xlsx": xlsx_p, "feat": feat_p})
        except ValueError as e:
            scan_queue.put({"status": "error_input", "data": str(e)})
        except Exception as e:
            scan_queue.put({"status": "error", "data": str(e)})

    def check_queue():
        nonlocal progress_dialog
        try:
            result = scan_queue.get_nowait()
            
            if progress_dialog:
                progress_dialog.close()
                progress_dialog = None
            
            btn_run.config(state="normal")
            
            if result["status"] == "ok":
                status_var.set("✓ Scan complete!")
                messagebox.showinfo("Success", f"Scan complete!\n\nExcel: {result['xlsx']}\nFeatures: {result['feat']}")
            elif result["status"] == "error_input":
                status_var.set("✗ Input error")
                messagebox.showerror("Input Error", result["data"])
            else:
                status_var.set("✗ Error occurred")
                messagebox.showerror("Error", result["data"])
        except queue.Empty:
            root.after(100, check_queue)

    def start_scan():
        nonlocal progress_dialog
        
        # Validate inputs
        try:
            params_dict = {}
            for key, var in param_vars.items():
                val_str = var.get()
                default_val = getattr(Config(), key)
                if isinstance(default_val, int):
                    params_dict[key] = int(val_str)
                elif isinstance(default_val, float):
                    params_dict[key] = float(val_str)
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter value: {e}")
            return
        
        btn_run.config(state="disabled")
        status_var.set("Starting scan...")
        
        progress_dialog = ProgressDialog(root)
        
        threading.Thread(
            target=thread_scan,
            args=(params_dict, symbols_var.get(), out_var.get()),
            daemon=True
        ).start()
        
        root.after(100, check_queue)

    btn_run = ttk.Button(btn_frame, text="▶ Run Scan", command=start_scan, width=15)
    btn_run.pack(side="left", padx=5)

    def open_folder():
        path = out_var.get()
        try:
            if sys.platform == 'win32':
                os.startfile(path)
            elif sys.platform == 'darwin':
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open folder: {e}")

    ttk.Button(btn_frame, text="📁 Open Output", command=open_folder, width=15).pack(side="left", padx=5)

    root.mainloop()

# ====== CLI Entry ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moonstock Scanner — Production Ready")
    parser.add_argument("--gui", action="store_true", help="Launch GUI (default)")
    parser.add_argument("--cli", action="store_true", help="Force CLI mode")
    parser.add_argument("--symbols", type=str, help="Path to symbols CSV")
    parser.add_argument("--out", type=str, help="Output directory")
    args = parser.parse_args()

    try:
        is_frozen = getattr(sys, "frozen", False)
        no_args = len(sys.argv) == 1
        
        if args.cli:
            print("Running in CLI mode...")
            csv_p, xlsx_p, feat_p = run_scan(
                args.symbols or str(DEFAULT_SYMBOLS),
                args.out or str(DEFAULT_OUTPUT_DIR),
                None, None
            )
            print(f"\n✓ Complete!")
            print(f"  CSV:      {csv_p}")
            print(f"  Excel:    {xlsx_p}")
            print(f"  Features: {feat_p}")
        elif args.gui or is_frozen or no_args:
            if not HAS_TK:
                print("Tkinter unavailable. Falling back to CLI...")
                csv_p, xlsx_p, feat_p = run_scan(
                    args.symbols or str(DEFAULT_SYMBOLS),
                    args.out or str(DEFAULT_OUTPUT_DIR),
                    None, None
                )
                print(f"\n✓ Complete!\n  CSV: {csv_p}\n  Excel: {xlsx_p}\n  Features: {feat_p}")
            else:
                launch_gui()
        else:
            csv_p, xlsx_p, feat_p = run_scan(
                args.symbols or str(DEFAULT_SYMBOLS),
                args.out or str(DEFAULT_OUTPUT_DIR),
                None, None
            )
            print(f"\n✓ Complete!\n  CSV: {csv_p}\n  Excel: {xlsx_p}\n  Features: {feat_p}")
            
    except KeyboardInterrupt:
        print("\n\n✗ Scan cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        if HAS_TK:
            try:
                messagebox.showerror("Fatal Error", str(e))
            except:
                pass
        sys.exit(1)