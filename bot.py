#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAUUSD Quant Bot - Zero-API-Key Streamlined Edition v31.0
Author: Global Head of Quantitative Risk & Trading Systems
"""

import os
import sys
import json
import logging
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import requests
except ImportError as e:
    print(f"FATAL: missing dependency: {e}", file=sys.stderr)
    sys.exit(1)

# ===========================================================================
# ENVIRONMENT HELPERS
# ===========================================================================
def _env_float(k: str, d: float) -> float:
    v = (os.getenv(k) or "").strip()
    try:
        return float(v) if v else d
    except ValueError:
        return d

def _env_bool(k: str, d: bool = False) -> bool:
    v = (os.getenv(k) or "").strip().lower()
    return v in ("true", "1", "yes", "on") if v else d

def _env_str(k: str, d: str = "") -> str:
    return (os.getenv(k) or "").strip() or d

# ===========================================================================
# CONFIGURATION (No External API Keys Needed!)
# ===========================================================================
@dataclass(frozen=True)
class Config:
    telegram_token: str = _env_str("TELEGRAM_TOKEN")
    telegram_chat_id: str = _env_str("TELEGRAM_CHAT_ID")
    
    ticker: str = "GC=F"  # Yahoo Finance kode untuk Gold Futures
    timeframe_minutes: int = 15
    min_candles: int = 300
    
    account_equity: float = _env_float("ACCOUNT_EQUITY", 1000.0)
    risk_percent: float = _env_float("RISK_PERCENT", 1.0)
    
    audit_log_path: str = str(Path.home() / ".xauusd_audit.jsonl")

CONFIG = Config()

# ===========================================================================
# LOGGING
# ===========================================================================
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        return json.dumps(payload, default=str)

def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("xauusd_v31")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(JsonFormatter())
    logger.addHandler(sh)
    return logger

log = _setup_logging()

# ===========================================================================
# TELEGRAM NOTIFIER (Optional & Graceful)
# ===========================================================================
def send_telegram_alert(message: str) -> None:
    if not CONFIG.telegram_token or not CONFIG.telegram_chat_id:
        log.info("Telegram credentials not set. Skipping notification.")
        return
    try:
        url = f"https://api.telegram.org/bot{CONFIG.telegram_token}/sendMessage"
        payload = {"chat_id": CONFIG.telegram_chat_id, "text": message, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=5)
        if response.status_code != 200:
            log.warning(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        log.warning(f"Telegram alert error: {e}")

# ===========================================================================
# DATA FETCHER (Using Free YFinance)
# ===========================================================================
def fetch_market_data() -> pd.DataFrame:
    log.info(f"Fetching market data for {CONFIG.ticker} via Yahoo Finance...")
    try:
        df = yf.download(CONFIG.ticker, period="60d", interval="15m", progress=False)
        if df.empty:
            raise ValueError("Empty dataframe returned from Yahoo Finance.")
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        
        rename_map = {"date": "datetime", "datetime": "datetime"}
        for k, v in rename_map.items():
            if k in df.columns and "datetime" not in df.columns:
                df = df.rename(columns={k: "datetime"})
                
        if "datetime" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "datetime"})
            
        df = df[["datetime", "open", "high", "low", "close", "volume"]].dropna()
        log.info(f"Successfully loaded {len(df)} candles.")
        return df
    except Exception as e:
        log.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# ===========================================================================
# QUANTITATIVE CORE & INDICATORS
# ===========================================================================
def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    d = df["close"].diff()
    g = d.clip(lower=0).rolling(14).mean()
    l = (-d.clip(upper=0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + g / (l + 1e-12))
    
    prev = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    
    df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()
    
    return df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# ===========================================================================
# MAIN EXECUTION ROUTINE
# ===========================================================================
def main() -> None:
    log.info("XAUUSD Zero-API-Key Bot v31.0 — Execution Started")
    
    df = fetch_market_data()
    if len(df) < CONFIG.min_candles:
        log.warning("Insufficient data candles retrieved.")
        return
        
    df_feat = calculate_features(df)
    latest = df_feat.iloc[-1]
    
    report = (
        f"🤖 *XAUUSD Quant Bot v31.0 (Zero-API)*\n"
        f"Status: `Operational`\n"
        f"📅 Waktu: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC`\n"
        f"💰 Harga Terakhir: `${latest['close']:.2f}`\n"
        f"📊 RSI (14): `{latest['rsi']:.2f}`\n"
        f"📉 ATR: `{latest['atr']:.2f}`\n"
        f"💼 Ekuitas Akun: `${CONFIG.account_equity:,.2f}`"
    )
    
    print("\n=======================================================")
    print(report.replace("*", "").replace("`", ""))
    print("=======================================================\n")
    
    send_telegram_alert(report)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.critical(f"Fatal error: {e}\n{traceback.format_exc()}")
        sys.exit(1)
