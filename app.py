# app.py complet avec modifications
import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import pytz

st.set_page_config(page_title="Forex Multi-Timeframe Scanner", layout="wide")

PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
    "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

st.title("📊 Forex Multi-Timeframe Signal Scanner Pro")
st.write("Scanner optimisé avec analyse parallèle et calculs SL/TP automatiques")

@st.cache_resource
def get_oanda_client() -> Tuple[API, str]:
    account_id = st.secrets.get("OANDA_ACCOUNT_ID")
    token = st.secrets.get("OANDA_ACCESS_TOKEN")
    if not account_id or not token:
        raise RuntimeError("OANDA secrets manquants.")
    client = API(access_token=token)
    return client, account_id

try:
    client, ACCOUNT_ID = get_oanda_client()
except Exception as e:
    st.error(f"Erreur configuration OANDA: {e}")
    st.stop()

@st.cache_data(ttl=30)
def get_candles(pair: str, tf: str, count: int = 200, include_incomplete: bool = False) -> pd.DataFrame:
    gran = GRANULARITY_MAP.get(tf)
    if gran is None:
        return pd.DataFrame()

    try:
        params = {"granularity": gran, "count": count, "price": "M"}
        req = InstrumentsCandles(instrument=pair, params=params)
        client.request(req)
        candles = req.response.get("candles", [])

        records = []
        for c in candles:
            if not include_incomplete and not c.get("complete", True):
                continue
            try:
                records.append({
                    "time": c["time"],
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c.get("volume", 0))
                })
            except:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except:
        return pd.DataFrame()

def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    def weighted_mean(x):
        return np.dot(x, weights) / weights.sum() if len(x) == length else np.nan
    return series.rolling(length).apply(weighted_mean, raw=True)

def hma(series: pd.Series, length: int = 20) -> pd.Series:
    half = max(1, int(length / 2))
    sqrt_l = max(1, int(np.sqrt(length)))
    return wma(2 * wma(series, half) - wma(series, length), sqrt_l)

def rsi(series: pd.Series, length: int = 7) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

@st.cache_data(ttl=120)
def check_mtf_trend(pair: str, tf: str) -> Dict[str, any]:
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher = map_higher.get(tf)
    if not higher:
        return {"trend": "neutral", "strength": 0}

    df = get_candles(pair, higher, count=100)
    if df.empty or len(df) < 50:
        return {"trend": "neutral", "strength": 0}

    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]

    distance_pct = abs((ema20 - ema50) / ema50) * 100

    if ema20 > ema50 and price > ema20:
        strength = min(distance_pct * 10, 100)
        return {"trend": "bullish", "strength": round(strength, 1)}
    elif ema20 < ema50 and price < ema20:
        strength = min(distance_pct * 10, 100)
        return {"trend": "bearish", "strength": round(strength, 1)}

    return {"trend": "neutral", "strength": 0}

def analyze_pair(pair: str, tf: str, candles_count: int, max_candles_back: int = 3) -> Optional[Dict]:
    df = get_candles(pair, tf, count=candles_count, include_incomplete=True)
    if df.empty or len(df) < 30:
        return None

    df = df.sort_values("time").reset_index(drop=True)
    df["hma20"] = hma(df["close"], 20)
    df["rsi7"] = rsi(df["close"], 7)
    df["atr14"] = atr(df, 14)
    df["hma_up"] = df["hma20"] > df["hma20"].shift(1)

    last_n = df.tail(max_candles_back)
    hma_became_bullish = False
    hma_became_bearish = False

    for i in range(len(last_n) - 1):
        curr_up = last_n.iloc[i+1]["hma_up"]
        prev_up = last_n.iloc[i]["hma_up"]
        if curr_up and not prev_up:
            hma_became_bullish = True
        if not curr_up and prev_up:
            hma_became_bearish = True

    last = df.iloc[-1]
    rsi_bullish = last["rsi7"] > 50
    rsi_bearish = last["rsi7"] < 50

    rsi_crossed_up_recently = any(
        (last_n.iloc[i]["rsi7"] > 50) and (last_n.iloc[i-1]["rsi7"] <= 50)
        for i in range(1, len(last_n))
    )
    rsi_crossed_down_recently = any(
        (last_n.iloc[i]["rsi7"] < 50) and (last_n.iloc[i-1]["rsi7"] >= 50)
        for i in range(1, len(last_n))
    )

    mtf_info = check_mtf_trend(pair, tf)
    mtf_trend = mtf_info["trend"]
    mtf_strength = mtf_info["strength"]

    raw_buy = (hma_became_bullish or last["hma_up"]) and rsi_bullish
    raw_sell = (hma_became_bearish or not last["hma_up"]) and rsi_bearish

    has_rsi_confirmation = rsi_crossed_up_recently or rsi_crossed_down_recently

    buy = raw_buy and mtf_trend == "bullish"
    sell = raw_sell and mtf_trend == "bearish"

    signal = None
    confidence = 0

    if buy:
        signal = "🟢 ACHAT"
        rsi_strength = (last["rsi7"] - 50) / 50 * 100
        confidence = (rsi_strength * 0.4 + mtf_strength * 0.6)
                if has_rsi_confirmation:
