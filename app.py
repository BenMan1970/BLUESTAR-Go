# app.py
import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Tuple
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles

st.set_page_config(page_title="Forex Multi-Timeframe Scanner", layout="wide")

# ----------------------
# Config interface
# ----------------------
st.title("📊 Forex Multi-Timeframe Signal Scanner (HMA20 + RSI7)")
st.write("Scan H1 / H4 / D1 — Signaux HMA20 + RSI7 validés par timeframe supérieur")

# --- Paires par défaut (exemple ~28)
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
    "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"
]

# Timeframes map pour OANDA
GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D"}

# Scanning options
selected_pairs = st.multiselect("Paires à scanner", PAIRS_DEFAULT, default=PAIRS_DEFAULT[:12])
scan_button = st.button("🔄 Lancer le scan")
max_pairs_to_scan = st.number_input("Max paires (pour limiter appels API)", min_value=1, max_value=len(selected_pairs), value=min(12, len(selected_pairs)))
scan_count_per_tf = st.selectbox("Nombre de bougies (par TF)", [100, 200, 300], index=0)

# ----------------------
# OANDA client (secure)
# ----------------------
@st.cache_resource
def get_oanda_client() -> Tuple[API, str]:
    account_id = st.secrets.get("OANDA_ACCOUNT_ID")
    token = st.secrets.get("OANDA_ACCESS_TOKEN")
    if not account_id or not token:
        raise RuntimeError("OANDA secrets manquants. Ajoutez OANDA_ACCOUNT_ID et OANDA_ACCESS_TOKEN dans Streamlit Secrets.")
    client = API(access_token=token)
    return client, account_id

try:
    client, ACCOUNT_ID = get_oanda_client()
except Exception as e:
    st.error(f"Erreur configuration OANDA: {e}")
    st.stop()

# ----------------------
# Helpers: fetch & indicators
# ----------------------
@st.cache_data(ttl=120)  # cache 2 minutes
def get_candles(pair: str, tf: str, count: int = 200) -> pd.DataFrame:
    """Récupère les bougies (utilise mapping GRANULARITY_MAP)."""
    gran = GRANULARITY_MAP.get(tf)
    if gran is None:
        return pd.DataFrame()
    params = {"granularity": gran, "count": count, "price": "M"}
    req = InstrumentsCandles(instrument=pair, params=params)
    client.request(req)
    candles = req.response.get("candles", [])
    records = []
    for c in candles:
        # inclure toutes les bougies complètes ou non (on veut clôtures), mais OANDA 'mid' existe
        try:
            records.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"])
            })
        except Exception:
            continue
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["time"] = pd.to_datetime(df["time"])
    return df

def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    def calc(x):
        if len(x) < length:
            return np.nan
        return np.dot(x, weights) / weights.sum()
    return series.rolling(length).apply(calc, raw=True)

def hma(series: pd.Series, length: int = 20) -> pd.Series:
    half = max(1, int(length / 2))
    sqrt_l = max(1, int(np.sqrt(length)))
    wma_half = wma(series, half)
    wma_full = wma(series, length)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_l)

def rsi(series: pd.Series, length: int = 7) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(length).mean()
    avg_loss = loss.rolling(length).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def check_mtf_trend(pair: str, tf: str) -> str:
    """Retourne 'bullish'/'bearish'/'neutral' en regardant le timeframe supérieur."""
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}  # W sera mappé en "D" pour OANDA si besoin
    higher = map_higher.get(tf, "H4")
    if higher == "W":
        gran = "W"
    else:
        gran = GRANULARITY_MAP.get(higher, "H4")
    # utiliser get_candles (cache) - demander moins de bougies
    df = get_candles(pair, higher if higher in GRANULARITY_MAP else "H4", count=100)
    if df.empty or len(df) < 20:
        return "neutral"
    close = df["close"]
    ema_fast = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_slow = close.ewm(span=50, adjust=False).mean().iloc[-1]
    if ema_fast > ema_slow:
        return "bullish"
    elif ema_fast < ema_slow:
        return "bearish"
    return "neutral"

# ----------------------
# Scan logic
# ----------------------
def analyze_pair(pair: str, tf: str, candles_count: int):
    df = get_candles(pair, tf, count=candles_count)
    if df.empty or len(df) < 20:
        return None
    df = df.sort_values("time").reset_index(drop=True)
    df["hma20"] = hma(df["close"], 20)
    df["rsi7"] = rsi(df["close"], 7)
    # conditions : changement pente HMA (bullish_change / bearish_change) + rsi crossing 50
    df["hma_up"] = df["hma20"] > df["hma20"].shift(1)
    df["hma_bullish_change"] = df["hma_up"] & (~df["hma_up"].shift(1).fillna(False))
    df["hma_bearish_change"] = (~df["hma_up"]) & (df["hma_up"].shift(1).fillna(False))
    df["rsi_cross_up"] = (df["rsi7"] > 50) & (df["rsi7"].shift(1) <= 50)
    df["rsi_cross_down"] = (df["rsi7"] < 50) & (df["rsi7"].shift(1) >= 50)

    last = df.iloc[-1]
    raw_buy = bool(last.get("hma_bullish_change", False)) and bool(last.get("rsi_cross_up", False))
    raw_sell = bool(last.get("hma_bearish_change", False)) and bool(last.get("rsi_cross_down", False))
    # validation MTF (higher timeframe)
    mtf_trend = check_mtf_trend(pair, tf)
    buy = raw_buy and mtf_trend == "bullish"
    sell = raw_sell and mtf_trend == "bearish"
    signal = None
    if buy:
        signal = "Achat"
    elif sell:
        signal = "Vente"
    return {
        "Instrument": pair,
        "Timeframe": tf,
        "Signal": signal,
        "MTF Trend": mtf_trend,
        "Price": round(last["close"], 5),
        "Time": last["time"]
    }

# ----------------------
# UI: lancement du scan
# ----------------------
placeholder = st.empty()
if scan_button:
    placeholder.info("Lancement du scan... (limite d'appels: {})".format(max_pairs_to_scan))
    results = []
    pairs_to_scan = selected_pairs[:max_pairs_to_scan]
    tfs = ["H1", "H4", "D1"]
    # itérer paires x timeframes
    for pair in pairs_to_scan:
        for tf in tfs:
            try:
                res = analyze_pair(pair, tf, candles_count=scan_count_per_tf)
                # small delay to avoid bursts (optionnel)
                time.sleep(0.2)
                if res and res["Signal"]:
                    results.append(res)
            except Exception as e:
                st.warning(f"Erreur {pair} {tf}: {e}")
    if results:
        df_res = pd.DataFrame(results).sort_values(by="Time", ascending=False)
        st.dataframe(df_res.reset_index(drop=True), use_container_width=True, height=600)
    else:
        st.info("Aucun signal détecté pour l'instant.")
else:
    st.write("Appuyez sur **Lancer le scan** pour démarrer.")

