import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import oandapyV20
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
import ta

# --- Configuration Streamlit ---
st.set_page_config(page_title="Forex Multi-Timeframe Scanner", layout="wide")

st.title("📊 Forex Multi-Timeframe Signal Scanner (HMA20 + RSI7)")
st.write("Ce tableau classe les paires selon le **signal le plus récent** sur H1, H4 et D1 à partir de l’API OANDA.")

# --- OANDA API ---
account_id = st.secrets["OANDA_ACCOUNT_ID"]
access_token = st.secrets["OANDA_ACCESS_TOKEN"]
api = API(access_token=access_token)

# --- Paramètres ---
PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF", "AUD_USD", "NZD_USD",
    "USD_CAD", "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "CAD_JPY",
    "NZD_JPY", "EUR_AUD", "EUR_CAD", "EUR_NZD", "GBP_AUD", "GBP_CAD",
    "GBP_NZD", "AUD_CAD", "AUD_NZD", "CAD_CHF", "CHF_JPY", "AUD_CHF",
    "NZD_CHF", "EUR_CHF", "GBP_CHF", "USD_SEK"
]
TIMEFRAMES = {"H1": "60", "H4": "240", "D1": "1440"}

# --- Fonctions indicateurs ---
def hma(series, length=20):
    half_len = int(length / 2)
    sqrt_len = int(np.sqrt(length))
    wma_half = ta.trend.wma_indicator(series, window=half_len)
    wma_full = ta.trend.wma_indicator(series, window=length)
    diff = 2 * wma_half - wma_full
    return ta.trend.wma_indicator(diff, window=sqrt_len)

def get_signal(data):
    data["hma20"] = hma(data["close"], 20)
    data["rsi7"] = ta.momentum.RSIIndicator(data["close"], 7).rsi()
    data["signal"] = np.where(
        (data["close"] > data["hma20"]) & (data["rsi7"] > 50), "BUY",
        np.where((data["close"] < data["hma20"]) & (data["rsi7"] < 50), "SELL", "")
    )
    return data

def fetch_candles(pair, granularity="H1", count=100):
    params = {"granularity": granularity, "count": count, "price": "M"}
    r = InstrumentsCandles(instrument=pair, params=params)
    api.request(r)
    data = r.response["candles"]
    df = pd.DataFrame([{
        "time": c["time"],
        "close": float(c["mid"]["c"])
    } for c in data if c["complete"]])
    df["time"] = pd.to_datetime(df["time"])
    return df

# --- Scanner ---
def scan():
    all_results = []
    for pair in PAIRS:
        try:
            for tf_name, tf_val in TIMEFRAMES.items():
                df = fetch_candles(pair, tf_val)
                df = get_signal(df)
                last = df.iloc[-1]
                if last["signal"]:
                    all_results.append({
                        "pair": pair,
                        "timeframe": tf_name,
                        "signal": last["signal"],
                        "price": round(last["close"], 5),
                        "time": last["time"]
                    })
        except Exception as e:
            st.warning(f"Erreur {pair}: {e}")
    return pd.DataFrame(all_results)

if st.button("🔄 Lancer le scan"):
    with st.spinner("Analyse en cours..."):
        results = scan()
        if not results.empty:
            results = results.sort_values(by="time", ascending=False)
            st.dataframe(results, use_container_width=True)
        else:
            st.info("Aucun signal détecté pour l’instant.")
