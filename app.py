import streamlit as st
import oandapyV20
from oandapyV20.endpoints import instruments
import pandas as pd

# --- Connexion à OANDA ---
OANDA_ACCOUNT_ID = st.secrets["OANDA_ACCOUNT_ID"]
OANDA_ACCESS_TOKEN = st.secrets["OANDA_ACCESS_TOKEN"]
client = oandapyV20.API(access_token=OANDA_ACCESS_TOKEN)

# --- Mapping des timeframes internes vers OANDA ---
GRANULARITY_MAP = {
    "H1": "H1",
    "H4": "H4",
    "D1": "D"   # OANDA utilise "D" et non "D1"
}

# --- Fonction pour récupérer les bougies ---
def get_candles(pair: str, tf: str, count: int = 300):
    """Récupère les bougies d'une paire donnée pour un timeframe (H1, H4, D1)."""
    try:
        if tf not in GRANULARITY_MAP:
            raise ValueError(f"Timeframe invalide: {tf}")

        params = {
            "granularity": GRANULARITY_MAP[tf],
            "count": count,
            "price": "M"
        }
        r = instruments.InstrumentsCandles(instrument=pair, params=params)
        client.request(r)
        data = r.response.get("candles", [])

        if not data:
            return pd.DataFrame()

        records = []
        for c in data:
            records.append({
                "time": c["time"],
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"])
            })
        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        return df

    except Exception as e:
        st.error(f"Erreur {pair}: {e}")
        return pd.DataFrame()

