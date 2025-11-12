import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Tuple, List, Dict, Optional
from oandapyV20 import API
from oandapyV20.endpoints.instruments import InstrumentsCandles
from concurrent.futures import ThreadPoolExecutor, as_completed
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Forex Multi-Timeframe Scanner", layout="wide")

# ----------------------
# Configuration
# ----------------------
PAIRS_DEFAULT = [
    "EUR_USD","GBP_USD","USD_JPY","USD_CHF","AUD_USD","NZD_USD","USD_CAD","EUR_GBP",
    "EUR_JPY","GBP_JPY","AUD_JPY","CAD_JPY","NZD_JPY","EUR_AUD","EUR_CAD","EUR_NZD",
    "GBP_AUD","GBP_CAD","GBP_NZD","AUD_CAD","AUD_NZD","CAD_CHF","CHF_JPY","AUD_CHF",
    "NZD_CHF","EUR_CHF","GBP_CHF","USD_SEK"
]

GRANULARITY_MAP = {"H1": "H1", "H4": "H4", "D1": "D", "W": "W"}

# ----------------------
# Interface
# ----------------------
st.title("📊 Forex Multi-Timeframe Signal Scanner Pro")
st.write("Scanner optimisé avec analyse parallèle et visualisations avancées")

# ----------------------
# OANDA API
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
# Fonctions utilitaires
# ----------------------
@st.cache_data(ttl=60)  # Cache réduit à 60s pour données plus fraîches
def get_candles(pair: str, tf: str, count: int = 200) -> pd.DataFrame:
    """Télécharge les bougies OANDA avec gestion d'erreur robuste."""
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
            if not c.get("complete", True):  # Ignorer les bougies incomplètes
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
            except (KeyError, ValueError):
                continue
        
        if not records:
            return pd.DataFrame()
        
        df = pd.DataFrame(records)
        df["time"] = pd.to_datetime(df["time"])
        return df
    except Exception as e:
        st.warning(f"Erreur récupération {pair} {tf}: {str(e)[:50]}")
        return pd.DataFrame()

# --- Calculs indicateurs (optimisés)
def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average optimisé."""
    weights = np.arange(1, length + 1)
    def weighted_mean(x):
        return np.dot(x, weights) / weights.sum() if len(x) == length else np.nan
    return series.rolling(length).apply(weighted_mean, raw=True)

def hma(series: pd.Series, length: int = 20) -> pd.Series:
    """Hull Moving Average."""
    half = max(1, int(length / 2))
    sqrt_l = max(1, int(np.sqrt(length)))
    return wma(2 * wma(series, half) - wma(series, length), sqrt_l)

def rsi(series: pd.Series, length: int = 7) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range pour volatilité."""
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
    """Analyse tendance multi-timeframe avec force du signal."""
    map_higher = {"H1": "H4", "H4": "D1", "D1": "W"}
    higher = map_higher.get(tf, "H4")
    
    df = get_candles(pair, higher, count=100)
    if df.empty or len(df) < 50:
        return {"trend": "neutral", "strength": 0}
    
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    
    # Calcul de la force de la tendance
    distance_pct = abs((ema20 - ema50) / ema50) * 100
    
    if ema20 > ema50 and price > ema20:
        strength = min(distance_pct * 10, 100)  # Normaliser à 0-100
        return {"trend": "bullish", "strength": round(strength, 1)}
    elif ema20 < ema50 and price < ema20:
        strength = min(distance_pct * 10, 100)
        return {"trend": "bearish", "strength": round(strength, 1)}
    
    return {"trend": "neutral", "strength": 0}

# ----------------------
# Analyse d'une paire (optimisée)
# ----------------------
def analyze_pair(pair: str, tf: str, candles_count: int) -> Optional[Dict]:
    """Analyse complète avec gestion d'erreur robuste."""
    df = get_candles(pair, tf, count=candles_count)
    if df.empty or len(df) < 30:
        return None
    
    df = df.sort_values("time").reset_index(drop=True)
    
    # Calcul indicateurs
    df["hma20"] = hma(df["close"], 20)
    df["rsi7"] = rsi(df["close"], 7)
    df["atr14"] = atr(df, 14)
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()
    
    # Signaux HMA
    df["hma_up"] = df["hma20"] > df["hma20"].shift(1)
    df["hma_bullish_change"] = df["hma_up"] & (~df["hma_up"].shift(1).fillna(False))
    df["hma_bearish_change"] = (~df["hma_up"]) & (df["hma_up"].shift(1).fillna(False))
    
    # Signaux RSI
    df["rsi_cross_up"] = (df["rsi7"] > 50) & (df["rsi7"].shift(1) <= 50)
    df["rsi_cross_down"] = (df["rsi7"] < 50) & (df["rsi7"].shift(1) >= 50)
    
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else last
    
    # Détection signaux bruts
    raw_buy = bool(last.get("hma_bullish_change", False)) and bool(last.get("rsi_cross_up", False))
    raw_sell = bool(last.get("hma_bearish_change", False)) and bool(last.get("rsi_cross_down", False))
    
    # Validation MTF
    mtf_info = check_mtf_trend(pair, tf)
    mtf_trend = mtf_info["trend"]
    mtf_strength = mtf_info["strength"]
    
    # Signaux validés
    buy = raw_buy and mtf_trend == "bullish"
    sell = raw_sell and mtf_trend == "bearish"
    
    signal = None
    confidence = 0
    
    if buy:
        signal = "🟢 Achat"
        # Calcul confiance basé sur RSI et force MTF
        rsi_strength = (last["rsi7"] - 50) / 50 * 100  # 0-100%
        confidence = (rsi_strength * 0.4 + mtf_strength * 0.6)
    elif sell:
        signal = "🔴 Vente"
        rsi_strength = (50 - last["rsi7"]) / 50 * 100
        confidence = (rsi_strength * 0.4 + mtf_strength * 0.6)
    
    if signal is None:
        return None
    
    # Calcul niveaux SL/TP basés sur ATR
    atr_value = last["atr14"]
    price = last["close"]
    
    if buy:
        sl = price - (2 * atr_value)
        tp = price + (3 * atr_value)
    else:
        sl = price + (2 * atr_value)
        tp = price - (3 * atr_value)
    
    return {
        "Instrument": pair,
        "TF": tf,
        "Signal": signal,
        "Confiance": f"{round(confidence, 1)}%",
        "Prix": round(price, 5),
        "SL": round(sl, 5),
        "TP": round(tp, 5),
        "R:R": "1:1.5",
        "RSI": round(last["rsi7"], 1),
        "MTF": f"{mtf_trend} ({mtf_strength}%)",
        "Heure": last["time"],
        "_confidence_val": confidence,  # Pour tri
        "_df": df.tail(50)  # Pour graphique
    }

# ----------------------
# Scan parallélisé
# ----------------------
def scan_parallel(pairs: List[str], tfs: List[str], candles_count: int, max_workers: int = 5) -> List[Dict]:
    """Scan parallélisé pour performances optimales."""
    results = []
    tasks = [(pair, tf) for pair in pairs for tf in tfs]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(analyze_pair, pair, tf, candles_count): (pair, tf)
            for pair, tf in tasks
        }
        
        progress_bar = st.progress(0)
        completed = 0
        total = len(tasks)
        
        for future in as_completed(future_to_task):
            completed += 1
            progress_bar.progress(completed / total)
            
            try:
                result = future.result(timeout=10)
                if result and result["Signal"]:
                    results.append(result)
            except Exception as e:
                pair, tf = future_to_task[future]
                st.warning(f"Erreur {pair} {tf}: {str(e)[:50]}")
        
        progress_bar.empty()
    
    return results

# ----------------------
# Visualisation graphique
# ----------------------
def plot_signal_chart(data: Dict):
    """Graphique interactif du signal."""
    df = data["_df"]
    
    fig = go.Figure()
    
    # Chandeliers
    fig.add_trace(go.Candlestick(
        x=df["time"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Prix"
    ))
    
    # HMA20
    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["hma20"],
        name="HMA20",
        line=dict(color="blue", width=2)
    ))
    
    # Niveaux SL/TP
    last_time = df["time"].iloc[-1]
    fig.add_hline(y=data["Prix"], line_dash="dash", line_color="white", annotation_text="Entry")
    fig.add_hline(y=data["SL"], line_dash="dot", line_color="red", annotation_text="SL")
    fig.add_hline(y=data["TP"], line_dash="dot", line_color="green", annotation_text="TP")
    
    fig.update_layout(
        title=f"{data['Instrument']} - {data['TF']} - {data['Signal']}",
        xaxis_title="Date",
        yaxis_title="Prix",
        height=400,
        template="plotly_dark"
    )
    
    return fig

# ----------------------
# Interface utilisateur
# ----------------------
col1, col2 = st.columns([1, 3])

with col1:
    st.sidebar.header("⚙️ Configuration")
    selected_pairs = st.sidebar.multiselect(
        "Paires à scanner :",
        PAIRS_DEFAULT,
        default=PAIRS_DEFAULT[:10]  # Par défaut 10 paires
    )
    
    max_pairs = st.sidebar.number_input(
        "Max paires :",
        min_value=1,
        max_value=len(selected_pairs),
        value=min(10, len(selected_pairs))
    )
    
    selected_tfs = st.sidebar.multiselect(
        "Timeframes :",
        ["H1", "H4", "D1"],
        default=["H4", "D1"]
    )
    
    candles_count = st.sidebar.selectbox(
        "Bougies par TF :",
        [100, 150, 200],
        index=1
    )
    
    max_workers = st.sidebar.slider(
        "Workers parallèles :",
        min_value=3,
        max_value=10,
        value=5,
        help="Plus = rapide mais charge API"
    )
    
    min_confidence = st.sidebar.slider(
        "Confiance min (%) :",
        min_value=0,
        max_value=100,
        value=50
    )
    
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5min)")
    
    scan_button = st.sidebar.button("🔄 LANCER LE SCAN", type="primary", use_container_width=True)

# ----------------------
# Scan principal
# ----------------------
if scan_button or auto_refresh:
    if auto_refresh:
        st.sidebar.info("Prochain scan dans 5 min...")
        time.sleep(300)  # 5 minutes
        st.rerun()
    
    with st.spinner("🔍 Scan en cours..."):
        start_time = time.time()
        pairs_to_scan = selected_pairs[:max_pairs]
        
        results = scan_parallel(pairs_to_scan, selected_tfs, candles_count, max_workers)
        
        # Filtrer par confiance
        results = [r for r in results if float(r["Confiance"].strip("%")) >= min_confidence]
        
        elapsed = time.time() - start_time
        st.success(f"✅ Scan terminé en {elapsed:.1f}s - {len(results)} signaux trouvés")
    
    if results:
        # Tri par confiance
        results.sort(key=lambda x: x["_confidence_val"], reverse=True)
        
        # Affichage tableau
        df_display = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400,
            column_config={
                "Signal": st.column_config.TextColumn("Signal", width="small"),
                "Confiance": st.column_config.ProgressColumn("Confiance", format="%s", min_value=0, max_value=100),
            }
        )
        
        # Graphiques des meilleurs signaux
        st.subheader("📈 Top 3 Signaux")
        cols = st.columns(3)
        for idx, result in enumerate(results[:3]):
            with cols[idx]:
                st.plotly_chart(plot_signal_chart(result), use_container_width=True)
        
        # Export CSV
        csv = df_display.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Télécharger les signaux (CSV)",
            csv,
            f"signals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    else:
        st.info("Aucun signal détecté avec les critères actuels. Réduisez la confiance minimale ou élargissez la sélection.")
else:
    st.info("👈 Configurez le scanner et cliquez sur **LANCER LE SCAN**")
    
    # Statistiques
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Paires disponibles", len(PAIRS_DEFAULT))
    col2.metric("Timeframes", "3 (H1, H4, D1)")
    col3.metric("Indicateurs", "HMA20 + RSI7 + ATR")
    col4.metric("Validation", "Multi-TF")
   
