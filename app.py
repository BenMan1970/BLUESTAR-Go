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
st.write("Scanner optimisé avec analyse parallèle et calculs SL/TP automatiques")

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
@st.cache_data(ttl=60)
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
            if not c.get("complete", True):
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
        strength = min(distance_pct * 10, 100)
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
        signal = "🟢 ACHAT"
        rsi_strength = (last["rsi7"] - 50) / 50 * 100
        confidence = (rsi_strength * 0.4 + mtf_strength * 0.6)
    elif sell:
        signal = "🔴 VENTE"
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
        risk_pips = abs(price - sl)
    else:
        sl = price + (2 * atr_value)
        tp = price - (3 * atr_value)
        risk_pips = abs(price - sl)
    
    reward_pips = abs(tp - price)
    rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
    
    return {
        "Instrument": pair,
        "TF": tf,
        "Signal": signal,
        "Confiance": round(confidence, 1),
        "Prix": round(price, 5),
        "SL": round(sl, 5),
        "TP": round(tp, 5),
        "R:R": f"1:{round(rr_ratio, 1)}",
        "RSI": round(last["rsi7"], 1),
        "Tendance": mtf_trend.upper(),
        "Force": f"{mtf_strength}%",
        "Heure": last["time"].strftime("%Y-%m-%d %H:%M"),
        "_confidence_val": confidence,
        "_time_raw": last["time"]
    }

# ----------------------
# Scan parallélisé
# ----------------------
def scan_parallel(pairs: List[str], tfs: List[str], candles_count: int, max_workers: int = 5) -> List[Dict]:
    """Scan parallélisé pour performances optimales."""
    results = []
    tasks = [(pair, tf) for pair in pairs for tf in tfs]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(analyze_pair, pair, tf, candles_count): (pair, tf)
            for pair, tf in tasks
        }
        
        completed = 0
        total = len(tasks)
        
        for future in as_completed(future_to_task):
            completed += 1
            progress_bar.progress(completed / total)
            status_text.text(f"Analyse en cours... {completed}/{total}")
            
            try:
                result = future.result(timeout=10)
                if result and result["Signal"]:
                    results.append(result)
            except Exception:
                pass
        
        progress_bar.empty()
        status_text.empty()
    
    return results

# ----------------------
# Interface utilisateur
# ----------------------
st.sidebar.header("⚙️ Configuration du Scanner")

selected_pairs = st.sidebar.multiselect(
    "Paires à scanner :",
    PAIRS_DEFAULT,
    default=PAIRS_DEFAULT[:10]
)

max_pairs = st.sidebar.number_input(
    "Nombre max de paires :",
    min_value=1,
    max_value=len(selected_pairs) if selected_pairs else 28,
    value=min(10, len(selected_pairs)) if selected_pairs else 10
)

selected_tfs = st.sidebar.multiselect(
    "Timeframes :",
    ["H1", "H4", "D1"],
    default=["H4", "D1"]
)

candles_count = st.sidebar.selectbox(
    "Bougies par timeframe :",
    [100, 150, 200],
    index=1
)

max_workers = st.sidebar.slider(
    "Threads parallèles :",
    min_value=3,
    max_value=10,
    value=5,
    help="Plus = rapide mais charge API OANDA"
)

min_confidence = st.sidebar.slider(
    "Confiance minimale (%) :",
    min_value=0,
    max_value=100,
    value=40,
    help="Filtrer les signaux faibles"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Rafraîchissement")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5min)", help="Scan automatique toutes les 5 minutes")
refresh_interval = st.sidebar.selectbox("Intervalle (min) :", [3, 5, 10, 15], index=1)

st.sidebar.markdown("---")
scan_button = st.sidebar.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True)

# ----------------------
# Fonction session de marché
# ----------------------
def get_market_session():
    """Détermine la session de marché active et sa qualité."""
    from datetime import datetime
    import pytz
    
    # Heure actuelle à Tunis
    tz_tunis = pytz.timezone('Africa/Tunis')
    now = datetime.now(tz_tunis)
    hour = now.hour
    minute = now.minute
    current_time = hour + minute / 60
    
    # Définition des sessions (heures de Tunis)
    sessions = {
        "Tokyo": {"start": 1, "end": 10, "quality": "🟡 Moyenne", "pairs": "JPY", "color": "orange"},
        "Londres": {"start": 9, "end": 18, "quality": "🟢 Excellente", "pairs": "EUR, GBP", "color": "green"},
        "New York": {"start": 14, "end": 23, "quality": "🟢 Excellente", "pairs": "USD", "color": "green"},
        "Overlap": {"start": 14, "end": 18, "quality": "🔥 Maximum", "pairs": "Toutes", "color": "red"}
    }
    
    active_sessions = []
    best_quality = "🔵 Faible"
    best_color = "blue"
    
    # Vérifier les sessions actives
    for name, info in sessions.items():
        if info["start"] <= current_time < info["end"]:
            active_sessions.append(name)
            if info["quality"] == "🔥 Maximum":
                best_quality = info["quality"]
                best_color = info["color"]
            elif info["quality"] == "🟢 Excellente" and best_quality != "🔥 Maximum":
                best_quality = info["quality"]
                best_color = info["color"]
            elif info["quality"] == "🟡 Moyenne" and best_quality == "🔵 Faible":
                best_quality = info["quality"]
                best_color = info["color"]
    
    if not active_sessions:
        best_quality = "🔵 Faible"
        best_color = "blue"
        active_sessions = ["Marché calme"]
    
    return {
        "sessions": ", ".join(active_sessions),
        "quality": best_quality,
        "color": best_color,
        "hour": now.strftime("%H:%M")
    }

# ----------------------
# Statistiques d'en-tête avec indicateur de session
# ----------------------
market_info = get_market_session()

# Alerte session
if market_info["quality"] == "🔥 Maximum":
    st.success(f"⚡ **SESSION OPTIMALE ACTIVE** - {market_info['sessions']} - Qualité: {market_info['quality']}")
elif market_info["quality"] == "🟢 Excellente":
    st.info(f"✅ **Session active** - {market_info['sessions']} - Qualité: {market_info['quality']}")
elif market_info["quality"] == "🟡 Moyenne":
    st.warning(f"⏰ **Session modérée** - {market_info['sessions']} - Qualité: {market_info['quality']}")
else:
    st.error(f"💤 **Marché calme** - Peu de volatilité attendue - Qualité: {market_info['quality']}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Paires disponibles", len(PAIRS_DEFAULT))
col2.metric("Paires sélectionnées", len(selected_pairs) if selected_pairs else 0)
col3.metric("Timeframes actifs", len(selected_tfs) if selected_tfs else 0)
col4.metric("Indicateurs", "HMA20 + RSI7 + ATR14")
col5.metric("Heure Tunis", market_info["hour"], market_info["sessions"])

st.markdown("---")

# ----------------------
# Scan principal
# ----------------------
if scan_button or auto_refresh:
    if not selected_pairs or not selected_tfs:
        st.error("⚠️ Veuillez sélectionner au moins une paire et un timeframe")
        st.stop()
    
    if auto_refresh and not scan_button:
        countdown = st.empty()
        for remaining in range(refresh_interval * 60, 0, -1):
            mins, secs = divmod(remaining, 60)
            countdown.info(f"⏱️ Prochain scan dans {mins:02d}:{secs:02d}")
            time.sleep(1)
        countdown.empty()
        st.rerun()
    
    with st.spinner("🔍 Scan en cours..."):
        start_time = time.time()
        pairs_to_scan = selected_pairs[:max_pairs]
        
        results = scan_parallel(pairs_to_scan, selected_tfs, candles_count, max_workers)
        
        # Filtrer par confiance
        results = [r for r in results if r["_confidence_val"] >= min_confidence]
        
        elapsed = time.time() - start_time
    
    # Résultats
    st.success(f"✅ Scan terminé en **{elapsed:.1f}s** - **{len(results)} signaux** trouvés")
    
    if results:
        # Tri par confiance décroissante
        results.sort(key=lambda x: x["_confidence_val"], reverse=True)
        
        # Préparation données affichage
        df_display = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ])
        
        # Coloration des signaux
        def highlight_signal(row):
            if "ACHAT" in str(row["Signal"]):
                return ['background-color: rgba(0, 255, 0, 0.1)'] * len(row)
            elif "VENTE" in str(row["Signal"]):
                return ['background-color: rgba(255, 0, 0, 0.1)'] * len(row)
            return [''] * len(row)
        
        # Affichage tableau principal
        st.subheader("📋 Signaux détectés")
        st.dataframe(
            df_display.style.apply(highlight_signal, axis=1),
            use_container_width=True,
            height=500
        )
        
        # Top signaux
        st.markdown("---")
        st.subheader("🏆 Top 5 Signaux par Confiance")
        
        cols = st.columns(5)
        for idx, result in enumerate(results[:5]):
            with cols[idx]:
                signal_emoji = "🟢" if "ACHAT" in result["Signal"] else "🔴"
                st.metric(
                    f"{signal_emoji} {result['Instrument']}",
                    f"{result['Prix']}",
                    f"{result['TF']} - {result['Confiance']:.0f}%"
                )
                st.caption(f"SL: {result['SL']} | TP: {result['TP']}")
                st.caption(f"R:R {result['R:R']} | RSI {result['RSI']}")
        
        # Analyse par timeframe
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Répartition par Timeframe")
            tf_counts = df_display["TF"].value_counts()
            st.bar_chart(tf_counts)
        
        with col2:
            st.subheader("📈 Répartition Achat/Vente")
            signal_counts = df_display["Signal"].value_counts()
            st.bar_chart(signal_counts)
        
        # Export CSV
        st.markdown("---")
        csv = df_display.to_csv(index=False).encode('utf-8')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="📥 Télécharger les signaux (CSV)",
            data=csv,
            file_name=f"forex_signals_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("ℹ️ Aucun signal détecté avec les critères actuels.")
        st.markdown("""
        **Suggestions :**
        - Réduire la confiance minimale
        - Augmenter le nombre de paires scannées
        - Ajouter d'autres timeframes
        - Réessayer dans quelques minutes
        """)

else:
    st.info("👈 Configurez le scanner dans la barre latérale et cliquez sur **LANCER LE SCAN**")
    
    st.markdown("---")
    st.markdown("""
    ### 📚 Guide d'utilisation
    
    **Stratégie :**
    - Signal ACHAT : HMA20 devient haussière + RSI7 > 50 + tendance MTF haussière
    - Signal VENTE : HMA20 devient baissière + RSI7 < 50 + tendance MTF baissière
    
    **Niveaux :**
    - SL calculé à 2x ATR du prix d'entrée
    - TP calculé à 3x ATR du prix d'entrée
    - Ratio risque/récompense ~1:1.5
    
    **Confiance :**
    - Score basé sur force RSI (40%) + force tendance MTF (60%)
    - Recommandé : ≥ 40% pour signaux fiables
    
    **Performance :**
    - Scan parallélisé : 5-10x plus rapide que séquentiel
    - Cache 60s pour réduire appels API
    - Auto-refresh optionnel pour trading actif
    
    **⏰ Meilleures heures de trading (Tunis) :**
    - 🔥 **14h-18h** : Overlap Londres-NY (OPTIMAL)
    - 🟢 **9h-18h** : Session Londres
    - 🟢 **14h-23h** : Session New York
    - 🟡 **1h-10h** : Session Tokyo (JPY uniquement)
    - 🔵 **23h-1h** : Marché calme (éviter)
    """)
