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
st.title("📊 Forex Multi-Timeframe Scanner Pro - Signaux Instantanés")
st.write("✨ Détection des changements HMA + croisements RSI en temps réel")

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
@st.cache_data(ttl=30)  # Cache réduit à 30s pour signaux instantanés
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

@st.cache_data(ttl=60)
def check_mtf_trend(pair: str, tf: str) -> Dict[str, any]:
    """Analyse tendance multi-timeframe SIMPLIFIÉE.
    H1 → validé par H4
    H4 → validé par D1  
    D1 → validé par W
    """
    map_higher = {
        "H1": "H4",
        "H4": "D1",
        "D1": "W"
    }
    
    higher = map_higher.get(tf)
    if not higher:
        return {"trend": "neutral", "strength": 0, "aligned": False}
    
    df = get_candles(pair, higher, count=100)
    if df.empty or len(df) < 50:
        return {"trend": "neutral", "strength": 0, "aligned": False}
    
    close = df["close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    price = close.iloc[-1]
    
    # Calcul de la force de la tendance
    distance_pct = abs((ema20 - ema50) / ema50) * 100
    
    if ema20 > ema50 and price > ema20:
        strength = min(distance_pct * 10, 100)
        return {"trend": "bullish", "strength": round(strength, 1), "aligned": True}
    elif ema20 < ema50 and price < ema20:
        strength = min(distance_pct * 10, 100)
        return {"trend": "bearish", "strength": round(strength, 1), "aligned": True}
    
    return {"trend": "neutral", "strength": 0, "aligned": False}

# ----------------------
# Analyse INSTANTANÉE d'une paire
# ----------------------
def analyze_pair(pair: str, tf: str, candles_count: int) -> Optional[Dict]:
    """
    Détection INSTANTANÉE des signaux selon vos critères EXACTS :
    
    1. HMA20 change de couleur (rouge→vert = ACHAT, vert→rouge = VENTE)
    2. RSI7 dépasse légèrement 50 (croisement sur les 2 dernières bougies)
    3. MTF aligné avec timeframe supérieur
    """
    df = get_candles(pair, tf, count=candles_count)
    if df.empty or len(df) < 30:
        return None
    
    df = df.sort_values("time").reset_index(drop=True)
    
    # Calcul indicateurs
    df["hma20"] = hma(df["close"], 20)
    df["rsi7"] = rsi(df["close"], 7)
    df["atr14"] = atr(df, 14)
    
    # Direction HMA
    df["hma_up"] = df["hma20"] > df["hma20"].shift(1)
    
    # ============================================
    # CRITÈRE 1 : HMA change de couleur MAINTENANT
    # ============================================
    last = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Changement HMA sur la DERNIÈRE bougie uniquement
    hma_turned_bullish = last["hma_up"] and not prev["hma_up"]
    hma_turned_bearish = not last["hma_up"] and prev["hma_up"]
    
    # ============================================
    # CRITÈRE 2 : RSI croise 50 (légèrement)
    # ============================================
    # ACHAT : RSI croise 50 vers le HAUT (au-dessus de 50)
    # VENTE : RSI croise 50 vers le BAS (en-dessous de 50)
    rsi_crossed_up = (prev["rsi7"] <= 50 and last["rsi7"] > 50)
    
    rsi_crossed_down = (prev["rsi7"] >= 50 and last["rsi7"] < 50)
    
    # Distance au croisement (pour mesurer "légèrement")
    rsi_distance = abs(last["rsi7"] - 50)
    
    # ============================================
    # CRITÈRE 3 : MTF aligné
    # ============================================
    mtf_info = check_mtf_trend(pair, tf)
    mtf_aligned = mtf_info["aligned"]
    mtf_trend = mtf_info["trend"]
    mtf_strength = mtf_info["strength"]
    
    # ============================================
    # LOGIQUE DE SIGNAL INSTANTANÉ
    # ============================================
    signal = None
    confidence = 0
    
    # SIGNAL ACHAT : HMA devient verte + RSI croise 50 vers le haut + MTF haussier
    if hma_turned_bullish and rsi_crossed_up and mtf_trend == "bullish" and mtf_aligned:
        signal = "🟢 ACHAT"
        
        # Confiance basée sur :
        # - Force RSI au-dessus de 50 (max 40 points)
        # - Force tendance MTF (max 60 points)
        # - Bonus si croisement RSI proche de 50 (léger)
        rsi_strength = min((last["rsi7"] - 50) * 2, 40)
        proximity_bonus = max(0, 10 - rsi_distance) * 2  # Bonus si RSI proche de 50
        
        confidence = rsi_strength + (mtf_strength * 0.6) + proximity_bonus
    
    # SIGNAL VENTE : HMA devient rouge + RSI croise 50 vers le bas + MTF baissier
    elif hma_turned_bearish and rsi_crossed_down and mtf_trend == "bearish" and mtf_aligned:
        signal = "🔴 VENTE"
        
        rsi_strength = min((50 - last["rsi7"]) * 2, 40)
        proximity_bonus = max(0, 10 - rsi_distance) * 2
        
        confidence = rsi_strength + (mtf_strength * 0.6) + proximity_bonus
    
    confidence = min(confidence, 100)
    
    if signal is None:
        return None
    
    # ============================================
    # Calcul niveaux SL/TP basés sur ATR
    # ============================================
    atr_value = last["atr14"]
    price = last["close"]
    
    if "ACHAT" in signal:
        sl = price - (2 * atr_value)
        tp = price + (3 * atr_value)
    else:
        sl = price + (2 * atr_value)
        tp = price - (3 * atr_value)
    
    risk_pips = abs(price - sl)
    reward_pips = abs(tp - price)
    rr_ratio = reward_pips / risk_pips if risk_pips > 0 else 0
    
    # Temps écoulé depuis le signal (en minutes)
    now = datetime.now(pytz.UTC)
    signal_time = last["time"].replace(tzinfo=pytz.UTC)
    minutes_ago = (now - signal_time).total_seconds() / 60
    
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
        "Il y a": f"{int(minutes_ago)}min",
        "Heure": last["time"].strftime("%Y-%m-%d %H:%M"),
        "_confidence_val": confidence,
        "_time_raw": last["time"],
        "_minutes_ago": minutes_ago
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
# Fonction session de marché
# ----------------------
def get_market_session():
    """Détermine la session de marché active et sa qualité."""
    tz_tunis = pytz.timezone('Africa/Tunis')
    now = datetime.now(tz_tunis)
    hour = now.hour
    minute = now.minute
    current_time = hour + minute / 60
    
    sessions = {
        "Tokyo": {"start": 1, "end": 10, "quality": "🟡 Moyenne", "pairs": "JPY", "color": "orange"},
        "Londres": {"start": 9, "end": 18, "quality": "🟢 Excellente", "pairs": "EUR, GBP", "color": "green"},
        "New York": {"start": 14, "end": 23, "quality": "🟢 Excellente", "pairs": "USD", "color": "green"},
        "Overlap": {"start": 14, "end": 18, "quality": "🔥 Maximum", "pairs": "Toutes", "color": "red"}
    }
    
    active_sessions = []
    best_quality = "🔵 Faible"
    best_color = "blue"
    
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
# Interface utilisateur
# ----------------------
st.sidebar.header("⚙️ Configuration du Scanner")

with st.sidebar.expander("🔧 Filtrer les paires (optionnel)", expanded=False):
    selected_pairs = st.multiselect(
        "Désélectionner les paires à ignorer :",
        PAIRS_DEFAULT,
        default=PAIRS_DEFAULT
    )

if not selected_pairs:
    selected_pairs = PAIRS_DEFAULT

selected_tfs = st.sidebar.multiselect(
    "Timeframes :",
    ["H1", "H4", "D1"],
    default=["H1", "H4", "D1"],
    help="H1 validé par H4 | H4 validé par D1 | D1 validé par W"
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
    value=30,
    help="Filtrer les signaux faibles"
)

# Filtre de fraîcheur des signaux
max_age_minutes = st.sidebar.slider(
    "Signaux récents uniquement (min) :",
    min_value=5,
    max_value=60,
    value=15,
    help="Ignorer les signaux plus vieux que X minutes"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Rafraîchissement")
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True, help="Scan automatique")
refresh_interval = st.sidebar.selectbox("Intervalle (min) :", [1, 2, 3, 5], index=2)

st.sidebar.markdown("---")
scan_button = st.sidebar.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True)

# ----------------------
# Statistiques d'en-tête
# ----------------------
market_info = get_market_session()

if market_info["quality"] == "🔥 Maximum":
    st.success(f"⚡ **SESSION OPTIMALE ACTIVE** - {market_info['sessions']} - {market_info['quality']}")
elif market_info["quality"] == "🟢 Excellente":
    st.info(f"✅ **Session active** - {market_info['sessions']} - {market_info['quality']}")
elif market_info["quality"] == "🟡 Moyenne":
    st.warning(f"⏰ **Session modérée** - {market_info['sessions']} - {market_info['quality']}")
else:
    st.error(f"💤 **Marché calme** - {market_info['quality']}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Paires scannées", len(selected_pairs))
col2.metric("Timeframes", f"{len(selected_tfs)}/3")
col3.metric("Validations", "HMA+RSI+MTF")
col4.metric("Fraîcheur", f"< {max_age_minutes}min")
col5.metric("Heure Tunis", market_info["hour"])

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
        results = scan_parallel(selected_pairs, selected_tfs, candles_count, max_workers)
        
        # Filtrer par confiance ET fraîcheur
        results = [
            r for r in results 
            if r["_confidence_val"] >= min_confidence 
            and r["_minutes_ago"] <= max_age_minutes
        ]
        
        elapsed = time.time() - start_time
    
    total_analyzed = len(selected_pairs) * len(selected_tfs)
    st.success(f"✅ Scan terminé en **{elapsed:.1f}s** - **{total_analyzed} analyses** - **{len(results)} signaux instantanés**")
    
    if results:
        # Tri par fraîcheur (plus récent en premier)
        results.sort(key=lambda x: x["_minutes_ago"])
        
        # Marqueur pour signaux très récents (< 5 min)
        for result in results:
            if result["_minutes_ago"] <= 5:
                result["Signal"] = "⚡ " + result["Signal"]
        
        df_display = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ])
        
        def highlight_signal(row):
            if "ACHAT" in str(row["Signal"]):
                return ['background-color: rgba(0, 255, 0, 0.15)'] * len(row)
            elif "VENTE" in str(row["Signal"]):
                return ['background-color: rgba(255, 0, 0, 0.15)'] * len(row)
            return [''] * len(row)
        
        st.subheader("📋 Signaux Instantanés Détectés")
        st.dataframe(
            df_display.style.apply(highlight_signal, axis=1),
            use_container_width=True,
            height=500
        )
        
        # Top 5 signaux
        st.markdown("---")
        st.subheader("🏆 Top 5 Signaux Plus Récents")
        
        cols = st.columns(min(5, len(results)))
        for idx, result in enumerate(results[:5]):
            with cols[idx]:
                signal_emoji = "🟢" if "ACHAT" in result["Signal"] else "🔴"
                st.metric(
                    f"{signal_emoji} {result['Instrument']}",
                    f"{result['Prix']}",
                    f"{result['TF']} - {result['Confiance']:.0f}%"
                )
                st.caption(f"⏱️ Il y a {result['Il y a']}")
                st.caption(f"SL: {result['SL']} | TP: {result['TP']}")
                st.caption(f"R:R {result['R:R']} | RSI {result['RSI']}")
        
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
        st.info("ℹ️ Aucun signal instantané détecté pour le moment.")
        
        with st.expander("🔍 Pourquoi aucun signal ?"):
            st.markdown(f"""
            **Critères STRICTS pour un signal instantané :**
            
            ✅ **1. HMA20 change de couleur** sur la DERNIÈRE bougie :
            - Rouge → Vert = Signal ACHAT
            - Vert → Rouge = Signal VENTE
            
            ✅ **2. RSI7 croise 50** (sur les 2 dernières bougies) :
            - ACHAT : RSI passe de ≤50 à >50 (au-dessus)
            - VENTE : RSI passe de ≥50 à <50 (en-dessous)
            
            ✅ **3. MTF aligné** :
            - H1 → H4 doit être haussier/baissier
            - H4 → D1 doit être haussier/baissier
            - D1 → W doit être haussier/baissier
            
            ✅ **4. Signal récent** : < {max_age_minutes} minutes
            
            **Actions suggérées :**
            - 🔽 Réduire la confiance minimale à 0%
            - ⏰ Augmenter la fraîcheur à 30-60 minutes
            - 🔄 Attendre la prochaine bougie (signaux apparaissent à la clôture)
            - 📊 Vérifier que vous êtes dans une session active
            
            **Note :** Ces signaux sont RARES car les 3 critères doivent être simultanés !
            """)

else:
    st.info("👈 Configurez le scanner et cliquez sur **LANCER LE SCAN**")
    
    st.markdown("---")
    st.markdown("""
    ### 📚 Guide - Signaux Instantanés
    
    **Stratégie Précise :**
    
    🎯 **Signal ACHAT détecté quand :**
    1. HMA20 passe du rouge au vert (sur la dernière bougie)
    2. RSI7 croise la ligne 50 vers le HAUT (passe au-dessus de 50)
    3. Tendance MTF haussière sur TF supérieur
    
    🎯 **Signal VENTE détecté quand :**
    1. HMA20 passe du vert au rouge (sur la dernière bougie)
    2. RSI7 croise la ligne 50 vers le BAS (passe en-dessous de 50)
    3. Tendance MTF baissière sur TF supérieur
    
    **Niveaux automatiques :**
    - SL : 2x ATR
    - TP : 3x ATR
    - R:R : ~1:1.5
    
    **Confiance (score) :**
    - 40% : Force RSI
    - 60% : Force tendance MTF
    - Bonus : Proximité du croisement RSI à 50
    
    **💡 Conseils d'utilisation :**
    - Auto-refresh 2-3 min pour capturer les signaux en temps réel
    - Vérifier signaux sur TradingView avant d'entrer en position
    - Privilégier les signaux avec ⚡ (très récents, < 5 min)
    - Augmenter la fraîcheur si vous voulez voir l'historique récent
    """)
