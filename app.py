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
@st.cache_data(ttl=30)
def get_candles(pair: str, tf: str, count: int = 200, include_incomplete: bool = False) -> pd.DataFrame:
    """Télécharge les bougies OANDA avec gestion d'erreur robuste.
    
    Args:
        include_incomplete: Si True, inclut la dernière bougie même si incomplete (pour analyse en temps réel)
    """
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
            # Si include_incomplete=True, garder aussi la dernière bougie incomplete
            if not c.get("complete", True) and not include_incomplete:
                continue
            try:
                records.append({
                    "time": c["time"],
                    "open": float(c["mid"]["o"]),
                    "high": float(c["mid"]["h"]),
                    "low": float(c["mid"]["l"]),
                    "close": float(c["mid"]["c"]),
                    "volume": int(c.get("volume", 0)),
                    "complete": c.get("complete", True)
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

@st.cache_data(ttl=30)
def check_mtf_trend(pair: str, tf: str) -> Dict[str, any]:
    """Analyse tendance multi-timeframe avec force du signal."""
    map_higher = {
        "H1": "H4",
        "H4": "D1",
        "D1": "W"
    }
    
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

# ----------------------
# Analyse d'une paire (optimisée)
# ----------------------
def analyze_pair(pair: str, tf: str, candles_count: int, max_candles_back: int = 3) -> Optional[Dict]:
    """Analyse complète avec gestion d'erreur robuste et filtre de fraîcheur."""
    df = get_candles(pair, tf, count=candles_count)
    if df.empty or len(df) < 30:
        return None
    
    df = df.sort_values("time").reset_index(drop=True)
    
    # Calcul indicateurs
    df["hma20"] = hma(df["close"], 20)
    df["rsi7"] = rsi(df["close"], 7)
    df["atr14"] = atr(df, 14)
    
    # Signaux HMA - Direction actuelle
    df["hma_up"] = df["hma20"] > df["hma20"].shift(1)
    
    last_n = df.tail(max_candles_back) if max_candles_back < 999 else df.tail(3)
    
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
    
    # Indicateur si la bougie est en cours ou complète
    is_incomplete = not last.get("complete", True) if "complete" in last else False
    
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
            confidence *= 1.2
    elif sell:
        signal = "🔴 VENTE"
        rsi_strength = (50 - last["rsi7"]) / 50 * 100
        confidence = (rsi_strength * 0.4 + mtf_strength * 0.6)
        if has_rsi_confirmation:
            confidence *= 1.2
    
    confidence = min(confidence, 100)
    
    if signal is None:
        return None
    
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
        "Signal": signal + (" 🔄" if is_incomplete else ""),
        "Confiance": round(confidence, 1),
        "Prix": round(price, 5),
        "SL": round(sl, 5),
        "TP": round(tp, 5),
        "R:R": f"1:{round(rr_ratio, 1)}",
        "RSI": round(last["rsi7"], 1),
        "Tendance": mtf_trend.upper(),
        "Force": f"{mtf_strength}%",
        "Heure": last["time"].strftime("%Y-%m-%d %H:%M") + (" ⏳" if is_incomplete else ""),
        "_confidence_val": confidence,
        "_time_raw": last["time"],
        "_incomplete": is_incomplete
    }

# ----------------------
# Scan parallélisé
# ----------------------
def scan_parallel(pairs: List[str], tfs: List[str], candles_count: int, max_workers: int = 5, max_candles_back: int = 3, include_incomplete: bool = False) -> List[Dict]:
    """Scan parallélisé pour performances optimales."""
    results = []
    tasks = [(pair, tf) for pair in pairs for tf in tfs]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(analyze_pair, pair, tf, candles_count, max_candles_back, include_incomplete): (pair, tf)
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
    value=20,
    help="Filtrer les signaux faibles - Réduire pour voir plus de signaux"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Mode de scan")

scan_mode = st.sidebar.radio(
    "Type de bougies :",
    ["Bougies complètes uniquement (recommandé)", "Inclure bougie en cours (temps réel)"],
    help="Bougies complètes = signaux fiables mais avec délai. Temps réel = signaux instantanés mais peuvent changer"
)

include_incomplete = (scan_mode == "Inclure bougie en cours (temps réel)")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⏱️ Fraîcheur des signaux")

signal_freshness = st.sidebar.selectbox(
    "Ne garder que les signaux de :",
    ["Dernière bougie uniquement", "2 dernières bougies", "3 dernières bougies", "Toutes les bougies"],
    index=1,
    help="Filtrer les signaux trop anciens"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔄 Rafraîchissement")
auto_refresh = st.sidebar.checkbox("Auto-refresh (5min)", help="Scan automatique toutes les 5 minutes")
refresh_interval = st.sidebar.selectbox("Intervalle (min) :", [3, 5, 10, 15], index=1)

st.sidebar.markdown("---")
scan_button = st.sidebar.button("🚀 LANCER LE SCAN", type="primary", use_container_width=True)

# ----------------------
# Statistiques d'en-tête
# ----------------------
market_info = get_market_session()

if market_info["quality"] == "🔥 Maximum":
    st.success(f"⚡ **SESSION OPTIMALE ACTIVE** - {market_info['sessions']} - Qualité: {market_info['quality']}")
elif market_info["quality"] == "🟢 Excellente":
    st.info(f"✅ **Session active** - {market_info['sessions']} - Qualité: {market_info['quality']}")
elif market_info["quality"] == "🟡 Moyenne":
    st.warning(f"⏰ **Session modérée** - {market_info['sessions']} - Qualité: {market_info['quality']}")
else:
    st.error(f"💤 **Marché calme** - Peu de volatilité attendue - Qualité: {market_info['quality']}")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Paires scannées", len(selected_pairs) if selected_pairs else 28)
col2.metric("Timeframes", f"{len(selected_tfs) if selected_tfs else 0}/3")
col3.metric("Validations MTF", "H1→H4 | H4→D1 | D1→W")
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
    
    freshness_map = {
        "Dernière bougie uniquement": 1,
        "2 dernières bougies": 2,
        "3 dernières bougies": 3,
        "Toutes les bougies": 999
    }
    max_candles_ago = freshness_map.get(signal_freshness, 2)
    
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
        pairs_to_scan = selected_pairs
        
        results = scan_parallel(pairs_to_scan, selected_tfs, candles_count, max_workers, max_candles_ago, include_incomplete)
        
        results = [r for r in results if r["_confidence_val"] >= min_confidence]
        
        elapsed = time.time() - start_time
    
    total_analyzed = len(pairs_to_scan) * len(selected_tfs)
    freshness_text = signal_freshness.lower()
    
    mode_text = "temps réel 🔄" if include_incomplete else "bougies complètes"
    st.success(f"✅ Scan terminé en **{elapsed:.1f}s** - **{total_analyzed} analyses** - **{len(results)} signaux** ({mode_text}, {freshness_text}, confiance ≥ {min_confidence}%)")
    
    if results:
        # Avertissement si mode temps réel activé
        if include_incomplete:
            st.warning("⚠️ **Mode temps réel activé** : Les signaux avec 🔄 sont basés sur la bougie en cours et peuvent changer avant la clôture !")
        
        # Séparer les résultats par timeframe
        results_by_tf = {}
        for result in results:
            tf = result["TF"]
            if tf not in results_by_tf:
                results_by_tf[tf] = []
            results_by_tf[tf].append(result)
        
        # Afficher un tableau par timeframe
        tf_order = ["H1", "H4", "D1"]
        
        for tf in tf_order:
            if tf not in results_by_tf:
                continue
                
            tf_results = results_by_tf[tf]
            
            # Trier par confiance décroissante
            tf_results.sort(key=lambda x: x["_confidence_val"], reverse=True)
            
            # Marquer le plus récent
            most_recent_time = max(r["_time_raw"] for r in tf_results)
            for result in tf_results:
                if result["_time_raw"] == most_recent_time:
                    result["Signal"] = "⭐ " + result["Signal"]
            
            # Créer le dataframe pour ce timeframe
            df_tf = pd.DataFrame([
                {k: v for k, v in r.items() if not k.startswith("_")}
                for r in tf_results
            ])
            
            # Affichage du tableau
            st.markdown(f"### 📊 Timeframe {tf} - {len(tf_results)} signal(s)")
            
            def highlight_signal(row):
                if "ACHAT" in str(row["Signal"]):
                    return ['background-color: rgba(0, 255, 0, 0.15)'] * len(row)
                elif "VENTE" in str(row["Signal"]):
                    return ['background-color: rgba(255, 0, 0, 0.15)'] * len(row)
                return [''] * len(row)
            
            st.dataframe(
                df_tf.style.apply(highlight_signal, axis=1),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("---")
        
        # Top 5 signaux globaux
        st.subheader("🏆 Top 5 Signaux par Confiance (tous TF)")
        
        results_sorted = sorted(results, key=lambda x: x["_confidence_val"], reverse=True)
        
        cols = st.columns(min(5, len(results_sorted)))
        for idx, result in enumerate(results_sorted[:5]):
            with cols[idx]:
                signal_emoji = "🟢" if "ACHAT" in result["Signal"] else "🔴"
                st.metric(
                    f"{signal_emoji} {result['Instrument']}",
                    f"{result['Prix']}",
                    f"{result['TF']} - {result['Confiance']:.0f}%"
                )
                st.caption(f"SL: {result['SL']} | TP: {result['TP']}")
                st.caption(f"R:R {result['R:R']} | RSI {result['RSI']}")
        
        # Statistiques
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Répartition par Timeframe")
            tf_counts = pd.Series({tf: len(results_by_tf.get(tf, [])) for tf in tf_order})
            st.bar_chart(tf_counts)
        
        with col2:
            st.subheader("📈 Répartition Achat/Vente")
            all_signals = [r["Signal"].replace("⭐ ", "") for r in results]
            signal_counts = pd.Series(all_signals).value_counts()
            st.bar_chart(signal_counts)
        
        # Export CSV
        st.markdown("---")
        df_all = pd.DataFrame([
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ])
        csv = df_all.to_csv(index=False).encode('utf-8')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        st.download_button(
            label="📥 Télécharger tous les signaux (CSV)",
            data=csv,
            file_name=f"forex_signals_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
    else:
        st.info("ℹ️ Aucun signal détecté avec les critères actuels.")
        
        with st.expander("🔍 Diagnostic - Pourquoi aucun signal ?"):
            st.markdown("""
            **Critères requis pour un signal :**
            1. ✅ HMA20 devient haussière/baissière (ou est déjà dans cette direction)
            2. ✅ RSI7 au-dessus/en-dessous de 50
            3. ✅ Tendance MTF alignée (H1→H4, H4→D1, D1→W)
            4. ✅ Confiance ≥ seuil défini
            
            **Actions à essayer :**
            - 🔽 Réduire la **confiance minimale** à 0% (voir TOUS les signaux)
            - 🔄 Attendre la prochaine bougie (les signaux apparaissent à la clôture)
            - ⏰ Vérifier que vous êtes dans une session active (Londres/NY)
            - 📊 Les marchés peuvent être en consolidation (aucune tendance claire)
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
    - Recommandé : ≥ 20% pour signaux fiables
    
    **⏰ Meilleures heures de trading (Tunis) :**
    - 🔥 **14h-18h** : Overlap Londres-NY (OPTIMAL)
    - 🟢 **9h-18h** : Session Londres
    - 🟢 **14h-23h** : Session New York
    - 🟡 **1h-10h** : Session Tokyo (JPY uniquement)
    - 🔵 **23h-1h** : Marché calme (éviter)
    
    **⭐ = Signal le plus récent du timeframe**
    **🔄 = Signal basé sur bougie en cours (mode temps réel)**
    **⏳ = Heure de la bougie en cours**
    """)
   
