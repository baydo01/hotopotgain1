import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
import json
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- AI & ML ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import plotly.graph_objects as go
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Meta-Learner", layout="wide")
st.title("🏦 Hedge Fund AI: Otonom Jüri & Stacking")

# =============================================================================
# 1. AYARLAR
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"

# SENİN İSTEDİĞİN LİSTE
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

with st.sidebar:
    st.header("⚙️ Ayarlar")
    use_ga = st.checkbox("Genetic Algoritma (GA) Aktif", value=True) # Varsayılan Açık
    ga_gens = st.number_input("GA Jenerasyon Sayısı", 1, 20, 5) # Varsayılan 5
    st.info("Bu model; Senin Puanın, HMM ve AI modellerini Lojistik Regresyon ile yarıştırır.")

# =============================================================================
# 2. GOOGLE SHEETS BAĞLANTISI
# =============================================================================

def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except: pass
    elif os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    
    if not creds: return None
    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except: return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                         "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                         "Son_Islem_Log", "Son_Islem_Zamani"]
        
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            defaults = []
            for t in TARGET_COINS:
                defaults.append([t, "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"])
            for d in defaults: sheet.append_row(d)
            time.sleep(2)
    except: pass
        
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df = df[df['Ticker'].astype(str).str.len() > 3]
    
    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    return df, sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy(); df_export = df_export.astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except: pass

# =============================================================================
# 3. AI MOTORU (SENİN PUANIN + LOGREG + GA)
# =============================================================================

def apply_kalman_filter(prices):
    n_iter = len(prices); sz = (n_iter,); Q = 1e-5; R = 0.01 ** 2
    xhat = np.zeros(sz); P = np.zeros(sz); xhatminus = np.zeros(sz); Pminus = np.zeros(sz); K = np.zeros(sz)
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]; Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R); xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return pd.Series(xhat, index=prices.index)

def calculate_heuristic_score(df):
    """Senin 5 Adımlı Puanlama Sistemin"""
    if len(df) < 150: return pd.Series(0.0, index=df.index)
    
    # 1. Kısa Vade (5 Günlük Yön)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    # 2. Orta Vade (30 Günlük Yön)
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    # 3. Uzun Vade
    s3 = np.where(df['close'] > df['close'].rolling(150).mean(), 1, -1)
    # 4. Volatilite
    vol = df['close'].pct_change().rolling(20).std()
    s4 = np.where(vol < vol.shift(1), 1, -1)
    # 5. Momentum
    s5 = np.sign(df['close'].diff(10).fillna(0))
    
    return (s1 + s2 + s3 + s4 + s5) / 5.0 # -1 ile 1 arası

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df) < 150: return None
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W': df_res = df.resample('W').agg(agg).dropna()
    else: df_res = df.copy()
    
    if len(df_res) < 150: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    
    # TARGET: Gelecek 1 bar arttı mı? (1/0) -> LogReg bunu öğrenecek
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_meta_learner(df, rf_depth=5, xgb_params=None):
    """
    JÜRİ EĞİTİMİ & VALIDATION SİMÜLASYONU
    """
    test_size = 60
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    # 1. Alt Modelleri Eğit
    X_train = train[['log_ret', 'range', 'heuristic']]
    y_train = train['target']
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=rf_depth, random_state=42).fit(X_train, y_train)
    
    if xgb_params is None: xgb_params = {'n_estimators':50, 'max_depth':3, 'learning_rate':0.1}
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params).fit(X_train, y_train)
    
    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(train[['log_ret', 'range']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None

    # 2. META-MODEL (JÜRİ) EĞİTİMİ
    hmm_pred = np.zeros(len(train))
    if hmm:
        probs = hmm.predict_proba(X_hmm)
        bull = np.argmax(hmm.means_[:, 0]); bear = np.argmin(hmm.means_[:, 0])
        hmm_pred = probs[:, bull] - probs[:, bear]
    
    # Jüriye Giren Veriler: [RF, XGB, HMM, SENİN PUANIN]
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_train)[:, 1],
        'XGB': xgb_clf.predict_proba(X_train)[:, 1],
        'HMM': hmm_pred,
        'Heuristic': train['heuristic'].values
    })
    
    # JÜRİ (LOGISTIC REGRESSION) KARARI ÖĞRENİYOR
    meta_model = LogisticRegression().fit(meta_X, y_train)
    weights = meta_model.coef_[0] # [RF_w, XGB_w, HMM_w, Heuristic_w]
    
    # 3. SİMÜLASYON (Validation - Son 60 gün)
    sim_eq = [100]; hodl_eq = [100]; cash=100; coin=0; p0 = test['close'].iloc[0]
    
    # Test seti sinyalleri
    X_hmm_t = scaler.transform(test[['log_ret', 'range']])
    hmm_p_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_s_t = hmm_p_t[:, np.argmax(hmm.means_[:,0])] - hmm_p_t[:, np.argmin(hmm.means_[:,0])] if hmm else np.zeros(len(test))
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(test[['log_ret', 'range', 'heuristic']])[:,1],
        'XGB': xgb_clf.predict_proba(test[['log_ret', 'range', 'heuristic']])[:,1],
        'HMM': hmm_s_t,
        'Heuristic': test['heuristic'].values
    })
    
    # Jüri final olasılığı (0-1 arası)
    f_probs = meta_model.predict_proba(mx_test)[:,1]
    
    for i in range(len(test)):
        pr = test['close'].iloc[i]
        s = (f_probs[i]-0.5)*2 # -1 ile 1 arası sinyal
        
        if s>0.2 and cash>0: coin=cash/pr; cash=0
        elif s<-0.2 and coin>0: cash=coin*pr; coin=0
        
        sim_eq.append(cash+coin*pr)
        hodl_eq.append((100/p0)*pr)
        
    info = {
        "weights": weights,
        "bot_eq": sim_eq[1:], "hodl_eq": hodl_eq[1:], "dates": test.index,
        "alpha": (sim_eq[-1]-hodl_eq[-1]),
        "bot_roi": (sim_eq[-1]-100),
        "hodl_roi": (hodl_eq[-1]-100),
        "conf": f_probs[-1], # Son barın olasılığı
        "my_score": test['heuristic'].iloc[-1]
    }
    
    return (f_probs[-1]-0.5)*2, info

# --- GA OPTİMİZASYONU (PARAMETRELERİ SEÇER) ---
def ga_optimize(df, n_gen=5):
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,), overwrite=True)
        creator.create('Individual', list, fitness=creator.FitnessMax, overwrite=True)
    
    toolbox = base.Toolbox()
    toolbox.register('rf', np.random.randint, 3, 10)
    toolbox.register('xgb', np.random.randint, 2, 6)
    toolbox.register('individual', tools.initCycle, creator.Individual, (toolbox.rf, toolbox.xgb), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    def eval_ind(ind):
        # Hızlı bir backtest yapıp en iyi parametreyi bulur
        sig, inf = train_meta_learner(df, rf_depth=ind[0], xgb_params={'max_depth':ind[1], 'n_estimators':30, 'learning_rate':0.1})
        return (inf['bot_roi'],)

    toolbox.register('evaluate', eval_ind)
    toolbox.register('mate', tools.cxTwoPoint); toolbox.register('mutate', tools.mutUniformInt, low=2, up=10, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=5)
    try:
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, verbose=False)
        best = tools.selBest(pop, 1)[0]
        return {'rf': best[0], 'xgb': {'max_depth':best[1], 'n_estimators':30, 'learning_rate':0.1}}
    except: return None

def analyze_ticker_smart(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None: 
        status_placeholder.error("Veri Yok")
        return "HATA", 0.0, "YOK", None
    
    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W'}
    best_s = -99; decision = "BEKLE"; win_tf = "YOK"; best_inf = None
    
    for tf_name, tf_code in timeframes.items():
        status_placeholder.text(f"{tf_name} analiz ediliyor (GA + Jüri)...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # GA ile en iyi parametreleri bul
        params = ga_optimize(df, n_gen=ga_gens) if st.session_state.get('use_ga', True) else None
        rf_d = params['rf'] if params else 5
        xgb_p = params['xgb'] if params else None
        
        # Eğit ve Tahmin Et
        sig, info = train_meta_learner(df, rf_depth=rf_d, xgb_params=xgb_p)
        
        if abs(sig) > best_s:
            best_s = abs(sig)
            win_tf = tf_name
            best_inf = info
            if sig > 0.25: decision = "AL"
            elif sig < -0.25: decision = "SAT"
            else: decision = "BEKLE"
            
    return decision, current_price, win_tf, best_inf

# =============================================================================
# 4. ARAYÜZ VE ÇALIŞTIRMA
# =============================================================================

if st.button("🚀 ANALİZİ BAŞLAT", type="primary"):
    # Session state ayarı (Sidebar'dan okumak için)
    st.session_state['use_ga'] = use_ga
    
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Hata: Portföy yüklenemedi.")
    else:
        updated = pf_df.copy()
        prog = st.progress(0)
        sim_summary = [] # Özet Rapor İçin
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            if len(str(ticker)) < 3: continue
            
            with st.expander(f"🧠 {ticker} Analiz Raporu", expanded=True):
                ph = st.empty()
                dec, prc, tf, info = analyze_ticker_smart(ticker, ph)
                
                if dec != "HATA" and info:
                    sim_summary.append({
                        "Coin": ticker,
                        "Bot Getiri": info['bot_roi'],
                        "HODL Getiri": info['hodl_roi'],
                        "Alpha": info['alpha']
                    })
                    
                    w = info['weights']
                    w_abs = np.abs(w); w_norm = w_abs / (np.sum(w_abs)+1e-9) * 100
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"### Karar: **{dec}**")
                        st.markdown(f"**Senin Puanın:** {info['my_score']:.2f}")
                        st.markdown(f"**Fiyat:** ${prc:.2f}")
                        st.markdown("**Modelin Güven Ağırlıkları (Otonom):**")
                        w_df = pd.DataFrame({
                            'Kaynak': ['RandomForest', 'XGBoost', 'HMM', 'Senin Kuralın'],
                            'Güven (%)': w_norm
                        })
                        st.dataframe(w_df, hide_index=True)
                    
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['bot_eq'], name="Bot", line=dict(color='green', width=2)))
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['hodl_eq'], name="HODL", line=dict(color='gray', dash='dot')))
                        
                        alpha_val = info['alpha']
                        color_ti = "green" if alpha_val > 0 else "red"
                        fig.update_layout(title=f"Son 60 Gün Simülasyonu | Alpha: {alpha_val:+.1f}%", title_font_color=color_ti, height=250, template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # İşlem Kaydı
                    stt = row['Durum']
                    log_msg = row['Son_Islem_Log']
                    
                    if stt == 'COIN' and dec == 'SAT':
                        amt = float(row['Miktar'])
                        if amt > 0:
                            updated.at[idx, 'Durum'] = 'CASH'; updated.at[idx, 'Nakit_Bakiye_USD'] = amt * prc
                            updated.at[idx, 'Miktar'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                            log_msg = f"SAT ({tf}) A:{alpha_val:.1f}"; updated.at[idx, 'Son_Islem_Zamani'] = time_str
                    elif stt == 'CASH' and dec == 'AL':
                        cash = float(row['Nakit_Bakiye_USD'])
                        if cash > 1:
                            updated.at[idx, 'Durum'] = 'COIN'; updated.at[idx, 'Miktar'] = cash / prc
                            updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                            log_msg = f"AL ({tf}) A:{alpha_val:.1f}"; updated.at[idx, 'Son_Islem_Zamani'] = time_str
                    
                    val = (float(updated.at[idx, 'Miktar']) * prc) if updated.at[idx, 'Durum'] == 'COIN' else float(updated.at[idx, 'Nakit_Bakiye_USD'])
                    updated.at[idx, 'Kaydedilen_Deger_USD'] = val
                    ph.success("Analiz Tamamlandı")

            prog.progress((i+1)/len(updated))
        
        save_portfolio(updated, sheet)
        
        # --- ÖZET RAPOR ---
        st.divider()
        st.subheader("🏆 Simülasyon & Model Performans Raporu")
        if sim_summary:
            sum_df = pd.DataFrame(sim_summary)
            col1, col2, col3 = st.columns(3)
            col1.metric("Ort. Bot Getirisi", f"%{sum_df['Bot Getiri'].mean():.2f}")
            col2.metric("Ort. HODL Getirisi", f"%{sum_df['HODL Getiri'].mean():.2f}")
            col3.metric("GENEL MODEL ALPHA", f"%{sum_df['Alpha'].mean():.2f}", delta_color="normal")
            st.dataframe(sum_df)
            
        st.success("✅ Tüm Analizler ve Raporlama Bitti!")

st.divider()
try:
    df_v, _ = load_and_fix_portfolio()
    if not df_v.empty:
        st.subheader("📋 Mevcut Portföy")
        st.dataframe(df_v)
        tot = df_v['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Varlık", f"${tot:.2f}")
except: pass
