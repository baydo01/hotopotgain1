import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
import json
import random
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- ML & AI ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- GENETİK ALGORİTMA ---
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: BEAST MODE", layout="wide")
st.title("🦍 Hedge Fund AI: Canavar Modu (Full GA Optimization)")

# =============================================================================
# 1. AYARLAR
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

with st.sidebar:
    st.header("⚙️ Canavar Ayarları")
    ga_gens = st.number_input("GA Jenerasyon (Evrim Sayısı)", 1, 100, 5, help="Ne kadar yüksekse o kadar iyi öğrenir ama süre uzar.")
    pop_size = st.number_input("Popülasyon Büyüklüğü", 5, 100, 20, help="Her jenerasyonda kaç farklı strateji denensin?")
    st.warning("⚠️ Bu mod ağırdır. Her coin için binlerce simülasyon yapar.")

# =============================================================================
# 2. BAĞLANTI VE OTO-KURULUM
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
# 3. VERİ İŞLEME VE HEURISTIC
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
    """Gelişmiş 5 Adımlı Puanlama"""
    if len(df) < 150: return pd.Series(0.0, index=df.index)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    # Uzun vade: 200 günlük ortalamanın üstünde mi?
    s3 = np.where(df['close'] > df['close'].rolling(200).mean(), 1, -1)
    # Volatilite
    vol = df['close'].pct_change().rolling(20).std()
    s4 = np.where(vol < vol.shift(1), 1, -1)
    # Momentum
    s5 = np.sign(df['close'].diff(10).fillna(0))
    
    # Normalize (-1 ile 1 arası)
    return (s1 + s2 + s3 + s4 + s5) / 5.0

def calculate_momentum(df, window=14):
    return np.sign(df['close'].pct_change(window).fillna(0))

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
    if df is None or len(df) < 200: return None
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W': df_res = df.resample('W').agg(agg).dropna()
    elif timeframe == 'M': df_res = df.resample('ME').agg(agg).dropna()
    else: df_res = df.copy()
    
    if len(df_res) < 150: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['momentum'] = calculate_momentum(df_res)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

# =============================================================================
# 4. GENETİK ALGORİTMA (CANAVAR MOTORU)
# =============================================================================

def run_heavy_ga(df, n_gen, pop_size):
    """
    Bu fonksiyon, modellerin (RF, XGB, LGBM) en iyi hiperparametrelerini
    EVRİMSEL SÜREÇ ile bulur.
    """
    # Veriyi Eğitim/Test olarak ayır (Backtest için)
    split = int(len(df) * 0.8)
    train = df.iloc[:split]
    test = df.iloc[split:]
    
    X_cols = ['log_ret', 'range', 'heuristic', 'momentum']
    y_train = train['target']
    y_test = test['target']

    # DEAP Kurulumu
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,), overwrite=True)
        creator.create('Individual', list, fitness=creator.FitnessMax, overwrite=True)
        
    toolbox = base.Toolbox()
    
    # GENLER: [RF_Depth, RF_Est, XGB_Depth, XGB_LR, LGBM_LR]
    toolbox.register('rf_depth', random.randint, 3, 15)
    toolbox.register('rf_est', random.randint, 20, 100)
    toolbox.register('xgb_depth', random.randint, 3, 10)
    toolbox.register('xgb_lr', random.uniform, 0.01, 0.3)
    toolbox.register('lgbm_lr', random.uniform, 0.01, 0.3)
    
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.rf_depth, toolbox.rf_est, toolbox.xgb_depth, toolbox.xgb_lr, toolbox.lgbm_lr), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    # Fitness Fonksiyonu (Simülasyon Başarısı)
    def evaluate(ind):
        rf_d, rf_n, xgb_d, xgb_l, lgb_l = ind
        
        # Modelleri bu ayarlarla eğit
        model1 = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d, n_jobs=-1).fit(train[X_cols], y_train)
        model2 = xgb.XGBClassifier(n_estimators=50, max_depth=xgb_d, learning_rate=xgb_l, n_jobs=-1, verbosity=0).fit(train[X_cols], y_train)
        
        # Tahmin (Doğruluk oranı bizim puanımızdır)
        p1 = model1.predict(test[X_cols])
        p2 = model2.predict(test[X_cols])
        
        # Ensemble (Basit)
        final_p = np.round((p1 + p2) / 2)
        acc = accuracy_score(y_test, final_p)
        return (acc,)

    toolbox.register('evaluate', evaluate)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)
    
    # Evrimi Başlat
    pop = toolbox.population(n=pop_size)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, verbose=False)
    
    best_ind = tools.selBest(pop, 1)[0]
    
    # En iyi parametreleri döndür
    return {
        'rf_depth': int(max(3, best_ind[0])), 
        'rf_est': int(max(10, best_ind[1])),
        'xgb_depth': int(max(3, best_ind[2])),
        'xgb_lr': max(0.01, best_ind[3]),
        'lgbm_lr': max(0.01, best_ind[4])
    }

# =============================================================================
# 5. EĞİTİM, JÜRİ VE SİMÜLASYON (BEYİN)
# =============================================================================

def train_and_simulate_full(df, best_params):
    """
    Optimize edilmiş parametrelerle modelleri eğitir,
    Lojistik Regresyon (Jüri) ile ağırlıkları öğrenir,
    Geçmişi simüle eder ve bugünün kararını verir.
    """
    # Son 60 gün Test/Validation
    test_size = 60
    if len(df) < test_size + 50: return 0.0, None
    
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    X_cols = ['log_ret', 'range', 'heuristic', 'momentum']
    y_train = train['target']
    
    # 1. OPTİMİZE MODELLERİ EĞİT
    rf = RandomForestClassifier(n_estimators=best_params['rf_est'], max_depth=best_params['rf_depth'], random_state=42, n_jobs=-1).fit(train[X_cols], y_train)
    xgb_c = xgb.XGBClassifier(n_estimators=50, max_depth=best_params['xgb_depth'], learning_rate=best_params['xgb_lr'], n_jobs=-1, verbosity=0).fit(train[X_cols], y_train)
    lgb_c = lgb.LGBMClassifier(n_estimators=50, learning_rate=best_params['lgbm_lr'], verbose=-1).fit(train[X_cols], y_train)
    
    # HMM (Rejim Tespiti)
    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(train[['log_ret', 'range']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None
    
    # 2. JÜRİ HAZIRLIĞI (STACKING)
    hmm_pred = np.zeros(len(train))
    if hmm:
        probs = hmm.predict_proba(X_hmm)
        bull = np.argmax(hmm.means_[:, 0]); bear = np.argmin(hmm.means_[:, 0])
        hmm_pred = probs[:, bull] - probs[:, bear]
    
    # Jüriye Giren Veriler (Geçmiş Performanslar)
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(train[X_cols])[:, 1],
        'XGB': xgb_c.predict_proba(train[X_cols])[:, 1],
        'LGBM': lgb_c.predict_proba(train[X_cols])[:, 1],
        'HMM': hmm_pred,
        'Heuristic': train['heuristic'].values,
        'Momentum': train['momentum'].values
    })
    
    # JÜRİ (LOGISTIC REGRESSION) AĞIRLIKLARI ÖĞRENİYOR
    meta_model = LogisticRegression().fit(meta_X, y_train)
    weights = meta_model.coef_[0] # [w_rf, w_xgb, w_lgbm, w_hmm, w_heur, w_mom]
    
    # 3. SİMÜLASYON (Son 60 gün ne olurdu?)
    sim_eq = [100]; hodl_eq = [100]; cash=100; coin=0; p0 = test['close'].iloc[0]
    
    # Test Seti için Girdiler
    X_hmm_t = scaler.transform(test[['log_ret', 'range']])
    hmm_s_t = np.zeros(len(test))
    if hmm:
        pt = hmm.predict_proba(X_hmm_t)
        hmm_s_t = pt[:, np.argmax(hmm.means_[:, 0])] - pt[:, np.argmin(hmm.means_[:, 0])]
        
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(test[X_cols])[:, 1],
        'XGB': xgb_c.predict_proba(test[X_cols])[:, 1],
        'LGBM': lgb_c.predict_proba(test[X_cols])[:, 1],
        'HMM': hmm_s_t,
        'Heuristic': test['heuristic'].values,
        'Momentum': test['momentum'].values
    })
    
    # Jüri Kararları
    f_probs = meta_model.predict_proba(mx_test)[:, 1]
    
    for i in range(len(test)):
        p = test['close'].iloc[i]
        s = (f_probs[i] - 0.5) * 2
        # Eşikler
        if s > 0.25 and cash > 0: coin = cash / p; cash = 0
        elif s < -0.25 and coin > 0: cash = coin * p; coin = 0
        
        sim_eq.append(cash + coin * p)
        hodl_eq.append((100 / p0) * p)
        
    # 4. BUGÜNÜN KARARI
    final_signal = (f_probs[-1] - 0.5) * 2
    
    info = {
        "weights": weights,
        "bot_eq": sim_eq[1:], "hodl_eq": hodl_eq[1:], "dates": test.index,
        "alpha": (sim_eq[-1] - hodl_eq[-1]),
        "bot_roi": (sim_eq[-1] - 100),
        "hodl_roi": (hodl_eq[-1] - 100),
        "conf": f_probs[-1],
        "scores": {
            "Heuristic": test['heuristic'].iloc[-1],
            "Momentum": test['momentum'].iloc[-1]
        }
    }
    return final_signal, info

# =============================================================================
# 6. TURNUVA (ZAMAN DİLİMİ SEÇİMİ)
# =============================================================================

def analyze_ticker_beast(ticker, status_ph, n_gen, pop_size):
    raw_df = get_raw_data(ticker)
    if raw_df is None: 
        status_ph.error("Veri Yok")
        return "HATA", 0.0, "YOK", None
    
    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
    
    best_alpha = -9999; final_decision = "BEKLE"; winning_tf = "YOK"; best_info = None
    
    for tf_name, tf_code in timeframes.items():
        status_ph.text(f"🧬 {tf_name} için Genetik Evrim Başlatılıyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # 1. GENETİK OPTİMİZASYON (EN İYİ PARAMETRELERİ BUL)
        best_params = run_heavy_ga(df, n_gen, pop_size)
        
        # 2. EĞİTİM VE SİMÜLASYON
        sig, info = train_and_simulate_full(df, best_params)
        
        if info and info['alpha'] > best_alpha:
            best_alpha = info['alpha']
            winning_tf = tf_name
            best_info = info
            if sig > 0.25: final_decision = "AL"
            elif sig < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"
            
    return final_decision, current_price, winning_tf, best_info

# =============================================================================
# 7. ANA ARAYÜZ
# =============================================================================

if st.button("🦍 CANAVAR MODU BAŞLAT (ANALİZ ET)", type="primary"):
    pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Portföy hatası.")
    else:
        updated = pf_df.copy()
        prog = st.progress(0)
        sim_summary = []
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            if len(str(ticker)) < 3: continue
            
            with st.expander(f"🧬 {ticker} Analiz & Evrim Raporu", expanded=True):
                ph = st.empty()
                # CANAVAR FONKSİYONU ÇAĞIR
                dec, prc, tf, info = analyze_ticker_beast(ticker, ph, ga_gens, 20) # Pop size 20
                
                if dec != "HATA" and info:
                    sim_summary.append({
                        "Coin": ticker, "TF": tf, "Alpha": info['alpha'],
                        "Bot": info['bot_roi'], "HODL": info['hodl_roi']
                    })
                    
                    # Ağırlıklar
                    w = info['weights']
                    w_df = pd.DataFrame({
                        'Model': ['RF', 'XGB', 'LGBM', 'HMM', 'Heuristic', 'Momentum'],
                        'Etki': w
                    })
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"### Karar: **{dec}**")
                        st.caption(f"Seçilen Zaman: {tf}")
                        st.markdown(f"**Senin Puanın:** {info['scores']['Heuristic']:.2f}")
                        st.dataframe(w_df, hide_index=True)
                    
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['bot_eq'], name="Bot", line=dict(color='green')))
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['hodl_eq'], name="HODL", line=dict(color='gray', dash='dot')))
                        col_ti = "green" if info['alpha'] > 0 else "red"
                        fig.update_layout(title=f"Alpha: {info['alpha']:+.1f}%", title_font_color=col_ti, height=250, template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # İşlem Kaydı
                    stt = row['Durum']
                    log_msg = row['Son_Islem_Log']
                    ts = datetime.now(pytz.timezone('Europe/Istanbul')).strftime("%d-%m %H:%M")
                    
                    if stt == 'COIN' and dec == 'SAT':
                        amt = float(row['Miktar']); cash_val = amt * prc
                        updated.at[idx, 'Durum'] = 'CASH'; updated.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                        updated.at[idx, 'Miktar'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                        log_msg = f"SAT ({tf}) A:{info['alpha']:.1f}"; updated.at[idx, 'Son_Islem_Zamani'] = ts
                    elif stt == 'CASH' and dec == 'AL':
                        cash = float(row['Nakit_Bakiye_USD']); amt = cash / prc
                        if cash > 1:
                            updated.at[idx, 'Durum'] = 'COIN'; updated.at[idx, 'Miktar'] = amt
                            updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                            log_msg = f"AL ({tf}) A:{info['alpha']:.1f}"; updated.at[idx, 'Son_Islem_Zamani'] = ts
                    
                    val = (float(updated.at[idx, 'Miktar']) * prc) if updated.at[idx, 'Durum'] == 'COIN' else float(updated.at[idx, 'Nakit_Bakiye_USD'])
                    updated.at[idx, 'Kaydedilen_Deger_USD'] = val
                    updated.at[idx, 'Son_Islem_Log'] = log_msg
                    ph.success(f"Tamamlandı. Alpha: {info['alpha']:+.1f}%")

            prog.progress((i+1)/len(updated))
        
        save_portfolio(updated, sheet)
        
        st.divider()
        if sim_summary:
            sdf = pd.DataFrame(sim_summary)
            c1, c2, c3 = st.columns(3)
            c1.metric("Genel Alpha", f"{sdf['Alpha'].mean():.2f}%")
            c2.metric("Bot ROI", f"{sdf['Bot'].mean():.2f}%")
            c3.metric("HODL ROI", f"{sdf['HODL'].mean():.2f}%")
            st.dataframe(sdf)
            
        st.success("✅ Canavar Analizi Tamamlandı!")

st.divider()
try:
    df_v, _ = load_and_fix_portfolio()
    if not df_v.empty:
        st.subheader("📋 Mevcut Portföy")
        st.dataframe(df_v)
        tot = df_v['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Varlık", f"${tot:.2f}")
except: pass
