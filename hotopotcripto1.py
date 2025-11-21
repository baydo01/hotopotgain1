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
import lightgbm as lgb # YENİ MODEL: LightGBM (GPBoost Alternatifi)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Multi-Model", layout="wide")
st.title("🏦 Hedge Fund AI: 6'lı Jüri & Otonom Karar")

# =============================================================================
# 1. AYARLAR
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

with st.sidebar:
    st.header("⚙️ Ayarlar")
    st.info("Yeni Eklenen Modeller:\n1. LightGBM (Hızlı Boosting)\n2. Momentum (14 Günlük Saf Trend)")

# =============================================================================
# 2. GOOGLE SHEETS
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
# 3. YENİ AI MOTORU (6 MODELLİ)
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

# --- YENİ ÖZELLİK: MOMENTUM SKORU ---
def calculate_momentum_score(df, window=14):
    """
    Saf Trend Takipçisi. Son 2 haftanın (14 bar) eğimine bakar.
    Bu model, senin istediğin 'son bir iki haftalık trend' modelidir.
    """
    # ROC (Rate of Change)
    roc = df['close'].pct_change(window).fillna(0)
    # Pozitifse +1, Negatifse -1 (Basitleştirilmiş)
    return np.sign(roc)

def calculate_heuristic_score(df):
    if len(df) < 150: return pd.Series(0.0, index=df.index)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    s3 = np.where(df['close'] > df['close'].rolling(150).mean(), 1, -1)
    vol = df['close'].pct_change().rolling(20).std()
    s4 = np.where(vol < vol.shift(1), 1, -1)
    s5 = np.sign(df['close'].diff(10).fillna(0))
    return (s1 + s2 + s3 + s4 + s5) / 5.0

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
    
    # YENİ: Momentum Skoru (Son 14 bar)
    df_res['momentum'] = calculate_momentum_score(df_res, window=14)
    
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_meta_learner(df):
    test_size = 60
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    X_cols = ['log_ret', 'range', 'heuristic', 'momentum']
    y_train = train['target']
    
    # 1. RANDOM FOREST
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42).fit(train[X_cols], y_train)
    
    # 2. XGBOOST
    xgb_c = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3).fit(train[X_cols], y_train)
    
    # 3. LIGHTGBM (GPBoost Alternatifi - Çok Hızlı)
    lgb_c = lgb.LGBMClassifier(n_estimators=50, max_depth=3, verbose=-1).fit(train[X_cols], y_train)
    
    # 4. HMM
    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(train[['log_ret', 'range']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None

    # --- META-MODEL İÇİN GİRDİ HAZIRLA ---
    hmm_pred = np.zeros(len(train))
    if hmm:
        probs = hmm.predict_proba(X_hmm)
        bull = np.argmax(hmm.means_[:, 0]); bear = np.argmin(hmm.means_[:, 0])
        hmm_pred = probs[:, bull] - probs[:, bear]
    
    # JÜRİ MASASI: [RF, XGB, LGBM, HMM, HEURISTIC, MOMENTUM]
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(train[X_cols])[:, 1],
        'XGB': xgb_c.predict_proba(train[X_cols])[:, 1],
        'LGBM': lgb_c.predict_proba(train[X_cols])[:, 1],
        'HMM': hmm_pred,
        'Heuristic': train['heuristic'].values,
        'Momentum': train['momentum'].values
    })
    
    # Ağırlıkları Öğren (Logistic Regression)
    meta_model = LogisticRegression().fit(meta_X, y_train)
    weights = meta_model.coef_[0] # [w_rf, w_xgb, w_lgbm, w_hmm, w_heur, w_mom]
    
    # --- SİMÜLASYON ---
    sim_eq = [100]; hodl_eq = [100]; cash=100; coin=0; p0 = test['close'].iloc[0]
    
    # Test tahminleri
    X_hmm_t = scaler.transform(test[['log_ret', 'range']])
    hmm_p_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_s_t = hmm_p_t[:, np.argmax(hmm.means_[:,0])] - hmm_p_t[:, np.argmin(hmm.means_[:,0])] if hmm else np.zeros(len(test))
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(test[X_cols])[:,1],
        'XGB': xgb_c.predict_proba(test[X_cols])[:,1],
        'LGBM': lgb_c.predict_proba(test[X_cols])[:,1],
        'HMM': hmm_s_t,
        'Heuristic': test['heuristic'].values,
        'Momentum': test['momentum'].values
    })
    
    f_probs = meta_model.predict_proba(mx_test)[:,1]
    
    for i in range(len(test)):
        pr = test['close'].iloc[i]
        s = (f_probs[i]-0.5)*2
        if s>0.25 and cash>0: coin=cash/pr; cash=0
        elif s<-0.25 and coin>0: cash=coin*pr; coin=0
        sim_eq.append(cash+coin*pr)
        hodl_eq.append((100/p0)*pr)
        
    info = {
        "weights": weights,
        "bot_eq": sim_eq[1:], "hodl_eq": hodl_eq[1:], "dates": test.index,
        "alpha": (sim_eq[-1]-hodl_eq[-1]),
        "bot_roi": (sim_eq[-1]-100),
        "hodl_roi": (hodl_eq[-1]-100),
        "conf": f_probs[-1],
        "scores": {
            "Heuristic": test['heuristic'].iloc[-1],
            "Momentum": test['momentum'].iloc[-1]
        }
    }
    return (f_probs[-1]-0.5)*2, info

def analyze_ticker_tournament(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return "HATA", 0.0, "YOK", None
    
    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
    
    best_alpha = -9999; final_decision = "BEKLE"; winning_tf = "YOK"; best_info = None
    
    for tf_name, tf_code in timeframes.items():
        status_placeholder.text(f"{tf_name} yarışıyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        sig, info = train_meta_learner(df)
        
        # Alpha'ya göre seçim (En çok kazandıranı seç)
        if info['alpha'] > best_alpha:
            best_alpha = info['alpha']
            winning_tf = tf_name
            best_info = info
            if sig > 0.25: final_decision = "AL"
            elif sig < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"
            
    return final_decision, current_price, winning_tf, best_info

# =============================================================================
# 4. ARAYÜZ
# =============================================================================

if st.button("🚀 ANALİZİ BAŞLAT", type="primary"):
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Hata: Portföy yüklenemedi.")
    else:
        updated = pf_df.copy()
        prog = st.progress(0)
        sim_summary = []
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            if len(str(ticker)) < 3: continue
            
            with st.expander(f"🧠 {ticker} Detaylı Rapor", expanded=True):
                ph = st.empty()
                dec, prc, tf, info = analyze_ticker_tournament(ticker, ph)
                
                if dec != "HATA" and info:
                    sim_summary.append({
                        "Coin": ticker, "TF": tf, "Alpha": info['alpha'],
                        "Bot": info['bot_roi'], "HODL": info['hodl_roi']
                    })
                    
                    # Ağırlık Görselleştirme
                    w = info['weights']
                    # Negatif ağırlıkları da görelim (Ters indikatör olabilirler)
                    w_df = pd.DataFrame({
                        'Model': ['RandomForest', 'XGBoost', 'LightGBM', 'HMM', 'Heuristic', 'Momentum (14d)'],
                        'Etki Katsayısı': w
                    })
                    
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"### Karar: **{dec}**")
                        st.caption(f"Zaman: {tf}")
                        st.dataframe(w_df, hide_index=True)
                    
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['bot_eq'], name="Bot", line=dict(color='green')))
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['hodl_eq'], name="HODL", line=dict(color='gray', dash='dot')))
                        color_ti = "green" if info['alpha'] > 0 else "red"
                        fig.update_layout(title=f"Alpha: {info['alpha']:+.1f}%", title_font_color=color_ti, height=250, template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0))
                        st.plotly_chart(fig, use_container_width=True)
                        
                    # İşlem
                    stt = row['Durum']
                    log_msg = row['Son_Islem_Log']
                    if stt == 'COIN' and dec == 'SAT':
                        amt = float(row['Miktar']); cash_val = amt * prc
                        updated.at[idx, 'Durum'] = 'CASH'; updated.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                        updated.at[idx, 'Miktar'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                        log_msg = f"SAT ({tf}) A:{info['alpha']:.1f}"; updated.at[idx, 'Son_Islem_Zamani'] = time_str
                    elif stt == 'CASH' and dec == 'AL':
                        cash = float(row['Nakit_Bakiye_USD']); amt = cash / prc
                        if cash > 1:
                            updated.at[idx, 'Durum'] = 'COIN'; updated.at[idx, 'Miktar'] = amt
                            updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                            log_msg = f"AL ({tf}) A:{info['alpha']:.1f}"; updated.at[idx, 'Son_Islem_Zamani'] = time_str
                    
                    val = (float(updated.at[idx, 'Miktar']) * prc) if updated.at[idx, 'Durum'] == 'COIN' else float(updated.at[idx, 'Nakit_Bakiye_USD'])
                    updated.at[idx, 'Kaydedilen_Deger_USD'] = val
                    updated.at[idx, 'Son_Islem_Log'] = log_msg
                    ph.success(f"Tamamlandı. Puanlar: Heuristic={info['scores']['Heuristic']:.2f}, Momentum={info['scores']['Momentum']:.0f}")

            prog.progress((i+1)/len(updated))
        
        save_portfolio(updated, sheet)
        
        # ÖZET TABLO
        st.divider()
        if sim_summary:
            sdf = pd.DataFrame(sim_summary)
            col1, col2, col3 = st.columns(3)
            col1.metric("Genel Alpha", f"{sdf['Alpha'].mean():.2f}%")
            col2.metric("Bot ROI", f"{sdf['Bot'].mean():.2f}%")
            col3.metric("HODL ROI", f"{sdf['HODL'].mean():.2f}%")
            st.dataframe(sdf)
            
        st.success("✅ Analiz Bitti!")

st.divider()
try:
    df_v, _ = load_and_fix_portfolio()
    if not df_v.empty:
        st.subheader("📋 Mevcut Portföy")
        st.dataframe(df_v)
        tot = df_v['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Varlık", f"${tot:.2f}")
except: pass
