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

# --- AI & ML KÜTÜPHANELERİ ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Ultimate Dashboard", layout="wide")
st.title("🏦 Hedge Fund AI: Şeffaf Yönetim & Simülasyon")

# =============================================================================
# 1. AYARLAR
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"

with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    st.info("Bu panel, yapay zekanın 'Ağırlık Kararlarını' ve 'Validation Testi' sonuçlarını canlı gösterir.")

# =============================================================================
# 2. GOOGLE SHEETS & OTO-KURULUM
# =============================================================================

def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
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
            defaults = [
                ["BTC-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["ETH-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["SOL-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["BNB-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["XRP-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"],
                ["DOGE-USD", "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"]
            ]
            for d in defaults: sheet.append_row(d)
            time.sleep(2)
    except: pass
        
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    df = df[df['Ticker'].astype(str).str.contains('-USD', na=False)]
    
    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    return df, sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy(); df_export = df_export.astype(str)
        sheet.clear(); sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except: pass

# =============================================================================
# 3. AI ENGINE: KALMAN, HEURISTIC, AUTO-WEIGHTING
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
    if len(df) < 150: return pd.Series(0.0, index=df.index)
    # 5 Adımlı Mantık (Yüzdesel)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    s3 = np.where(df['close'].pct_change(150) < -0.3, 1, 0) 
    vol = df['close'].pct_change().rolling(20).std()
    s4 = np.where(vol < vol.shift(1), 1, -1)
    s5 = np.sign(df['close'].diff(10).fillna(0))
    total = (s1 + s2 + s3 + s4 + s5) / 5.0
    return total

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
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_validate_and_predict(df):
    """
    Train/Test ayrımı yapar, Modelleri Eğitir, Ağırlıkları Öğrenir ve
    Geriye dönük SİMÜLASYON verilerini hazırlar.
    """
    # Son 60 gün VALIDATION/TEST seti
    test_size = 60
    if len(df) < test_size + 100: return 0.0, {}, None
    
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    # 1. BASE EĞİTİM
    X_train = train[['log_ret', 'range', 'heuristic']]
    y_train = train['target']
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1).fit(X_train, y_train)
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3, n_jobs=-1).fit(X_train, y_train)
    
    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(train[['log_ret', 'range']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None

    # 2. META-MODEL EĞİTİMİ (Logistic Reg. ile Ağırlık Öğrenme)
    # HMM Sinyali Üret
    hmm_pred = np.zeros(len(train))
    if hmm:
        probs = hmm.predict_proba(X_hmm)
        bull_idx = np.argmax(hmm.means_[:, 0])
        bear_idx = np.argmin(hmm.means_[:, 0])
        hmm_pred = probs[:, bull_idx] - probs[:, bear_idx]
    
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_train)[:, 1],
        'XGB': xgb_clf.predict_proba(X_train)[:, 1],
        'HMM': hmm_pred,
        'Heuristic': train['heuristic'].values
    })
    meta_model = LogisticRegression().fit(meta_X, y_train)
    
    # Öğrenilen Ağırlıklar
    weights = meta_model.coef_[0] 
    # Ağırlıkları normalize et (grafik için)
    abs_w = np.abs(weights)
    norm_w = abs_w / (np.sum(abs_w) + 1e-9) * 100
    
    # 3. VALIDATION SİMÜLASYONU (Geçmiş 60 gün ne yapardı?)
    # Test setindeki her gün için tahmin yap ve sanal cüzdan yönet
    
    sim_equity = [100] # 100$ ile başla
    hodl_equity = [100]
    cash = 100
    coin = 0
    start_price = test['close'].iloc[0]
    
    # HMM Test Sinyalleri
    X_hmm_test = scaler.transform(test[['log_ret', 'range']])
    hmm_test_probs = hmm.predict_proba(X_hmm_test) if hmm else np.zeros((len(test), 3))
    hmm_test_sig = hmm_test_probs[:, bull_idx] - hmm_test_probs[:, bear_idx] if hmm else np.zeros(len(test))
    
    # Diğer Test Sinyalleri
    rf_test = rf.predict_proba(test[['log_ret', 'range', 'heuristic']])[:, 1]
    xgb_test = xgb_clf.predict_proba(test[['log_ret', 'range', 'heuristic']])[:, 1]
    
    # Meta Input Test
    meta_X_test = pd.DataFrame({
        'RF': rf_test, 'XGB': xgb_test, 'HMM': hmm_test_sig, 'Heuristic': test['heuristic'].values
    })
    
    # Gün Gün Simülasyon
    final_probs = meta_model.predict_proba(meta_X_test)[:, 1]
    
    for i in range(len(test)):
        price = test['close'].iloc[i]
        sig = (final_probs[i] - 0.5) * 2
        
        # İşlem (Sanal)
        if sig > 0.20 and cash > 0: # AL
            coin = cash / price; cash = 0
        elif sig < -0.20 and coin > 0: # SAT
            cash = coin * price; coin = 0
            
        current_val = cash + (coin * price)
        sim_equity.append(current_val)
        
        # HODL
        hodl_val = (100 / start_price) * price
        hodl_equity.append(hodl_val)
        
    simulation_data = {
        "dates": test.index,
        "bot_eq": sim_equity[1:], # İlk gün hariç (uzunluk eşitleme)
        "hodl_eq": hodl_equity[1:],
        "bot_roi": (sim_equity[-1] - 100),
        "hodl_roi": (hodl_equity[-1] - 100)
    }

    # 4. ŞİMDİKİ KARAR (Son bar)
    last_row = df.iloc[[-1]]
    # ... (Burada son bar için özellik üretimi) ...
    # Basitçe son test verisini kullanabiliriz çünkü son bar orada
    final_signal = (final_probs[-1] - 0.5) * 2
    
    model_info = {
        "weights": norm_w, # [RF, XGB, HMM, Heuristic]
        "simulation": simulation_data,
        "ai_confidence": final_probs[-1]
    }
    
    return final_signal, model_info

def analyze_ticker(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None: 
        status_placeholder.error("Veri Yok")
        return "HATA", 0.0, "YOK", {}
    
    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W'}
    
    best_sig = -99; decision = "BEKLE"; win_tf = "YOK"; best_info = {}
    
    for tf_name, tf_code in timeframes.items():
        status_placeholder.text(f"{tf_name} simülasyonu yapılıyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        sig, info = train_validate_and_predict(df)
        
        if abs(sig) > best_sig:
            best_sig = abs(sig)
            win_tf = tf_name
            best_info = info
            if sig > 0.20: decision = "AL"
            elif sig < -0.20: decision = "SAT"
            else: decision = "BEKLE"
            
    return decision, current_price, win_tf, best_info

# =============================================================================
# 4. ARAYÜZ VE ÇALIŞTIRMA
# =============================================================================

if st.button("🚀 PORTFÖYÜ CANLI ANALİZ ET", type="primary"):
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    with st.spinner("Veritabanı (Sheets) açılıyor..."):
        pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Portföy yüklenemedi.")
    else:
        updated_pf = pf_df.copy()
        prog = st.progress(0)
        
        for i, (idx, row) in enumerate(updated_pf.iterrows()):
            ticker = row['Ticker']
            if len(str(ticker)) < 3: continue
            
            with st.expander(f"🔍 {ticker} Analiz Raporu", expanded=False):
                ph = st.empty()
                dec, prc, tf, info = analyze_ticker(ticker, ph)
                
                if dec != "HATA" and info:
                    # --- 1. DETAYLI MODEL RAPORU (İSTEDİĞİN YER) ---
                    c1, c2 = st.columns([1, 2])
                    
                    with c1:
                        st.markdown(f"### 🧠 Model Kararı: {dec}")
                        st.markdown(f"**Zaman Dilimi:** {tf}")
                        st.markdown(f"**Fiyat:** ${prc:.2f}")
                        st.markdown(f"**AI Güveni:** %{info['ai_confidence']*100:.1f}")
                        
                        # Ağırlık Tablosu
                        w = info['weights']
                        w_df = pd.DataFrame({
                            'Model': ['RandomForest', 'XGBoost', 'HMM', 'Senin Kuralın'],
                            'Etki (%)': [w[0], w[1], w[2], w[3]]
                        })
                        st.dataframe(w_df, hide_index=True)

                    with c2:
                        # SİMÜLASYON GRAFİĞİ
                        sim = info['simulation']
                        fig = make_subplots(specs=[[{"secondary_y": False}]])
                        fig.add_trace(go.Scatter(x=sim['dates'], y=sim['bot_eq'], name="Bot (Simüle)", line=dict(color='green', width=2)))
                        fig.add_trace(go.Scatter(x=sim['dates'], y=sim['hodl_eq'], name="HODL", line=dict(color='gray', dash='dot')))
                        
                        alpha = sim['bot_roi'] - sim['hodl_roi']
                        title_color = "green" if alpha > 0 else "red"
                        fig.update_layout(
                            title=f"Son 60 Gün Validasyon Testi | Alpha: %{alpha:.1f}",
                            title_font_color=title_color,
                            height=300, margin=dict(l=0, r=0, t=30, b=0),
                            template="plotly_dark", showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # --- İŞLEM KAYDI ---
                    status = row['Durum']
                    if status == 'COIN' and dec == 'SAT':
                        amt = float(row['Miktar'])
                        if amt > 0:
                            updated_pf.at[idx, 'Durum'] = 'CASH'; updated_pf.at[idx, 'Nakit_Bakiye_USD'] = amt * prc
                            updated_pf.at[idx, 'Miktar'] = 0.0; updated_pf.at[idx, 'Son_Islem_Fiyati'] = prc
                            updated_pf.at[idx, 'Son_Islem_Log'] = f"SAT ({tf})"; updated_pf.at[idx, 'Son_Islem_Zamani'] = time_str
                    elif status == 'CASH' and dec == 'AL':
                        cash = float(row['Nakit_Bakiye_USD'])
                        if cash > 1:
                            updated_pf.at[idx, 'Durum'] = 'COIN'; updated_pf.at[idx, 'Miktar'] = cash / prc
                            updated_pf.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated_pf.at[idx, 'Son_Islem_Fiyati'] = prc
                            updated_pf.at[idx, 'Son_Islem_Log'] = f"AL ({tf})"; updated_pf.at[idx, 'Son_Islem_Zamani'] = time_str
                    
                    val = (float(updated_pf.at[idx, 'Miktar']) * prc) if updated_pf.at[idx, 'Durum'] == 'COIN' else float(updated_pf.at[idx, 'Nakit_Bakiye_USD'])
                    updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = val
                    
                    ph.success(f"Analiz Tamamlandı. Karar: {dec}")

            prog.progress((i+1)/len(updated_pf))
        
        save_portfolio(updated_pf, sheet)
        st.success("✅ TÜM İŞLEMLER TAMAMLANDI!")

# LİSTELEME
st.divider()
try:
    df_v, _ = load_and_fix_portfolio()
    if not df_v.empty:
        st.subheader("📋 Mevcut Portföy Durumu")
        st.dataframe(df_v)
        
        total = df_v['Kaydedilen_Deger_USD'].sum()
        start_cap = df_v['Baslangic_USD'].sum()
        pnl = total - start_cap
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam Varlık", f"${total:.2f}")
        c2.metric("Net Kâr/Zarar", f"${pnl:.2f}", f"%{(pnl/start_cap)*100:.2f}" if start_cap>0 else "0%")
        c3.metric("Bot Durumu", "Aktif", "Otonom")
except: pass
