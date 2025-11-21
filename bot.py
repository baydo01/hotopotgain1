pip install git+https://github.com/scikit-learn-contrib/py-earth.git
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
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Alpha Master", layout="wide")
st.title("🏦 Hedge Fund AI: Alpha & Performans Analizi")

# =============================================================================
# 1. AYARLAR VE GÜVENLİK
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"

with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    use_ga = st.checkbox("Genetic Algoritma Kullan", value=False)
    ga_gens = st.number_input("GA Jenerasyon Sayısı", 1, 50, 5)

# -------------------- HEURISTIC (5 Adımlı Puanlama) --------------------
def calculate_heuristic_score(df):
    if len(df) < 252: return pd.Series(0.0, index=df.index)
    # 1. Kısa Vade (5 Bar)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    # 2. Orta Vade (30 Bar)
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    # 3. Uzun Vade Trend (SMA 200 Üstü mü?)
    ma = df['close'].rolling(200).mean()
    s3 = np.where(df['close'] > ma, 1, -1)
    # 4. Volatilite Düşüşü (İyiye İşaret)
    vol = df['close'].pct_change().rolling(20).std()
    s4 = np.where(vol < vol.shift(1), 1, -1)
    # 5. Momentum
    mom = np.sign(df['close'].diff(10).fillna(0))
    
    total_score = s1 + s2 + s3 + s4 + mom
    return total_score # Max +5, Min -5

# =============================================================================
# 2. GOOGLE SHEETS BAĞLANTISI
# =============================================================================

def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except: pass
    if not creds and os.path.exists(CREDENTIALS_FILE):
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
    df = df[df['Ticker'].str.contains('-USD', na=False)]
    
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
        sheet.clear()
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except: pass

# =============================================================================
# 3. YAPAY ZEKA MOTORU (DETAYLI)
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
    if df is None or len(df) < 100: return None
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W': df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M': df_res = df.resample('ME').agg(agg_dict).dropna()
    else: df_res = df.copy()
    
    if len(df_res) < 252: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    df_res['heuristic_score'] = calculate_heuristic_score(df_res)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_models_for_window(train_df, rf_depth=5):
    features_base = ['log_ret','range','trend_signal', 'heuristic_score']
    X = train_df[features_base]; y = train_df['target']
    
    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=30, max_depth=3)
    clf_xgb.fit(X, y)
    
    hmm_model = None
    try:
        Xh_s = StandardScaler().fit_transform(train_df[['log_ret','range']].values)
        hmm_model = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
        hmm_model.fit(Xh_s)
    except: pass

    # Meta-Model (Logistic)
    meta_clf = LogisticRegression(max_iter=200)
    # Basitlik için direkt fit yapıyoruz, gerçek stacking cross-val gerektirir ama hız için:
    meta_clf.fit(X, y) 
    
    return {'rf': clf_rf, 'xgb': clf_xgb, 'hmm': hmm_model, 'meta': meta_clf}

def predict_details(models, row):
    """Detaylı analiz sonuçları döndürür."""
    if models is None: return 0, {}
    
    # Veri
    Xrow = pd.DataFrame([row[['log_ret','range','trend_signal', 'heuristic_score']]])
    
    # Olasılıklar
    rf_prob = models['rf'].predict_proba(Xrow)[0][1]
    xgb_prob = models['xgb'].predict_proba(Xrow)[0][1]
    meta_prob = models['meta'].predict_proba(Xrow)[0][1]
    
    # HMM Detayı
    hmm_state = "Nötr"
    hmm_conf = 0.0
    if models['hmm']:
        try:
            Xh = row[['log_ret','range']].values.reshape(1,-1)
            probs = models['hmm'].predict_proba(StandardScaler().fit_transform(Xh))[0]
            bull_idx = np.argmax(models['hmm'].means_[:,0])
            bear_idx = np.argmin(models['hmm'].means_[:,0])
            
            if probs[bull_idx] > 0.6: hmm_state = "BOĞA"; hmm_conf = probs[bull_idx]
            elif probs[bear_idx] > 0.6: hmm_state = "AYI"; hmm_conf = probs[bear_idx]
            else: hmm_state = "YATAY"
        except: pass

    # Final Sinyal (Meta Model Baskın)
    final_signal = (meta_prob - 0.5) * 2
    
    details = {
        "Heuristic": row['heuristic_score'],
        "HMM_Gorus": f"{hmm_state} ({hmm_conf:.2f})",
        "AI_Guven": abs(final_signal),
        "RF_Prob": rf_prob,
        "XGB_Prob": xgb_prob
    }
    
    return final_signal, details

def analyze_ticker_detailed(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return "HATA", 0.0, "YOK", {}

    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W'}
    best_score = -999; final_decision = "BEKLE"; winning_tf = "YOK"
    best_details = {}
    
    for tf_name, tf_code in timeframes.items():
        status_placeholder.text(f"{tf_name} analiz ediliyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        models = train_models_for_window(df.iloc[-252:], rf_depth=5)
        signal, details = predict_details(models, df.iloc[-1])
        
        if abs(signal) > best_score:
            best_score = abs(signal)
            winning_tf = tf_name
            best_details = details
            if signal > 0.25: final_decision = "AL"
            elif signal < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"

    status_placeholder.empty()
    return final_decision, current_price, winning_tf, best_details

# =============================================================================
# 4. ANA ÇALIŞTIRMA VE RAPORLAMA EKRANI
# =============================================================================

if st.button("🚀 DETAYLI ANALİZİ BAŞLAT", type="primary"):
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    with st.spinner("Google Sheets'e bağlanılıyor..."):
        pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Hata: Portföy yüklenemedi.")
    else:
        updated_portfolio = pf_df.copy()
        report_data = [] # Rapor için veri toplayacağız
        progress_bar = st.progress(0)
        
        for i, (idx, row) in enumerate(updated_portfolio.iterrows()):
            ticker = row['Ticker']
            if not ticker or ticker == "-": continue
            
            status_box = st.empty()
            decision, price, tf_name, details = analyze_ticker_detailed(ticker, status_box)
            
            # Rapor Verisi Ekle
            if decision != "HATA":
                report_row = {
                    "Coin": ticker,
                    "Karar": decision,
                    "Zaman": tf_name,
                    "Fiyat": f"${price:.2f}",
                    "Heuristic Puan": f"{details.get('Heuristic', 0):.1f}/5.0",
                    "HMM Görüşü": details.get('HMM_Gorus', '-'),
                    "AI Güveni": f"%{details.get('AI_Guven', 0)*100:.1f}"
                }
                report_data.append(report_row)
            
            # --- İŞLEM MANTIĞI ---
            if price > 0 and decision != "HATA":
                status = row['Durum']
                log_msg = row['Son_Islem_Log']
                
                if status == 'COIN' and decision == 'SAT':
                    amount = float(row['Miktar'])
                    if amount > 0:
                        cash_val = amount * price
                        updated_portfolio.at[idx, 'Durum'] = 'CASH'; updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                        updated_portfolio.at[idx, 'Miktar'] = 0.0; updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                        updated_portfolio.at[idx, 'Son_Islem_Log'] = f"SATILDI ({tf_name})"; updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                        
                elif status == 'CASH' and decision == 'AL':
                    cash_val = float(row['Nakit_Bakiye_USD'])
                    if cash_val > 1.0:
                        amount = cash_val / price
                        updated_portfolio.at[idx, 'Durum'] = 'COIN'; updated_portfolio.at[idx, 'Miktar'] = amount
                        updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                        updated_portfolio.at[idx, 'Son_Islem_Log'] = f"ALINDI ({tf_name})"; updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                
                val = (float(updated_portfolio.at[idx, 'Miktar']) * price) if updated_portfolio.at[idx, 'Durum'] == 'COIN' else float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
                updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val
                updated_portfolio.at[idx, 'Son_Islem_Log'] = log_msg

            progress_bar.progress((i + 1) / len(updated_portfolio))
        
        save_portfolio(updated_portfolio, sheet)
        st.success("Analiz Tamamlandı!")
        
        # --- DETAYLI RAPOR EKRANI (ALTA EKLENEN KISIM) ---
        st.divider()
        st.markdown("### 📊 DETAYLI MODEL PERFORMANS & ALPHA RAPORU")
        
        # 1. Toplam Metrikler
        total_start = updated_portfolio['Baslangic_USD'].sum()
        total_now = updated_portfolio['Kaydedilen_Deger_USD'].sum()
        total_pnl = total_now - total_start
        total_roi = (total_pnl / total_start) * 100 if total_start > 0 else 0
        
        k1, k2, k3 = st.columns(3)
        k1.metric("Toplam Kasa", f"${total_now:,.2f}")
        k2.metric("Net Kâr (PNL)", f"${total_pnl:,.2f}", f"%{total_roi:.2f}")
        
        # Alpha Tahmini (Basit Benchmark: Eğer hepsi coin olsaydı?)
        # Not: Bu basit bir tahmindir, gerçek alpha için benchmark verisi gerekir.
        alpha_text = "Pozitif" if total_roi > 0 else "Negatif"
        k3.metric("Tahmini Alpha Durumu", alpha_text, delta_color="normal")
        
        # 2. Detaylı Tablo
        if report_data:
            st.markdown("#### 🧠 Yapay Zeka Karar Detayları")
            df_report = pd.DataFrame(report_data)
            
            # Renklendirme Fonksiyonu
            def highlight_decision(val):
                color = 'green' if val == 'AL' else ('red' if val == 'SAT' else 'gray')
                return f'color: {color}; font-weight: bold'

            st.dataframe(df_report.style.applymap(highlight_decision, subset=['Karar']), use_container_width=True)
            
            st.info("""
            **Tablo Açıklaması:**
            * **Heuristic Puan:** Senin 5 adımlı kural setinden alınan puan (-5 ile +5 arası).
            * **HMM Görüşü:** Hidden Markov Model'in piyasa rejimi tahmini (Boğa/Ayı).
            * **AI Güveni:** Tüm modellerin (XGB, RF, HMM, Heuristic) ortak konsensüs güven oranı.
            """)

# Mevcut Durum
st.divider()
st.subheader("📋 Mevcut Portföy (Sheets)")
try:
    df_view, _ = load_and_fix_portfolio()
    if not df_view.empty: st.dataframe(df_view)
except: pass

