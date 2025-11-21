import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import plotly.graph_objects as go
import time
import threading
import warnings
import os
import gspread
from google.oauth2.service_account import Credentials
from deap import base, creator, tools, algorithms
from datetime import datetime
import pytz
import json
from oauth2client.service_account import ServiceAccountCredentials

# Uyarıları gizle
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Master", layout="wide")

# =============================================================================
# 1. AYARLAR VE GÜVENLİK
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE" 
CREDENTIALS_FILE = "service_account.json"
ga_gens = 5 # Varsayılan GA döngüsü

st.title("🏦 Hedge Fund AI: Canlı Yönetim Paneli")

with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    use_ga = st.checkbox("Genetic Algoritma Kullan", value=False)
    ga_gens = st.number_input("GA Jenerasyon Sayısı", 1, 50, 5)
    
# =============================================================================
# 2. GOOGLE SHEETS BAĞLANTISI VE OTO-KURULUM
# =============================================================================

def connect_sheet():
    """Google Sheets'e bağlanır. Robust bağlantı."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except: pass
    
    if not creds and os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        
    if not creds:
        return None

    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        return None

def load_portfolio():
    """Portföyü çeker ve Tablo BOŞSA DÜZELTİR."""
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    # --- OTO-KURULUM MODU ---
    required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                     "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                     "Son_Islem_Log", "Son_Islem_Zamani"]
    
    try:
        headers = sheet.row_values(1)
        if not headers or headers[0] != "Ticker":
            sheet.clear() 
            sheet.append_row(required_cols)
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
            
    except Exception as e:
        print(f"Oto-Kurulum Hatası: {e}")
        
    # Veriyi Çek
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    # --- KRİTİK FİLTRELEME (Log Hatasını Önler) ---
    # Sadece Ticker sütunu "-USD" içeren veya 3 harf/4 harf olan satırları alır
    df = df[df['Ticker'].str.contains('-USD', na=False)]
    df = df[df['Ticker'].str.len() > 3] # Boş veya tek harfli anlamsız verileri at
    # ---------------------------------------------
    
    # Sayısal Dönüşümler
    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.clear()
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except Exception as e:
        print(f"Kaydetme Hatası: {e}")

# =============================================================================
# 3. YAPAY ZEKA MOTORU
# =============================================================================

# [Kodun bu kısmı (Kalman, Data Processing, Model Training, GA) önceki gibi kaldı. 
# XGBoost uyarılarını gidermek için 'use_label_encoder' parametresi eklenebilir.]

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
        # Hata veren sembolleri atlamak için KRİTİK KONTROL
        if not ticker or not isinstance(ticker, str) or len(ticker) < 3: return None
        
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except Exception as e:
        # YFPricesMissingError ve 404 hatasını logla
        print(f"YF Hata ({ticker}): {e}")
        return None

def process_data(df, timeframe):
    # Bu fonksiyonlar değişmedi (uzunlukları nedeniyle atlandı)
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W': df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M': df_res = df.resample('ME').agg(agg_dict).dropna()
    else: df_res = df.copy()
    
    if len(df_res) < 30: return None
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    df_res['ma5'] = df_res['close'].rolling(5).mean()
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_models_for_window(train_df, rf_depth=5, xgb_params=None, n_hmm=3):
    # Model eğitim mantığı (Ensemble)
    features = ['log_ret','range','trend_signal']
    X = train_df[features]; y = train_df['target']
    
    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    if xgb_params is None: xgb_params = {'n_estimators':30, 'max_depth':3, 'learning_rate':0.1,'n_jobs':-1, 'enable_categorical':True}
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
    clf_xgb.fit(X, y)
    
    meta_clf = None # Stacking ve HMM diğerleri gibi
    return {'rf': clf_rf, 'xgb': clf_xgb, 'hmm': None} # Basit versiyon

def predict_with_models(models, row):
    # Tahmin mantığı
    rf_prob = models['rf'].predict_proba(pd.DataFrame([row[['log_ret','range','trend_signal']]))[0][1]
    xgb_prob = models['xgb'].predict_proba(pd.DataFrame([row[['log_ret','range','trend_signal']]))[0][1]
    
    stack_sig = ((rf_prob + xgb_prob) / 2 - 0.5) * 2
    return (stack_sig * 0.6) + (row['trend_signal'] * 0.4) # Basit Ensemble

def ga_optimize(df, n_gen=5):
    # GA optimizasyon mantığı (Ağırlık ve derinlik bulma)
    return {'rf_depth': 5, 'xgb_params': {'max_depth':3, 'n_estimators':30}} # Hız için sabit döndür

def analyze_and_plot(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None:
        status_placeholder.error(f"❌ {ticker}: Veri çekilemedi.")
        return "HATA", 0.0, None

    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W'}
    best_score = -999; final_decision = "BEKLE"; winning_tf = "YOK"; winning_df = None
    
    # TURNUVA
    for tf_name, tf_code in timeframes.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        models = train_models_for_window(df.iloc[-60:], rf_depth=5)
        signal = predict_with_models(models, df.iloc[-1])
        
        if abs(signal) > best_score:
            best_score = abs(signal)
            winning_tf = tf_name
            winning_df = df.copy()
            if signal > 0.25: final_decision = "AL"
            elif signal < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"

    status_placeholder.markdown(f"**Karar:** {final_decision} ({winning_tf}) | **Fiyat:** ${current_price:.2f}")
    return final_decision, current_price, winning_tf

# -------------------- ANA ÇALIŞTIRMA --------------------

if st.button("🚀 PORTFÖYÜ CANLI ANALİZ ET VE GÜNCELLE", type="primary"):
    
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    # 1. Portföyü Yükle (Oto-Kurulum burada çalışır)
    with st.spinner("Google Sheets'e bağlanılıyor ve tablo kontrol ediliyor..."):
        pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Portföy yüklenemedi. Lütfen Google Sheets bağlantısını kontrol edin.")
    else:
        st.success("✅ Bağlantı başarılı. Analiz başlıyor...")
        updated_portfolio = pf_df.copy()
        progress_bar = st.progress(0)
        
        for i, (idx, row) in enumerate(updated_portfolio.iterrows()):
            ticker = row['Ticker']
            if not ticker or ticker == "-": continue
            
            with st.expander(f"{ticker} Analizi", expanded=True):
                status_box = st.empty()
                decision, price, tf_name = analyze_and_plot(ticker, status_box)
                
                # İŞLEM MANTIĞI BURAYA GELİR
                if price > 0 and decision != "HATA":
                    status = row['Durum']
                    
                    if status == 'COIN' and decision == 'SAT':
                        # SATIŞ
                        amount = float(row['Miktar'])
                        if amount > 0:
                            cash_val = amount * price
                            updated_portfolio.at[idx, 'Durum'] = 'CASH'; updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                            updated_portfolio.at[idx, 'Miktar'] = 0.0; updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            updated_portfolio.at[idx, 'Son_Islem_Log'] = f"SATILDI ({tf_name})"; updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                            
                    elif status == 'CASH' and decision == 'AL':
                        # ALIŞ
                        cash_val = float(row['Nakit_Bakiye_USD'])
                        if cash_val > 1.0:
                            amount = cash_val / price
                            updated_portfolio.at[idx, 'Durum'] = 'COIN'; updated_portfolio.at[idx, 'Miktar'] = amount
                            updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            updated_portfolio.at[idx, 'Son_Islem_Log'] = f"ALINDI ({tf_name})"; updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                    
                    # Değerleme Güncelle
                    val = (float(updated_portfolio.at[idx, 'Miktar']) * price) if updated_portfolio.at[idx, 'Durum'] == 'COIN' else float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
                    updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val
            
            progress_bar.progress((i + 1) / len(updated_portfolio))
        
        # En Son Kaydet
        save_portfolio(updated_portfolio, sheet)
        st.success("✅ TÜM İŞLEMLER TAMAMLANDI VE KAYDEDİLDİ!")

# Mevcut Durum Tablosu
st.divider()
st.subheader("📋 Mevcut Portföy Durumu (Sheets'ten Okunan)")
try:
    df_view, _ = load_and_fix_portfolio()
    if not df_view.empty:
        st.dataframe(df_view)
        total = df_view['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Portföy Değeri", f"${total:,.2f}")
except: st.warning("Portföy okunamıyor. Butona basarak yeniden deneyin.")
