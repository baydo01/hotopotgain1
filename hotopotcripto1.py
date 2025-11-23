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

# --- İstatistik ve ML ---
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from arch import arch_model
from sklearn.impute import KNNImputer, SimpleImputer
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Defensive", layout="wide", page_icon="🛡️")
st.title("🛡️ Hedge Fund AI: Defensive (Stop-Loss + Trend Filter)")

# =============================================================================
# 1. AYARLAR
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "2y"  # Veri periyodu biraz kısaltıldı, daha güncel trende odaklanması için

# =============================================================================
# 2. GOOGLE SHEETS
# =============================================================================
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        elif os.path.exists(CREDENTIALS_FILE):
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        if not creds: return None
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except: return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", "Son_Islem_Log"]
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            defaults = []
            for t in TARGET_COINS: defaults.append([t, "CASH", 0, 0, 10, 10, 10, "KURULUM"])
            for d in defaults: sheet.append_row(d)
            time.sleep(2)
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, sheet
    except: return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy().astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except: pass

# =============================================================================
# 3. GÜÇLENDİRİLMİŞ FEATURE ENGINEERING
# =============================================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df)<150: return None
    agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    if timeframe=='W': df_res=df.resample('W').agg(agg)
    else: df_res=df.copy()
    if len(df_res)<60: return None
    
    # --- YENİ GÖSTERGELER ---
    # 1. RSI (Momentum)
    df_res['rsi'] = calculate_rsi(df_res['close']).fillna(50)
    
    # 2. Trend Filtresi (SMA 50) - Fiyat 50 günlüğün üzerindeyse boğa, altındaysa ayı
    df_res['sma50'] = df_res['close'].rolling(50).mean()
    df_res['trend_filter'] = (df_res['close'] > df_res['sma50']).astype(int)
    
    # 3. Log Return ve Normal Return
    df_res['log_ret'] = np.log(df_res['close']/df_res['close'].shift(1))
    df_res['ret'] = df_res['close'].pct_change()
    
    # 4. Volatilite (ATR/SMA)
    hl = df_res['high'] - df_res['low']
    atr = hl.rolling(14).mean()
    df_res['vol_regime'] = (atr / df_res['close']).fillna(0)

    # Target (Gelecek)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(subset=['target', 'sma50', 'rsi'], inplace=True)
    
    return df_res

# --- MODELLEME VE SİMÜLASYON ---
def train_meta_learner(df):
    test_size = 60
    if len(df) < 150: return 0.0, None
    
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    features = ['log_ret', 'vol_regime', 'rsi', 'trend_filter'] # Trend ve RSI eklendi
    
    X_tr = train[features].fillna(0); y_tr = train['target']
    X_test = test[features].fillna(0)
    
    # Modeller (Daha tutucu ayarlar)
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=5, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, eval_metric='logloss').fit(X_tr, y_tr)
    
    # Meta Data
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'RSI_Norm': train['rsi'] / 100.0, # RSI'ı da meta modele verelim
    }, index=train.index).fillna(0)
    
    meta_model = xgb.XGBClassifier(n_estimators=50, max_depth=2).fit(meta_X, y_tr)
    
    # Test Tahminleri
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'XGB': xgb_c.predict_proba(X_test)[:,1],
        'RSI_Norm': test['rsi'] / 100.0,
    }, index=test.index).fillna(0)
    
    probs_ens = meta_model.predict_proba(mx_test)[:,1]
    
    # --- GELİŞMİŞ SİMÜLASYON (STOP-LOSS İLE) ---
    sim_balance = [100.0]
    p0 = test['close'].iloc[0]
    in_position = False
    entry_price = 0.0
    
    STOP_LOSS_PCT = 0.05  # %5 Stop Loss
    TAKE_PROFIT_PCT = 0.15 # %15 Kar Al
    
    for i in range(len(test)-1):
        current_price = test['close'].iloc[i]
        next_open = test['open'].iloc[i+1] # Bir sonraki açılışta işlem
        ret = test['ret'].iloc[i]
        
        # Sinyal Gücü
        prob = probs_ens[i]
        trend_ok = test['trend_filter'].iloc[i] == 1 # Sadece trend yukarıysa al
        
        # SATIŞ MANTIĞI (Stop Loss veya Sinyal)
        if in_position:
            # Anlık zarar hesapla
            pnl_pct = (current_price - entry_price) / entry_price
            
            # 1. Stop Loss: Zarar %5'i geçtiyse ACİL SAT
            if pnl_pct < -STOP_LOSS_PCT:
                sim_balance.append(sim_balance[-1]) # Nakite geç
                in_position = False
            # 2. Take Profit: Kar %15'i geçtiyse SAT
            elif pnl_pct > TAKE_PROFIT_PCT:
                 sim_balance.append(sim_balance[-1] * (1 + ret)) # O günün getirisini al ve çık
                 in_position = False
            # 3. Model "Düşecek" diyorsa SAT (Prob < 0.45)
            elif prob < 0.45:
                sim_balance.append(sim_balance[-1] * (1 + ret))
                in_position = False
            else:
                # Tutmaya devam et
                sim_balance.append(sim_balance[-1] * (1 + ret))
        
        # ALIŞ MANTIĞI
        else: # Nakitteyiz
            # Model çok eminse (>0.55) VE Trend yukarıysa AL
            # Ayı piyasasında (Trend aşağı) sadece çok çok eminse (>0.7) al
            threshold = 0.55 if trend_ok else 0.70
            
            if prob > threshold:
                in_position = True
                entry_price = current_price
                sim_balance.append(sim_balance[-1]) # Henüz getiri yok, pozisyona girdik
            else:
                sim_balance.append(sim_balance[-1]) # Nakitte kal
                
    roi = sim_balance[-1] - 100
    last_signal = (probs_ens[-1] - 0.5) * 2
    
    # Trend filtresi son sinyali baskılar
    last_trend = test['trend_filter'].iloc[-1]
    if last_trend == 0 and last_signal > 0:
        last_signal = last_signal * 0.5 # Ayı piyasasında al sinyalini zayıflat
        
    return last_signal, roi, sim_balance, test.index

# =============================================================================
# ARAYÜZ
# =============================================================================
st.markdown("### 🛡️ Defansif Portföy Yönetimi")
pf_df, sheet = load_and_fix_portfolio()

if not pf_df.empty:
    total_val = pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum() + pf_df['Nakit_Bakiye_USD'].sum()
    st.metric("Toplam Varlık", f"${total_val:.2f}")
    st.dataframe(pf_df, use_container_width=True)
    
    if st.button("🛡️ GÜVENLİ ANALİZİ BAŞLAT", type="primary"):
        updated = pf_df.copy()
        total_pool = updated['Nakit_Bakiye_USD'].sum()
        results = []
        prog = st.progress(0)
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            raw = get_raw_data(row['Ticker'])
            if raw is not None:
                # Günlük analiz
                processed = process_data(raw, 'D')
                if processed is not None:
                    # Basit Imputation
                    processed = processed.fillna(method='ffill').fillna(0)
                    
                    sig, roi, balance, dates = train_meta_learner(processed)
                    curr_price = raw['close'].iloc[-1]
                    
                    results.append({
                        'ticker': row['Ticker'], 'roi': roi, 'signal': sig, 
                        'price': curr_price, 'idx': idx, 'status': row['Durum'],
                        'dates': dates, 'balance': balance, 'amount': float(row['Miktar'])
                    })
                    
                    # Grafik
                    with st.expander(f"{row['Ticker']} | ROI: %{roi:.2f}"):
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dates, y=balance, name='AI Strategy', line=dict(color='green')))
                        st.plotly_chart(fig, use_container_width=True)
            prog.progress((i+1)/len(updated))
            
        # İŞLEM MANTIĞI (Satış Öncelikli)
        for r in results:
            # 1. SATIŞ: Sinyal çok düşükse VEYA ROI çok kötüyse (Stop Loss mantığı buraya da yansır)
            if r['status']=='COIN':
                if r['signal'] < -0.2:
                    val = r['amount'] * r['price']
                    total_pool += val
                    updated.at[r['idx'], 'Durum'] = 'CASH'
                    updated.at[r['idx'], 'Miktar'] = 0.0
                    updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0
                    updated.at[r['idx'], 'Son_Islem_Log'] = f"SAT (Signal: {r['signal']:.2f})"
                    st.toast(f"🔻 SATILDI: {r['ticker']}")
        
        # 2. ALIM
        buy_candidates = [r for r in results if r['signal'] > 0.3 and r['roi'] > -5] # ROI'si felaket olanı alma
        if buy_candidates and total_pool > 1.0:
            # Eşit dağıtım (Risk azaltmak için)
            per_share = total_pool / len(buy_candidates)
            for r in buy_candidates:
                if updated.at[r['idx'], 'Durum'] == 'CASH':
                    amt = per_share / r['price']
                    updated.at[r['idx'], 'Durum'] = 'COIN'
                    updated.at[r['idx'], 'Miktar'] = amt
                    updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0
                    updated.at[r['idx'], 'Son_Islem_Fiyati'] = r['price']
                    updated.at[r['idx'], 'Son_Islem_Log'] = f"AL (Güven: {r['signal']:.2f})"
        
        # Değerleme Güncelleme
        for idx, row in updated.iterrows():
            p = next((r['price'] for r in results if r['idx']==idx), 0.0)
            if p>0 and updated.at[idx,'Durum']=='COIN':
                updated.at[idx,'Kaydedilen_Deger_USD'] = float(updated.at[idx,'Miktar']) * p
            elif updated.at[idx,'Durum']=='CASH':
                updated.at[idx,'Kaydedilen_Deger_USD'] = updated.at[idx,'Nakit_Bakiye_USD']

        save_portfolio(updated, sheet)
        st.success("✅ Defansif Analiz Tamamlandı!")
