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

# İstatiksel ve ML Kütüphaneleri
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Yönetim Paneli", layout="wide", page_icon="🏦")

# =============================================================================
# 1. AYARLAR VE SABİTLER
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

# =============================================================================
# 2. GOOGLE SHEETS ENTEGRASYONU
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
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", "Son_Islem_Log", "Son_Islem_Zamani"]
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            defaults = []
            for t in TARGET_COINS: defaults.append([t, "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"])
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
# 3. ANALİZ MOTORU (ÖZETLENMİŞ)
# =============================================================================
# (Buradaki fonksiyonlar bot.py ile birebir aynıdır, yer kaplamaması için standart hallerini kullanıyoruz)

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
    if timeframe=='W': df_res=df.resample('W').agg(agg).dropna()
    elif timeframe=='M': df_res=df.resample('ME').agg(agg).dropna()
    else: df_res=df.copy()
    if len(df_res)<100: return None
    
    # Basit Özellik Üretimi (Detaylısı bot.py ile aynı olmalı)
    df_res['ret'] = df_res['close'].pct_change()
    df_res['log_ret'] = np.log(df_res['close']/df_res['close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['range_vol_delta'] = df_res['range'].pct_change(5).fillna(0)
    
    # Heuristic
    df_res['heuristic'] = (np.sign(df_res['close'].pct_change(5)) + np.sign(df_res['close'].pct_change(30)))/2.0
    
    # Avg Scores
    avg_feats = df_res[['ret']].rolling(100).mean()
    df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats.fillna(0)).flatten()
    
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

def ga_optimize(df):
    return {'rf_depth': 5, 'rf_nest': 100, 'xgb_params': {'max_depth':5, 'n_estimators':100}}

def train_meta_learner(df, params):
    test_size=30
    train=df.iloc[:-test_size]; test=df.iloc[-test_size:]
    
    # HMM
    scaler_hmm = StandardScaler()
    X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None
    
    hmm_probs = hmm.predict_proba(X_hmm) if hmm else np.zeros((len(train),3))
    hmm_df = pd.DataFrame(hmm_probs, columns=['HMM_0','HMM_1','HMM_2'], index=train.index)

    # ML Models
    base_features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[base_features]; y_tr = train['target']
    X_test = test[base_features]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5).fit(X_tr, y_tr)
    
    # Meta-X
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'ETC': etc.predict_proba(X_tr)[:,1],
        'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'Heuristic': train['heuristic'],
        'HMM_0': hmm_df['HMM_0'], 'HMM_1': hmm_df['HMM_1'], 'HMM_2': hmm_df['HMM_2']
    }, index=train.index).fillna(0)
    
    # Train Meta
    meta_model = LogisticRegression(C=1.0, solver='liblinear').fit(meta_X, y_tr)
    weights = meta_model.coef_[0]
    
    # Test Prep
    X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']])
    hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_df_t = pd.DataFrame(hmm_probs_t, columns=['HMM_0','HMM_1','HMM_2'], index=test.index)
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'ETC': etc.predict_proba(X_test)[:,1],
        'XGB': xgb_c.predict_proba(X_test)[:,1],
        'Heuristic': test['heuristic'],
        'HMM_0': hmm_df_t['HMM_0'], 'HMM_1': hmm_df_t['HMM_1'], 'HMM_2': hmm_df_t['HMM_2']
    }, index=test.index).fillna(0)
    
    probs = meta_model.predict_proba(mx_test)[:,1]
    
    # Sim
    sim_eq=[100]
    for i in range(len(test)):
        s = (probs[i]-0.5)*2
        # Basit simülasyon (Sadece getiri hesabı için)
        ret = test['ret'].iloc[i]
        if s > 0.1: sim_eq.append(sim_eq[-1] * (1+ret))
        else: sim_eq.append(sim_eq[-1])
        
    weights_dict = dict(zip(meta_X.columns, weights))
    return (probs[-1]-0.5)*2, {'bot_roi': sim_eq[-1]-100, 'weights': weights_dict, 'dates': test.index, 'sim_eq': sim_eq}

def analyze_ticker_tournament(ticker):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    
    current_price = float(raw_df['close'].iloc[-1])
    best_roi = -9999; final_res = None
    
    for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        sig, info = train_meta_learner(df, ga_optimize(df))
        if info and info['bot_roi'] > best_roi:
            best_roi = info['bot_roi']
            final_res = {
                'ticker': ticker, 'price': current_price, 'roi': best_roi,
                'signal': sig, 'tf': tf_name, 'info': info
            }
    return final_res

# =============================================================================
# ARAYÜZ TASARIMI
# =============================================================================

st.title("🏦 Hedge Fund AI: Yönetim Paneli")
st.markdown("Bu panel, arka planda çalışan botun kararlarını ve portföyün canlı durumunu gösterir.")

pf_df, sheet = load_and_fix_portfolio()

if not pf_df.empty:
    # --- 1. ÜST BİLGİ KARTLARI ---
    total_coin_val = pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    total_cash_val = pf_df[pf_df['Durum']=='CASH']['Kaydedilen_Deger_USD'].sum()
    # Nakit Bakiye sütunundaki (havuza alınmış ama dağıtılmamış) parayı da ekle
    parked_cash = pf_df['Nakit_Bakiye_USD'].sum() 
    
    total_portfolio = total_coin_val + parked_cash
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Portföy Değeri", f"${total_portfolio:.2f}", delta_color="normal")
    c2.metric("Aktif Yatırım (Coinler)", f"${total_coin_val:.2f}")
    c3.metric("Boştaki Nakit (Havuz)", f"${parked_cash:.2f}")
    
    st.divider()
    
    # --- 2. DETAYLI PORTFÖY TABLOSU ---
    st.subheader("📋 Mevcut Portföy Durumu")
    
    # Görsel olarak daha iyi bir tablo hazırlayalım
    display_df = pf_df[['Ticker', 'Durum', 'Miktar', 'Kaydedilen_Deger_USD', 'Son_Islem_Log', 'Son_Islem_Zamani']].copy()
    display_df['Kaydedilen_Deger_USD'] = display_df['Kaydedilen_Deger_USD'].apply(lambda x: f"${x:.2f}")
    display_df['Miktar'] = display_df['Miktar'].apply(lambda x: f"{x:.6f}")
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # --- 3. ANALİZ & İŞLEM TETİKLEME ---
    st.divider()
    col_btn, col_info = st.columns([1, 3])
    
    with col_info:
        st.info("💡 'Analiz Et ve Yönet' butonuna bastığınızda, tüm coinler güncel verilerle analiz edilir, en kârlı fırsat bulunur ve 'Ortak Kasa' mantığıyla para transferi yapılır.")
        
    if col_btn.button("🚀 ANALİZ ET VE YÖNET", type="primary"):
        updated = pf_df.copy()
        tz = pytz.timezone('Europe/Istanbul')
        time_str = datetime.now(tz).strftime("%d-%m %H:%M")
        
        # A. Ortak Havuzu Topla
        total_pool = updated['Nakit_Bakiye_USD'].sum()
        st.write(f"💰 **Başlangıç Nakit Havuzu:** ${total_pool:.2f}")
        
        results = []
        progress_bar = st.progress(0)
        
        # B. Tüm Coinleri Analiz Et
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            res = analyze_ticker_tournament(ticker)
            
            if res:
                res['idx'] = idx
                res['current_status'] = row['Durum']
                res['current_amount'] = float(row['Miktar'])
                results.append(res)
                
                # Model Ağırlıklarını Göster (Expandable)
                with st.expander(f"📊 {ticker} Detaylı Analiz (ROI: %{res['roi']:.2f})"):
                    wc1, wc2 = st.columns([1, 2])
                    w_df = pd.DataFrame.from_dict(res['info']['weights'], orient='index', columns=['Etki']).sort_values(by='Etki', ascending=False)
                    w_df['Etki'] = w_df['Etki'].abs() # Mutlak değer görseli daha iyi
                    
                    wc1.dataframe(w_df)
                    wc2.bar_chart(w_df)
                    
                    # Karar
                    decision = "AL" if res['signal'] > 0.1 else ("SAT" if res['signal'] < -0.1 else "BEKLE")
                    color = "green" if decision == "AL" else ("red" if decision == "SAT" else "gray")
                    st.markdown(f"### Model Kararı: :{color}[{decision}]")

            progress_bar.progress((i + 1) / len(updated))
            
        # C. Satışları Yap (Nakit Yarat)
        for r in results:
            if r['current_status'] == 'COIN' and r['signal'] < -0.1: # SAT
                revenue = r['current_amount'] * r['price']
                total_pool += revenue
                
                updated.at[r['idx'], 'Durum'] = 'CASH'
                updated.at[r['idx'], 'Miktar'] = 0.0
                updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0 # Havuza gitti
                updated.at[r['idx'], 'Son_Islem_Log'] = f"SAT ({r['tf']}) -> Havuz"
                updated.at[r['idx'], 'Son_Islem_Zamani'] = time_str
                
                st.toast(f"🔻 {r['ticker']} Satıldı! Havuza +${revenue:.2f} eklendi.")
        
        st.write(f"💵 **Dağıtılabilir Toplam Havuz:** ${total_pool:.2f}")
        
        # D. En İyi Alım Fırsatını Bul (Winner Takes All)
        buy_candidates = [r for r in results if r['signal'] > 0.1]
        buy_candidates.sort(key=lambda x: x['roi'], reverse=True)
        
        if buy_candidates and total_pool > 1.0:
            winner = buy_candidates[0]
            
            # Eğer kazanan zaten coin ise ve tutuyorsak dokunma, ama nakitteyse al
            if updated.at[winner['idx'], 'Durum'] == 'CASH':
                amount_to_buy = total_pool / winner['price']
                
                updated.at[winner['idx'], 'Durum'] = 'COIN'
                updated.at[winner['idx'], 'Miktar'] = amount_to_buy
                updated.at[winner['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated.at[winner['idx'], 'Son_Islem_Fiyati'] = winner['price']
                updated.at[winner['idx'], 'Son_Islem_Log'] = f"AL ({winner['tf']}) Lider"
                updated.at[winner['idx'], 'Son_Islem_Zamani'] = time_str
                
                # Diğer tüm nakitleri sıfırla (Hepsi winner'a gitti)
                for idx in updated.index:
                    if idx != winner['idx'] and updated.at[idx, 'Durum'] == 'CASH':
                        updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                
                st.success(f"🚀 YENİ YATIRIM: {winner['ticker']} (Beklenen ROI: %{winner['roi']:.2f}) - Yatırılan: ${total_pool:.2f}")
            else:
                st.info(f"👍 Lider ({winner['ticker']}) zaten elimizde. Pozisyon korunuyor.")
        
        elif total_pool > 0:
            # Hiçbir şey alınmadıysa parayı park et
            first_idx = updated.index[0]
            current_parked = float(updated.at[first_idx, 'Nakit_Bakiye_USD'])
            updated.at[first_idx, 'Nakit_Bakiye_USD'] = current_parked + total_pool
            
            # Diğer nakitleri temizle
            for idx in updated.index:
                if idx != first_idx and updated.at[idx, 'Durum'] == 'CASH':
                    updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                    
            st.warning(f"⏸️ Uygun alım fırsatı bulunamadı. ${total_pool:.2f} nakitte bekletiliyor.")

        # E. Değer Güncelleme ve Kayıt
        for idx, row in updated.iterrows():
            # Güncel fiyatı results listesinden bul (tekrar çekmemek için)
            price = next((r['price'] for r in results if r['idx'] == idx), 0.0)
            if price > 0:
                if updated.at[idx, 'Durum'] == 'COIN':
                    val = float(updated.at[idx, 'Miktar']) * price
                else:
                    val = float(updated.at[idx, 'Nakit_Bakiye_USD'])
                updated.at[idx, 'Kaydedilen_Deger_USD'] = val
        
        save_portfolio(updated, sheet)
        st.balloons()
        time.sleep(2)
        st.rerun()
