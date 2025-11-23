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

# İstatistik ve ML
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer # Yeni
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Smart Imputation", layout="wide")
st.title("🧠 Hedge Fund AI: Smart Imputation & Auto-Select")

# --- SABİTLER VE BAĞLANTI (Aynı) ---
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

# ... (connect_sheet, load_and_fix_portfolio, save_portfolio FONKSİYONLARI AYNEN GELECEK - bot.py'deki gibi) ...
# (Yer kaplamaması için burayı kısalttım, bot.py'deki aynı fonksiyonları kullanın)
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

# --- ANALİZ FONKSİYONLARI (bot.py ile birebir aynı olmalı) ---

def apply_kalman_filter(prices):
    xhat = np.zeros(len(prices)); P = np.zeros(len(prices)); xhatminus = np.zeros(len(prices)); Pminus = np.zeros(len(prices)); K = np.zeros(len(prices)); Q = 1e-5; R = 0.01**2
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, len(prices)):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k]/(Pminus[k]+R); xhat[k] = xhatminus[k]+K[k]*(prices.iloc[k]-xhatminus[k]); P[k] = (1-K[k])*Pminus[k]
    return pd.Series(xhat, index=prices.index)

def calculate_heuristic_score(df):
    if len(df)<150: return pd.Series(0.0, index=df.index)
    return (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0

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
    if timeframe=='W': df_res=df.resample('W').agg(agg) # dropna yok, impute edilecek
    elif timeframe=='M': df_res=df.resample('ME').agg(agg)
    else: df_res=df.copy()
    if len(df_res)<100: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change()
    df_res['avg_ret_5m'] = df_res['ret'].rolling(100).mean()*100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(750).mean()*100
    
    avg_feats = df_res[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    df_res['range_vol_delta'] = df_res['range'].pct_change(5)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(subset=['target'], inplace=True) # Sadece hedefi eksik olanı at
    return df_res

# --- SMART IMPUTATION ---
def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0), "Simple-Zero"
    imputers = {'KNN': KNNImputer(n_neighbors=5), 'MICE': IterativeImputer(max_iter=10, random_state=42), 'Mean': SimpleImputer(strategy='mean')}
    best_score = -np.inf; best_df = df.fillna(0); best_m = "Zero"
    
    val_size = 20
    tr = df.iloc[:-val_size]; val = df.iloc[-val_size:]
    y_tr = tr['target']; y_val = val['target']
    
    for name, imp in imputers.items():
        try:
            X_tr_imp = imp.fit_transform(tr[features])
            X_val_imp = imp.transform(val[features])
            rf = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X_tr_imp, y_tr)
            s = rf.score(X_val_imp, y_val)
            if s > best_score:
                best_score = s; best_m = name
                full_imp = imp.fit_transform(df[features])
                best_df = pd.DataFrame(full_imp, columns=features, index=df.index)
                for c in df.columns: 
                    if c not in features: best_df[c] = df[c]
        except: continue
    return best_df, best_m

# --- MODELLER ---
def estimate_models(train, test): return 0.0 # Streamlit için basitleştirildi

def ga_optimize(df, features):
    # Basit optimizasyon
    return {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}

def train_meta_learner(df, params):
    test_size=60
    if len(df)<150: return 0.0, None
    train=df.iloc[:-test_size]; test=df.iloc[-test_size:]
    
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[features]; y_tr = train['target']
    X_test = test[features]
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3).fit(X_tr, y_tr)
    
    # Meta (Basitleştirilmiş)
    meta_X = pd.DataFrame({'RF': rf.predict_proba(X_tr)[:,1], 'XGB': xgb_c.predict_proba(X_tr)[:,1]}, index=train.index)
    meta_model = LogisticRegression().fit(meta_X, y_tr)
    
    mx_test = pd.DataFrame({'RF': rf.predict_proba(X_test)[:,1], 'XGB': xgb_c.predict_proba(X_test)[:,1]}, index=test.index)
    probs = meta_model.predict_proba(mx_test)[:,1]
    
    sim_eq = [100]
    for i in range(len(test)):
        ret = test['ret'].iloc[i]
        if probs[i]>0.55: sim_eq.append(sim_eq[-1]*(1+ret))
        else: sim_eq.append(sim_eq[-1])
        
    weights = dict(zip(meta_X.columns, meta_model.coef_[0]))
    return (probs[-1]-0.5)*2, {'bot_roi': sim_eq[-1]-100, 'weights': weights, 'dates': test.index, 'sim_eq': sim_eq}

def analyze_ticker_tournament(ticker):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    
    current_price = float(raw_df['close'].iloc[-1])
    best_roi = -9999; final_res = None
    
    for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
        df_raw = process_data(raw_df, tf_code)
        if df_raw is None: continue
        
        # Smart Impute
        feats = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        df_imp, method = smart_impute(df_raw, feats)
        
        sig, info = train_meta_learner(df_imp, ga_optimize(df_imp, feats))
        
        if info and info['bot_roi'] > best_roi:
            best_roi = info['bot_roi']
            final_res = {
                'ticker': ticker, 'price': current_price, 'roi': best_roi,
                'signal': sig, 'tf': tf_name, 'info': info, 'method': method
            }
    return final_res

# =============================================================================
# ARAYÜZ
# =============================================================================
st.markdown("### 📈 Portföy Durumu & Smart Imputation")
pf_df, sheet = load_and_fix_portfolio()

if not pf_df.empty:
    total_coin = pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    parked = pf_df['Nakit_Bakiye_USD'].sum()
    total = total_coin + parked
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Toplam Varlık", f"${total:.2f}")
    c2.metric("Coinlerdeki Para", f"${total_coin:.2f}")
    c3.metric("Nakitteki Para", f"${parked:.2f}")
    
    st.dataframe(pf_df[['Ticker','Durum','Miktar','Kaydedilen_Deger_USD','Son_Islem_Log']], use_container_width=True)
    
    if st.button("🚀 ANALİZ ET (Imputation Testli)", type="primary"):
        updated = pf_df.copy()
        total_pool = updated['Nakit_Bakiye_USD'].sum()
        results = []
        prog = st.progress(0)
        tz = pytz.timezone('Europe/Istanbul')
        time_str = datetime.now(tz).strftime("%d-%m %H:%M")
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            res = analyze_ticker_tournament(ticker)
            if res:
                res['idx']=idx; res['status']=row['Durum']; res['amount']=float(row['Miktar'])
                results.append(res)
                
                with st.expander(f"📊 {ticker} | Imputation: {res['method']} | ROI: %{res['roi']:.2f}"):
                    info = res['info']
                    # Grafik
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_eq'], name='Bot', line=dict(color='#00CC96', width=2)))
                    st.plotly_chart(fig, use_container_width=True)
            prog.progress((i+1)/len(updated))
            
        # Ortak Kasa Mantığı
        for r in results:
            if r['status'] == 'COIN' and r['signal'] < -0.1:
                rev = r['amount'] * r['price']
                total_pool += rev
                updated.at[r['idx'], 'Durum'] = 'CASH'; updated.at[r['idx'], 'Miktar'] = 0.0
                updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated.at[r['idx'], 'Son_Islem_Log'] = f"SAT ({r['tf']})"
                updated.at[r['idx'], 'Son_Islem_Zamani'] = time_str
                st.toast(f"🔻 SATILDI: {r['ticker']}")

        buy_cands = [r for r in results if r['signal'] > 0.1]
        buy_cands.sort(key=lambda x: x['roi'], reverse=True)
        
        if buy_cands and total_pool > 1.0:
            winner = buy_cands[0]
            if updated.at[winner['idx'], 'Durum'] == 'CASH':
                amt = total_pool / winner['price']
                updated.at[winner['idx'], 'Durum'] = 'COIN'; updated.at[winner['idx'], 'Miktar'] = amt
                updated.at[winner['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated.at[winner['idx'], 'Son_Islem_Fiyati'] = winner['price']
                updated.at[winner['idx'], 'Son_Islem_Log'] = f"AL ({winner['tf']}) Lider"
                updated.at[winner['idx'], 'Son_Islem_Zamani'] = time_str
                
                for idx in updated.index:
                    if idx != winner['idx'] and updated.at[idx, 'Durum'] == 'CASH':
                        updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                st.success(f"🚀 YENİ YATIRIM: {winner['ticker']} (ROI: %{winner['roi']:.2f})")
        elif total_pool > 0:
            f_idx = updated.index[0]
            updated.at[f_idx, 'Nakit_Bakiye_USD'] += total_pool
            for idx in updated.index:
                if idx != f_idx and updated.at[idx, 'Durum'] == 'CASH': updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0

        # Değerleme
        for idx, row in updated.iterrows():
            price = next((r['price'] for r in results if r['idx'] == idx), 0.0)
            if price > 0:
                val = (float(updated.at[idx, 'Miktar']) * price) if updated.at[idx, 'Durum'] == 'COIN' else float(updated.at[idx, 'Nakit_Bakiye_USD'])
                updated.at[idx, 'Kaydedilen_Deger_USD'] = val

        save_portfolio(updated, sheet)
        st.success("✅ Analiz Tamamlandı!")
