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

# Sadece Gerekli ML Kütüphaneleri
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Pro", layout="wide", page_icon="⚡")
st.title("⚡ Hedge Fund AI: Professional Engine")

# --- AYARLAR ---
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "2y"

with st.sidebar:
    st.header("Sistem Durumu")
    st.success("✅ Gürültü Filtresi Aktif")
    st.success("✅ Adaptif Kalman Aktif")
    st.success("✅ XGBoost Meta-Learner Aktif")

# --- BAĞLANTI ---
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
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", "Son_Islem_Log", "Son_Islem_Zamani"])
            for t in TARGET_COINS: sheet.append_row([t, "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"])
    except: pass
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    return df, sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    try: sheet.update([df.columns.values.tolist()] + df.astype(str).values.tolist())
    except: pass

# --- OPTİMİZE EDİLMİŞ ANALİZ ---
def apply_kalman_filter(prices):
    xhat = np.zeros(len(prices)); P = np.zeros(len(prices)); xhatminus = np.zeros(len(prices)); Pminus = np.zeros(len(prices)); K = np.zeros(len(prices)); Q = 1e-5
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    rolling_std = pd.Series(prices).rolling(30).std().fillna(method='bfill').values
    for k in range(1, len(prices)):
        R = (rolling_std[k]*0.1)**2 if rolling_std[k]>0 else 0.01**2
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k]/(Pminus[k]+R); xhat[k] = xhatminus[k]+K[k]*(prices.iloc[k]-xhatminus[k]); P[k] = (1-K[k])*Pminus[k]
    return pd.Series(xhat, index=prices.index)

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df)<100: return None
    agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    if timeframe=='W': df_res=df.resample('W').agg(agg)
    else: df_res=df.copy()
    if len(df_res)<60: return None
    
    df_res['kalman'] = apply_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['close']/df_res['close'].shift(1))
    df_res['trend_kalman'] = np.where(df_res['close'] > df_res['kalman'], 1, -1)
    df_res['rsi_proxy'] = df_res['close'].pct_change().rolling(14).mean() / df_res['close'].pct_change().rolling(14).std()
    
    hl = df_res['high'] - df_res['low']
    hc = np.abs(df_res['high'] - df_res['close'].shift())
    lc = np.abs(df_res['low'] - df_res['close'].shift())
    tr = np.max(pd.concat([hl, hc, lc], axis=1), axis=1)
    atr = tr.rolling(14).mean()
    df_res['vol_regime'] = (atr / atr.rolling(50).mean()).fillna(1.0)
    
    df_res['momentum'] = (np.sign(df_res['close'].diff(5)) + np.sign(df_res['close'].diff(20)))
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

def train_smart_ensemble(df):
    test_size = 30
    features = ['log_ret', 'trend_kalman', 'vol_regime', 'momentum', 'rsi_proxy']
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    
    X_tr = train[features]; y_tr = train['target']
    X_te = test[features]
    
    imputer = SimpleImputer(strategy='mean')
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr_imp, y_tr)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr_imp, y_tr)
    xgb_b = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss').fit(X_tr_imp, y_tr)
    
    meta_X_tr = pd.DataFrame({'RF': rf.predict_proba(X_tr_imp)[:,1], 'ET': et.predict_proba(X_tr_imp)[:,1], 'XGB': xgb_b.predict_proba(X_tr_imp)[:,1], 'Vol': train['vol_regime'].values})
    meta_X_te = pd.DataFrame({'RF': rf.predict_proba(X_te_imp)[:,1], 'ET': et.predict_proba(X_te_imp)[:,1], 'XGB': xgb_b.predict_proba(X_te_imp)[:,1], 'Vol': test['vol_regime'].values})
    
    meta_model = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, eval_metric='logloss').fit(meta_X_tr, y_tr)
    probs = meta_model.predict_proba(meta_X_te)[:,1]
    
    sim_eq = [100]
    for i in range(len(test)):
        ret = (test['close'].iloc[i] - test['close'].iloc[i-1])/test['close'].iloc[i-1] if i>0 else 0
        pos_size = np.tanh(3 * (probs[i] - 0.5) * 2)
        if pos_size > 0.2: sim_eq.append(sim_eq[-1] * (1 + ret * abs(pos_size)))
        else: sim_eq.append(sim_eq[-1])
        
    info = {'roi': sim_eq[-1]-100, 'sim_eq': sim_eq, 'dates': test.index, 'imp': meta_model.feature_importances_, 'cols': meta_X_tr.columns}
    return (probs[-1]-0.5)*2, info

def analyze_ticker(ticker):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    current_p = float(raw_df['close'].iloc[-1])
    best_roi = -999; best_res = None
    
    for tf in ['GÜNLÜK', 'HAFTALIK']:
        code = 'D' if tf=='GÜNLÜK' else 'W'
        df = process_data(raw_df, code)
        if df is None: continue
        sig, info = train_smart_ensemble(df)
        if info['roi'] > best_roi:
            best_roi = info['roi']
            best_res = {'ticker':ticker, 'price':current_p, 'roi':best_roi, 'signal':sig, 'tf':tf, 'info':info}
    return best_res

# --- ARAYÜZ ---
pf_df, sheet = load_and_fix_portfolio()

if not pf_df.empty:
    total_coin = pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    parked = pf_df['Nakit_Bakiye_USD'].sum()
    
    c1,c2 = st.columns(2)
    c1.metric("Toplam Portföy", f"${total_coin+parked:.2f}")
    c2.metric("Nakit", f"${parked:.2f}")
    st.dataframe(pf_df[['Ticker','Durum','Miktar','Kaydedilen_Deger_USD','Son_Islem_Log']], use_container_width=True)
    
    if st.button("🚀 ANALİZ ET VE YÖNET", type="primary"):
        updated = pf_df.copy()
        total_pool = updated['Nakit_Bakiye_USD'].sum()
        results = []
        prog = st.progress(0)
        tz = pytz.timezone('Europe/Istanbul')
        time_str = datetime.now(tz).strftime("%d-%m %H:%M")
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            res = analyze_ticker(row['Ticker'])
            if res:
                res['idx']=idx; res['status']=row['Durum']; res['amount']=float(row['Miktar'])
                results.append(res)
                
                with st.expander(f"📊 {row['Ticker']} | ROI: %{res['roi']:.2f} | Sinyal Gücü: {res['signal']:.2f}"):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res['info']['dates'], y=res['info']['sim_eq'], name='Bot', line=dict(color='#00CC96', width=3)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature Importance
                    imp_df = pd.DataFrame({'Faktör': res['info']['cols'], 'Önem': res['info']['imp']}).sort_values(by='Önem', ascending=False)
                    st.bar_chart(imp_df.set_index('Faktör'))
            prog.progress((i+1)/len(updated))
            
        # Satış
        for r in results:
            if r['status']=='COIN' and r['signal'] < 0.2:
                rev = r['amount']*r['price']; total_pool+=rev
                updated.at[r['idx'],'Durum']='CASH'; updated.at[r['idx'],'Miktar']=0.0
                updated.at[r['idx'],'Nakit_Bakiye_USD']=0.0
                updated.at[r['idx'],'Son_Islem_Log']=f"SAT ({r['tf']})"
                st.toast(f"🔻 SATILDI: {r['ticker']}")

        # Alım
        buy_cands = [r for r in results if r['signal']>0.2 and r['roi']>0]
        total_sig = sum([r['signal'] for r in buy_cands])
        
        if buy_cands and total_pool > 1.0:
            for r in buy_cands:
                w = r['signal']/total_sig
                amt_usd = total_pool * w
                if updated.at[r['idx'],'Durum']=='CASH':
                    amt = amt_usd/r['price']
                    updated.at[r['idx'],'Durum']='COIN'; updated.at[r['idx'],'Miktar']=amt
                    updated.at[r['idx'],'Nakit_Bakiye_USD']=0.0
                    updated.at[r['idx'],'Son_Islem_Fiyati']=r['price']
                    updated.at[r['idx'],'Son_Islem_Log']=f"AL (Pay: %{w*100:.1f})"
                    updated.at[r['idx'],'Son_Islem_Zamani']=time_str
        elif total_pool > 0:
            f_idx = updated.index[0]
            updated.at[f_idx,'Nakit_Bakiye_USD'] += total_pool
            for idx in updated.index:
                if idx!=f_idx and updated.at[idx,'Durum']=='CASH': updated.at[idx,'Nakit_Bakiye_USD']=0.0
        
        for idx, row in updated.iterrows():
            p = next((r['price'] for r in results if r['idx']==idx), 0.0)
            if p>0: updated.at[idx,'Kaydedilen_Deger_USD'] = (float(updated.at[idx,'Miktar'])*p) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])

        save_portfolio(updated, sheet)
        st.success("✅ Tamamlandı!")
