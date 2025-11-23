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

# --- Yeni İstatiksel Kütüphaneler ---
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from arch import arch_model
from scipy.stats import boxcox, yeojohnson

# --- AI & ML Kütüphaneleri ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Auto-Select Modu", layout="wide")
st.title("🧠 Hedge Fund AI: Auto-Select Modu (XGB vs Ensemble)")

# =============================================================================
# 1. AYARLAR
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y" 

with st.sidebar:
    st.header("⚙️ Ayarlar")
    use_ga = st.checkbox("Genetic Algoritma (GA) Optimizasyonu", value=True)
    st.info("Sistem, Ensemble ve Solo XGBoost arasında performansı en iyi olanı otomatik seçer.")

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
# 3. ORTAK ANALİZ FONKSİYONLARI
# =============================================================================
def apply_kalman_filter(prices):
    xhat = np.zeros(len(prices)); P = np.zeros(len(prices)); xhatminus = np.zeros(len(prices)); Pminus = np.zeros(len(prices)); K = np.zeros(len(prices)); Q = 1e-5; R = 0.01**2
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, len(prices)):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k]/(Pminus[k]+R); xhat[k] = xhatminus[k]+K[k]*(prices.iloc[k]-xhatminus[k]); P[k] = (1-K[k])*Pminus[k]
    return pd.Series(xhat, index=prices.index)

def calculate_heuristic_score(df):
    if len(df)<150: return pd.Series(0.0, index=df.index)
    return (np.sign(df['close'].pct_change(5).fillna(0)) + np.sign(df['close'].pct_change(30).fillna(0)))/2.0

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
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change()
    
    df_res['avg_ret_5m'] = df_res['ret'].rolling(100).mean()*100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(750).mean()*100
    df_res['day_score'] = df_res.index.dayofweek.map(df_res.groupby(df_res.index.dayofweek)['ret'].mean().fillna(0)).fillna(0)
    
    avg_feats = df_res[['avg_ret_5m','avg_ret_3y','day_score']].fillna(0)
    if not avg_feats.empty: df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    else: df_res['historical_avg_score'] = 0.0
    
    df_res['range_vol_delta'] = df_res['range'].pct_change(5).fillna(0)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

def estimate_models(train, test):
    # Sadece ARIMA/GARCH tahminleri (Gösterim için)
    try:
        model = pm.auto_arima(np.log(train['close']/train['close'].shift(1)).dropna(), seasonal=False, trace=False, error_action='ignore')
        arima_ret = float((train['close'].iloc[-1] * np.exp(model.predict(1)[0]) / train['close'].iloc[-1]) - 1.0)
    except: arima_ret = 0.0
    return arima_ret

def ga_optimize(df):
    # Basitleştirilmiş GA: RF ve XGB Parametrelerini Seçer
    test_size = 30
    train = df.iloc[:-test_size]; val = df.iloc[-test_size:]
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    # 1. RF Optimize
    best_rf_score = -999; best_rf_params = {'d':5, 'n':100}
    for d in [3, 5, 7]:
        rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42).fit(train[features], train['target'])
        if rf.score(val[features], val['target']) > best_rf_score: best_rf_params={'d':d, 'n':100}
            
    # 2. XGBoost Optimize
    best_xgb_score = -999; best_xgb_params = {'d':3, 'lr':0.1, 'n':100}
    for d in [3, 5]:
        for lr in [0.01, 0.1]:
            xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=d, learning_rate=lr, eval_metric='logloss').fit(train[features], train['target'])
            if xgb_m.score(val[features], val['target']) > best_xgb_score: best_xgb_params={'d':d, 'lr':lr, 'n':100}
            
    return {'rf': best_rf_params, 'xgb': best_xgb_params}

def train_meta_learner_auto_select(df, params):
    test_size=30
    if len(df)<100: return 0.0, None
    train=df.iloc[:-test_size]; test=df.iloc[-test_size:]
    
    base_features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[base_features]; y_tr = train['target']
    X_test = test[base_features]
    
    if X_tr.empty: return 0.0, None
    
    # 1. XGBoost
    p_xgb = params['xgb']
    xgb_solo = xgb.XGBClassifier(n_estimators=p_xgb['n'], max_depth=p_xgb['d'], learning_rate=p_xgb['lr'], eval_metric='logloss').fit(X_tr, y_tr)
    
    # 2. Ensemble
    p_rf = params['rf']
    rf = RandomForestClassifier(n_estimators=p_rf['n'], max_depth=p_rf['d'], random_state=42).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=p_rf['n'], max_depth=p_rf['d'], random_state=42).fit(X_tr, y_tr)
    
    scaler_hmm = StandardScaler()
    X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None
    
    hmm_probs = hmm.predict_proba(X_hmm) if hmm else np.zeros((len(train),3))
    hmm_df = pd.DataFrame(hmm_probs, columns=['HMM_0','HMM_1','HMM_2'], index=train.index)
    
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'ETC': etc.predict_proba(X_tr)[:,1],
        'XGB': xgb_solo.predict_proba(X_tr)[:,1],
        'Heuristic': train['heuristic'],
        'HMM_0': hmm_df['HMM_0'], 'HMM_1': hmm_df['HMM_1'], 'HMM_2': hmm_df['HMM_2']
    }, index=train.index).fillna(0)
    
    scaler_meta = StandardScaler()
    meta_X_scaled = scaler_meta.fit_transform(meta_X)
    meta_model = LogisticRegression(C=1.0, solver='liblinear').fit(meta_X_scaled, y_tr)
    
    # Test
    X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']])
    hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_df_t = pd.DataFrame(hmm_probs_t, columns=['HMM_0','HMM_1','HMM_2'], index=test.index)
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'ETC': etc.predict_proba(X_test)[:,1],
        'XGB': xgb_solo.predict_proba(X_test)[:,1],
        'Heuristic': test['heuristic'],
        'HMM_0': hmm_df_t['HMM_0'], 'HMM_1': hmm_df_t['HMM_1'], 'HMM_2': hmm_df_t['HMM_2']
    }, index=test.index).fillna(0)
    
    mx_test_scaled = scaler_meta.transform(mx_test)
    
    probs_ensemble = meta_model.predict_proba(mx_test_scaled)[:,1]
    probs_xgb = xgb_solo.predict_proba(X_test)[:,1]
    
    # ROI Simülasyonu
    sim_ens=[100]; sim_xgb=[100]; sim_hodl=[100]; p0=test['close'].iloc[0]
    ce=100; ke=0; cx=100; kx=0
    
    for i in range(len(test)):
        p=test['close'].iloc[i]
        # Ens
        s_e = (probs_ensemble[i]-0.5)*2
        if s_e>0.1 and ce>0: ke=ce/p; ce=0
        elif s_e<-0.1 and ke>0: ce=ke*p; ke=0
        sim_ens.append(ce+ke*p)
        # XGB
        s_x = (probs_xgb[i]-0.5)*2
        if s_x>0.1 and cx>0: kx=cx/p; cx=0
        elif s_x<-0.1 and kx>0: cx=kx*p; kx=0
        sim_xgb.append(cx+kx*p)
        # HODL
        sim_hodl.append((100/p0)*p)
        
    roi_ens = sim_ens[-1]-100
    roi_xgb = sim_xgb[-1]-100
    
    if roi_xgb > roi_ens:
        final_sig = (probs_xgb[-1]-0.5)*2
        final_roi = roi_xgb
        method = "Solo XGBoost"
    else:
        final_sig = (probs_ensemble[-1]-0.5)*2
        final_roi = roi_ens
        method = "Ensemble"
        
    info = {
        'bot_roi': final_roi, 'method': method, 
        'sim_ens': sim_ens[1:], 'sim_xgb': sim_xgb[1:], 'sim_hodl': sim_hodl[1:],
        'dates': test.index, 'weights': dict(zip(meta_X.columns, meta_model.coef_[0]))
    }
    return final_sig, info

def analyze_ticker_tournament(ticker):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    current_price = float(raw_df['close'].iloc[-1])
    best_roi = -9999; final_res = None
    
    for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        params = ga_optimize(df)
        sig, info = train_meta_learner_auto_select(df, params)
        if info and info['bot_roi'] > best_roi:
            best_roi = info['bot_roi']
            final_res = {'ticker': ticker, 'price': current_price, 'roi': best_roi, 'signal': sig, 'tf': tf_name, 'info': info}
    return final_res

# =============================================================================
# ARAYÜZ
# =============================================================================

st.markdown("### 📈 Portföy Durumu")
pf_df, sheet = load_and_fix_portfolio()

if not pf_df.empty:
    total_coin = pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    parked = pf_df['Nakit_Bakiye_USD'].sum()
    total = total_coin + parked
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Toplam Varlık", f"${total:.2f}")
    c2.metric("Coinlerdeki Para", f"${total_coin:.2f}")
    c3.metric("Nakitteki Para", f"${parked:.2f}")
    
    st.dataframe(pf_df[['Ticker','Durum','Miktar','Kaydedilen_Deger_USD','Son_Islem_Log','Son_Islem_Zamani']], use_container_width=True, hide_index=True)
    
    if st.button("🚀 ANALİZ ET VE KARŞILAŞTIR (Manuel)", type="primary"):
        updated = pf_df.copy()
        tz = pytz.timezone('Europe/Istanbul')
        time_str = datetime.now(tz).strftime("%d-%m %H:%M")
        total_pool = updated['Nakit_Bakiye_USD'].sum()
        
        results = []
        prog = st.progress(0)
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            res = analyze_ticker_tournament(ticker)
            if res:
                res['idx']=idx; res['status']=row['Durum']; res['amount']=float(row['Miktar'])
                results.append(res)
                
                with st.expander(f"📊 {ticker} Analiz | Seçilen: {res['info']['method']} | ROI: %{res['roi']:.2f}"):
                    info = res['info']
                    color_ti = "green" if info['method']=="Ensemble" else "blue"
                    
                    # Grafik
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_ens'], name='Ensemble', line=dict(color='#00CC96', width=2)))
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_xgb'], name='XGBoost', line=dict(color='#636EFA', width=2)))
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_hodl'], name='HODL', line=dict(color='gray', width=1, dash='dot')))
                    fig.update_layout(title=f"Strateji Yarışı ({res['tf']})", height=300, template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption(f"Model: **{info['method']}** performansı daha iyi olduğu için seçildi.")
            
            prog.progress((i+1)/len(updated))
            
        # Ortak Kasa İşlemleri
        for r in results:
            if r['status'] == 'COIN' and r['signal'] < -0.1:
                rev = r['amount'] * r['price']
                total_pool += rev
                updated.at[r['idx'], 'Durum'] = 'CASH'; updated.at[r['idx'], 'Miktar'] = 0.0
                updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated.at[r['idx'], 'Son_Islem_Log'] = f"SAT ({r['info']['method']})"
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
                updated.at[winner['idx'], 'Son_Islem_Log'] = f"AL ({winner['info']['method']})"
                updated.at[winner['idx'], 'Son_Islem_Zamani'] = time_str
                
                for idx in updated.index:
                    if idx != winner['idx'] and updated.at[idx, 'Durum'] == 'CASH':
                        updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                st.success(f"🚀 ALINDI: {winner['ticker']} (ROI: %{winner['roi']:.2f})")
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
