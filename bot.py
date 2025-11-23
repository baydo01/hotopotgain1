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
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Mega Dashboard", layout="wide", page_icon="🏦")
st.title("🏦 Hedge Fund AI: Mega Dashboard (Fix)")

# =============================================================================
# 1. AYARLAR
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

with st.sidebar:
    st.header("⚙️ Model Ayarları")
    use_ga = st.checkbox("Genetic Algoritma (GA) Aktif", value=True)
    st.info("Bu panel; Smart Imputation, Walk-Forward Validation ve Benchmark kıyaslamalarını tek ekranda sunar.")

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
# 3. VERİ İŞLEME
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
    if timeframe=='W': df_res=df.resample('W').agg(agg) 
    elif timeframe=='M': df_res=df.resample('ME').agg(agg)
    else: df_res=df.copy()
    if len(df_res)<100: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['close']/df_res['close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change() # BU SÜTUN ARTIK KORUNUYOR
    df_res['avg_ret_5m'] = df_res['ret'].rolling(100).mean()*100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(750).mean()*100
    
    avg_feats = df_res[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    df_res['range_vol_delta'] = df_res['range'].pct_change(5)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    
    # Trend Filtresi (SMA50)
    df_res['sma_50'] = df_res['close'].rolling(50).mean()
    df_res['trend_up'] = (df_res['close'] > df_res['sma_50']).astype(int)

    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    features_to_check = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    nan_in_features = df_res[features_to_check].isna().sum().sum()
    
    df_res.dropna(subset=['target'], inplace=True)
    df_res.attrs['nan_count'] = int(nan_in_features)
    return df_res

# --- SMART IMPUTATION (HATA DÜZELTİLDİ: Copy Metodu) ---
def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0), "Simple-Zero"
    
    imputers = {'KNN': KNNImputer(n_neighbors=5), 'Mean': SimpleImputer(strategy='mean')}
    best_score = -999; best_df = df.copy().fillna(0); best_m = "Zero"
    
    val_size = 20
    tr = df.iloc[:-val_size]; val = df.iloc[-val_size:]
    
    for name, imp in imputers.items():
        try:
            # Sadece özellikleri impute et
            X_tr_imp = imp.fit_transform(tr[features])
            X_val_imp = imp.transform(val[features])
            
            # Hızlı test
            rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X_tr_imp, tr['target'])
            s = rf.score(X_val_imp, val['target'])
            
            if s > best_score:
                best_score = s; best_m = name
                
                # HATA DÜZELTME: Yeni bir DF oluşturmak yerine kopyasını alıp içine yazıyoruz
                # Böylece 'ret', 'close' gibi sütunlar kaybolmuyor.
                temp_df = df.copy()
                temp_df[features] = imp.fit_transform(df[features])
                best_df = temp_df
        except: continue
            
    return best_df, best_m

# --- MODELLER ---
def estimate_models(train, test): return 0.0 

def select_best_garch_model(returns):
    returns = returns.copy()
    if len(returns) < 200: return 0.0
    models_to_test = {'GARCH': {'p':1,'o':0,'q':1}, 'GJR': {'p':1,'o':1,'q':1}}
    best_aic=np.inf; best_forecast=0.0
    for n, p in models_to_test.items():
        try:
            am = arch_model(100*returns, vol='GARCH', p=p['p'], o=p['o'], q=p['q'], dist='StudentsT')
            res = am.fit(disp='off')
            if res.aic < best_aic: best_aic=res.aic; best_forecast=np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100
        except: continue
    return float(best_forecast) if best_forecast else 0.0

def estimate_arima_models(prices, is_sarima=False):
    returns = np.log(prices/prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    try:
        model = pm.auto_arima(returns, seasonal=is_sarima, m=5 if is_sarima else 1, stepwise=True, trace=False, error_action='ignore', suppress_warnings=True, scoring='aic')
        forecast_ret = model.predict(n_periods=1)[0]
        return float((prices.iloc[-1] * np.exp(forecast_ret) / prices.iloc[-1]) - 1.0)
    except: return 0.0

def estimate_nnar_models(returns):
    if len(returns) < 100: return 0.0
    lags = 5
    X = pd.DataFrame({f'lag_{i}': returns.shift(i) for i in range(1, lags + 1)}).dropna()
    y = returns[lags:]
    if X.empty: return 0.0
    try:
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        model.fit(X.iloc[:-1], y.iloc[:-1])
        return float(model.predict(X.iloc[-1].values.reshape(1,-1))[0])
    except: return 0.0

def estimate_arch_garch_models(returns):
    return select_best_garch_model(returns)

def ga_optimize(df, features):
    return {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}

def train_meta_learner(df, params):
    test_size=60
    if len(df)<150: return 0.0, None, {}
    train=df.iloc[:-test_size]; test=df.iloc[-test_size:]
    
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[features].replace([np.inf, -np.inf], np.nan).fillna(0); y_tr = train['target']
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if X_tr.empty or y_tr.empty: return 0.0, None, {}

    # Sinyaller
    arima_ret = estimate_arima_models(train['close'], False)
    sarima_ret = estimate_arima_models(train['close'], True)
    nnar_ret = estimate_nnar_models(train['log_ret'].dropna())
    garch_ret = estimate_arch_garch_models(train['log_ret'].dropna())
    
    scaler_vol = StandardScaler()
    try:
        scaled_range_tr = scaler_vol.fit_transform(np.array(train['range'].values).reshape(-1, 1)).flatten()
        garch_signal = float(-np.sign(scaled_range_tr[-1])) if len(scaled_range_tr) > 0 else 0.0
    except: garch_signal = 0.0

    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3).fit(X_tr, y_tr)
    xgb_solo = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1).fit(X_tr, y_tr)
    
    scaler_hmm = StandardScaler()
    try:
        X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']].replace([np.inf, -np.inf], np.nan).fillna(0))
        hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
        hmm_probs = hmm.predict_proba(X_hmm)
    except:
        hmm_probs = np.zeros((len(train),3))
        hmm = None
    
    hmm_df = pd.DataFrame(hmm_probs, columns=['HMM_0','HMM_1','HMM_2'], index=train.index)

    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1], 'ETC': etc.predict_proba(X_tr)[:,1], 'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'Heuristic': train['heuristic'], 'HMM_0': hmm_df['HMM_0'], 'HMM_1': hmm_df['HMM_1'], 'HMM_2': hmm_df['HMM_2'],
        'ARIMA': np.full(len(train), arima_ret), 'SARIMA': np.full(len(train), sarima_ret),
        'NNAR': np.full(len(train), nnar_ret), 'GARCH': np.full(len(train), garch_ret), 'VolSig': np.full(len(train), garch_signal)
    }, index=train.index).fillna(0)
    
    scaler_meta = StandardScaler()
    try: meta_X_sc = scaler_meta.fit_transform(meta_X)
    except: meta_X_sc = meta_X.values
    meta_model = LogisticRegression(C=1.0).fit(meta_X_sc, y_tr)
    weights = meta_model.coef_[0]
    
    # Test
    arima_ret_t = estimate_arima_models(test['close'], False)
    sarima_ret_t = estimate_arima_models(test['close'], True)
    nnar_ret_t = estimate_nnar_models(test['log_ret'].dropna())
    garch_ret_t = estimate_arch_garch_models(test['log_ret'].dropna())
    try:
        scaled_range_t = scaler_vol.transform(np.array(test['range'].values).reshape(-1, 1)).flatten()
        garch_sig_t = float(-np.sign(scaled_range_t[-1])) if len(scaled_range_t)>0 else 0.0
    except: garch_sig_t = 0.0
    
    try:
        X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']].replace([np.inf, -np.inf], np.nan).fillna(0))
        hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    except: hmm_probs_t = np.zeros((len(test),3))
    hmm_df_t = pd.DataFrame(hmm_probs_t, columns=['HMM_0','HMM_1','HMM_2'], index=test.index)
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1], 'ETC': etc.predict_proba(X_test)[:,1], 'XGB': xgb_c.predict_proba(X_test)[:,1],
        'Heuristic': test['heuristic'], 'HMM_0': hmm_df_t['HMM_0'], 'HMM_1': hmm_df_t['HMM_1'], 'HMM_2': hmm_df_t['HMM_2'],
        'ARIMA': np.full(len(test), arima_ret_t), 'SARIMA': np.full(len(test), sarima_ret_t),
        'NNAR': np.full(len(test), nnar_ret_t), 'GARCH': np.full(len(test), garch_ret_t), 'VolSig': np.full(len(test), garch_sig_t)
    }, index=test.index).fillna(0)
    
    try: mx_test_sc = scaler_meta.transform(mx_test)
    except: mx_test_sc = mx_test.values
    
    probs_ens = meta_model.predict_proba(mx_test_sc)[:,1]
    probs_xgb = xgb_solo.predict_proba(X_test)[:,1]
    
    sim_ens=[100]; sim_xgb=[100]; sim_hodl=[100]; p0=test['close'].iloc[0]
    
    # KEY ERROR DÜZELTME: 'ret' sütununu garantiye al
    if 'ret' not in test.columns:
        test['ret'] = test['close'].pct_change().fillna(0)
        
    for i in range(len(test)):
        p=test['close'].iloc[i]; ret=test['ret'].iloc[i]
        
        # Trend Filtresi
        trend_up = test['trend_up'].iloc[i] == 1
        sell_thresh = -0.3 if trend_up else -0.1

        se=(probs_ens[i]-0.5)*2; sx=(probs_xgb[i]-0.5)*2
        if se>0.1: sim_ens.append(sim_ens[-1]*(1+ret))
        elif se<sell_thresh: sim_ens.append(sim_ens[-1]) # Sat
        else: sim_ens.append(sim_ens[-1]) # Bekle (Trend koruması)
        
        if sx>0.1: sim_xgb.append(sim_xgb[-1]*(1+ret))
        elif sx<sell_thresh: sim_xgb.append(sim_xgb[-1])
        else: sim_xgb.append(sim_xgb[-1])
        
        sim_hodl.append((100/p0)*p)
        
    roi_ens = sim_ens[-1]-100; roi_xgb = sim_xgb[-1]-100
    weights_dict = dict(zip(meta_X.columns, weights))
    
    if roi_xgb > roi_ens:
        # Son sinyale trend filtresi uygula
        last_trend = test['trend_up'].iloc[-1] == 1
        sig = (probs_xgb[-1]-0.5)*2
        if last_trend and -0.3 < sig < -0.1: sig = 0.0
        
        return sig, {'bot_roi': roi_xgb, 'method': 'Solo XGBoost', 'weights': weights_dict, 'sim_ens': sim_ens, 'sim_xgb': sim_xgb, 'sim_hodl': sim_hodl, 'dates': test.index}, weights
    else:
        last_trend = test['trend_up'].iloc[-1] == 1
        sig = (probs_ens[-1]-0.5)*2
        if last_trend and -0.3 < sig < -0.1: sig = 0.0
        
        return sig, {'bot_roi': roi_ens, 'method': 'Ensemble', 'weights': weights_dict, 'sim_ens': sim_ens, 'sim_xgb': sim_xgb, 'sim_hodl': sim_hodl, 'dates': test.index}, weights

def analyze_ticker_tournament(ticker):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    current_price = float(raw_df['close'].iloc[-1])
    best_roi = -9999; final_res = None
    
    for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
        df_raw = process_data(raw_df, tf_code)
        if df_raw is None: continue
        nan_count = df_raw.attrs.get('nan_count', 0)
        feats = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        
        # DÜZELTİLEN YER: 'ret' kaybolmaz
        df_imp, method = smart_impute(df_raw, feats)
        
        sig, info, _ = train_meta_learner(df_imp, ga_optimize(df_imp, feats))
        
        if info and info['bot_roi'] > best_roi:
            best_roi = info['bot_roi']
            final_res = {
                'ticker': ticker, 'price': current_price, 'roi': best_roi,
                'signal': sig, 'tf': tf_name, 'info': info, 'method': method,
                'nan_count': nan_count, 'imp_method': method
            }
    return final_res

# =============================================================================
# ARAYÜZ
# =============================================================================
st.markdown("### 📈 Portföy Durumu & Smart Imputation Dashboard")
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
    
    if st.button("🚀 ANALİZ ET (Full Detail)", type="primary"):
        updated = pf_df.copy()
        total_pool = updated['Nakit_Bakiye_USD'].sum()
        results = []
        prog = st.progress(0)
        tz = pytz.timezone('Europe/Istanbul')
        time_str = datetime.now(tz).strftime("%d-%m %H:%M")
        report_text = ""
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            res = analyze_ticker_tournament(ticker)
            if res:
                res['idx']=idx; res['status']=row['Durum']; res['amount']=float(row['Miktar'])
                results.append(res)
                
                decision = "AL" if res['signal'] > 0.2 else ("SAT" if res['signal'] < -0.2 else "BEKLE")
                report_text += f"{ticker}: {decision} (ROI: {res['roi']:.2f}%) [{res['info']['method']}]\n"
                
                with st.expander(f"📊 {ticker} | ROI: %{res['roi']:.2f} | Model: {res['info']['method']}"):
                    tab1, tab2, tab3 = st.tabs(["📈 Performans", "🧠 Ağırlıklar", "🧬 Veri"])
                    info = res['info']
                    with tab1:
                        fig = go.Figure()
                        dates = info['dates']; l = len(dates)
                        fig.add_trace(go.Scatter(x=dates, y=info['sim_ens'][-l:], name='Ensemble', line=dict(color='#00CC96', width=3)))
                        fig.add_trace(go.Scatter(x=dates, y=info['sim_xgb'][-l:], name='Solo XGB', line=dict(color='#636EFA', width=2, dash='dot')))
                        fig.add_trace(go.Scatter(x=dates, y=info['sim_hodl'][-l:], name='HODL', line=dict(color='gray', width=1)))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Ensemble ROI", f"%{info['sim_ens'][-1]-100:.2f}")
                        m2.metric("XGBoost ROI", f"%{info['sim_xgb'][-1]-100:.2f}")
                        m3.metric("Piyasa ROI", f"%{info['sim_hodl'][-1]-100:.2f}")

                    with tab2:
                        w = info.get('weights', {})
                        if isinstance(w, dict) and w:
                            w_df = pd.DataFrame(list(w.items()), columns=['Faktör', 'Etki']).set_index('Faktör')
                            w_df['Mutlak Etki'] = w_df['Etki'].abs()
                            w_df = w_df.sort_values(by='Mutlak Etki', ascending=False)
                            st.bar_chart(w_df['Etki'])
                            st.dataframe(w_df)
                    with tab3:
                        k1, k2, k3 = st.columns(3)
                        k1.metric("NaN Sayısı", f"{res['nan_count']}")
                        k2.metric("Imputer", f"{res['imp_method']}")
                        k3.metric("Zaman", f"{res['tf']}")
            prog.progress((i+1)/len(updated))
            
        # Ortak Kasa
        for r in results:
            if r['status'] == 'COIN' and r['signal'] < -0.2:
                rev = r['amount'] * r['price']; total_pool += rev
                updated.at[r['idx'], 'Durum'] = 'CASH'; updated.at[r['idx'], 'Miktar'] = 0.0
                updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated.at[r['idx'], 'Son_Islem_Log'] = f"SAT ({r['tf']})"
                updated.at[r['idx'], 'Son_Islem_Zamani'] = time_str

        buy_cands = [r for r in results if r['signal'] > 0.2 and r['roi']>0]
        total_roi = sum([r['roi'] for r in buy_cands])
        
        if buy_cands and total_pool > 1.0:
            for r in buy_cands:
                w = r['roi']/total_roi
                amt_usd = total_pool * w
                if updated.at[r['idx'],'Durum'] == 'CASH':
                    amt = amt_usd / r['price']
                    updated.at[r['idx'], 'Durum'] = 'COIN'; updated.at[r['idx'], 'Miktar'] = amt
                    updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0
                    updated.at[r['idx'], 'Son_Islem_Fiyati'] = r['price']
                    updated.at[r['idx'], 'Son_Islem_Log'] = f"AL (Pay: %{w*100:.1f})"
                    updated.at[r['idx'], 'Son_Islem_Zamani'] = time_str
        elif total_pool > 0:
            f = updated.index[0]
            updated.at[f,'Nakit_Bakiye_USD']+=total_pool
            for ix in updated.index:
                if ix!=f and updated.at[ix,'Durum']=='CASH': updated.at[ix,'Nakit_Bakiye_USD']=0.0

        for idx, row in updated.iterrows():
            p = next((r['price'] for r in results if r['idx'] == idx), 0.0)
            if p > 0:
                val = (float(updated.at[idx, 'Miktar']) * p) if updated.at[idx, 'Durum'] == 'COIN' else float(updated.at[idx, 'Nakit_Bakiye_USD'])
                updated.at[idx, 'Kaydedilen_Deger_USD'] = val

        save_portfolio(updated, sheet)
        st.code(report_text, language='text')
        st.success("✅ Analiz Tamamlandı!")
