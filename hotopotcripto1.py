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
from statsmodels.stats.diagnostic import acorr_ljungbox # Ljung-Box testi için
import pmdarima as pm
from arch import arch_model
from scipy.stats import boxcox, yeojohnson # Dönüşümler için

# --- AI & ML Kütüphaneleri ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Gerekli uyarıları yoksay
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Canavar Motor", layout="wide")
st.title("🏦 Hedge Fund AI: Canavar Motor")

# =============================================================================
# 1. AYARLAR VE SABİTLER
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

with st.sidebar:
    st.header("⚙️ Ayarlar")
    use_ga = st.checkbox("Genetic Algoritma (GA) Optimizasyonu", value=True)
    ga_gens = st.number_input("GA Döngüsü", 1, 20, 5)
    st.info("Sistem, en yüksek Alpha'yı üreten zaman dilimini (Günlük/Haftalık/Aylık) seçer.")

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
# 3. AI MOTORU - VERİ İŞLEME VE ÖZELLİK MÜHENDİSLİĞİ
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
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    s3 = np.where(df['close'] > df['close'].rolling(150).mean(), 1, -1)
    vol = df['close'].pct_change().rolling(20).std()
    s4 = np.where(vol < vol.shift(1), 1, -1)
    s5 = np.sign(df['close'].diff(10).fillna(0))
    momentum = np.sign(df['close'].diff(20).fillna(0))
    return (s1 + s2 + s3 + s4 + s5 + momentum) / 6.0

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
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
    elif timeframe == 'M': df_res = df.resample('ME').agg(agg).dropna()
    else: df_res = df.copy()
    if len(df_res) < 100: return None

    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change()

    df_res['avg_ret_5m'] = df_res['ret'].rolling(window=100).mean() * 100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(window=750).mean() * 100

    df_res['day_of_week'] = df_res.index.dayofweek
    day_returns = df_res.groupby('day_of_week')['ret'].mean().fillna(0)
    df_res['day_score'] = df_res['day_of_week'].map(day_returns).fillna(0)
    
    avg_feats = df_res[['avg_ret_5m', 'avg_ret_3y', 'day_score']].fillna(0)
    if not avg_feats.empty:
        scaler_avg = StandardScaler()
        df_res['historical_avg_score'] = scaler_avg.fit_transform(avg_feats).mean(axis=1)
    else:
        df_res['historical_avg_score'] = 0.0

    df_res['range_vol_delta'] = df_res['range'].pct_change(5).fillna(0)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

# =============================================================================
# 4. AI MOTORU - MODEL EĞİTİMİ VE ENSEMBLE
# =============================================================================

def select_best_garch_model(returns):
    returns = returns.copy()
    if len(returns) < 200: return 0.0

    models_to_test = {
        'GARCH(1,1)': {'vol': 'GARCH', 'p': 1, 'o': 0, 'q': 1},
        'GJR-GARCH(1,1)': {'vol': 'GARCH', 'p': 1, 'o': 1, 'q': 1},
        'APARCH(1,1)': {'vol': 'APARCH', 'p': 1, 'o': 1, 'q': 1}
    }
    
    best_aic = np.inf; best_forecast = 0.0
    
    for name, params in models_to_test.items():
        try:
            am = arch_model(100 * returns, vol=params['vol'], p=params['p'], o=params['o'], q=params['q'], dist='StudentsT')
            res = am.fit(disp='off')
            
            # Kalıntı Analizi (Ljung-Box)
            lb_p = acorr_ljungbox(res.resid**2, lags=[10], return_df=True)['lb_pvalue'].iloc[-1]
            
            if res.aic < best_aic and lb_p > 0.05:
                best_aic = res.aic
                forecast = res.forecast(horizon=1)
                best_forecast = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
        except: continue
            
    return float(best_forecast) if best_forecast else 0.0

def estimate_arch_garch_models(returns):
    return select_best_garch_model(returns)

def estimate_arima_models(prices, is_sarima=False):
    returns = np.log(prices / prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    
    try:
        model = pm.auto_arima(returns, seasonal=is_sarima, m=5 if is_sarima else 1, stepwise=True, 
                              suppress_warnings=True, trace=False, error_action='ignore',
                              power_transform=True, d=None, D=None, scoring='aic')
        
        lb_p = acorr_ljungbox(model.resid(), lags=[10], return_df=True)['lb_pvalue'].iloc[-1]
        if lb_p < 0.05: return 0.0
        
        forecast_ret = model.predict(n_periods=1)[0]
        last_price = prices.iloc[-1]
        forecast_price = last_price * np.exp(forecast_ret)
        return float((forecast_price / last_price) - 1.0)
    except: return 0.0

def estimate_nnar_models(returns):
    if len(returns) < 100: return 0.0
    lags = 5
    X = pd.DataFrame({f'lag_{i}': returns.shift(i) for i in range(1, lags + 1)}).dropna()
    y = returns[lags:]
    if X.empty or len(X) < 10: return 0.0
    
    try:
        X_train = X.iloc[:-1]; y_train = y.iloc[:-1]
        X_forecast = X.iloc[-1].values.reshape(1, -1)
        nnar_model = MLPRegressor(hidden_layer_sizes=(10, ), max_iter=100, solver='lbfgs', random_state=42)
        nnar_model.fit(X_train, y_train)
        return float(nnar_model.predict(X_forecast)[0])
    except: return 0.0

def ga_optimize(df, n_gen=5):
    best_depth = 5; best_nest = 50; best_score = -999
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    for d in [3, 5, 7, 9]:
        for n in [20, 50, 100]:
            train = df.iloc[:-30]; test = df.iloc[-30:]
            current_features = [f for f in features if f in train.columns]
            if not current_features or train.empty: continue

            rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42).fit(train[current_features], train['target'])
            score_rf = rf.score(test[current_features], test['target'])
            
            etc = ExtraTreesClassifier(n_estimators=n, max_depth=d, random_state=42).fit(train[current_features], train['target'])
            score_etc = etc.score(test[current_features], test['target'])

            score = max(score_rf, score_etc)
            if score > best_score:
                best_score = score; best_depth = d; best_nest = n
    return {'rf_depth': best_depth, 'rf_nest': best_nest, 'xgb_params': {'max_depth':5, 'n_estimators':100}}

def train_meta_learner(df, params=None):
    rf_d = params['rf_depth'] if params else 5
    rf_n = params['rf_nest'] if params else 50
    xgb_d = params['xgb_params']['max_depth']
    xgb_n = params['xgb_params']['n_estimators']
    test_size = 60
    
    if len(df) < test_size + 50: return 0.0, None
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    
    base_features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[base_features]; y_tr = train['target']
    X_test = test[base_features]

    if X_tr.empty or y_tr.empty: return 0.0, None

    arima_getiri = estimate_arima_models(train['close'], is_sarima=False)
    sarima_getiri = estimate_arima_models(train['close'], is_sarima=True)
    nnar_getiri = estimate_nnar_models(train['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    garch_score_tr = estimate_arch_garch_models(train['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    
    scaler_vol = StandardScaler()
    scaled_range_tr = scaler_vol.fit_transform(np.array(train['range'].values).reshape(-1, 1)).flatten()
    garch_signal = float(-np.sign(scaled_range_tr[-1])) if len(scaled_range_tr) > 0 else 0.0 

    rf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d, random_state=42).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=rf_n, max_depth=rf_d, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=xgb_n, max_depth=xgb_d).fit(X_tr, y_tr)
    
    scaler_hmm = StandardScaler()
    X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None
    
    hmm_prob_df = pd.DataFrame(np.zeros((len(train), 3)), columns=[f'HMM_R{i}' for i in range(3)], index=train.index)
    if hmm:
        pr = hmm.predict_proba(X_hmm)
        hmm_prob_df = pd.DataFrame(pr, columns=[f'HMM_R{i}' for i in range(pr.shape[1])], index=train.index)
        
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'ETC': etc.predict_proba(X_tr)[:,1],
        'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'Heuristic': train['heuristic'].values,
        'Historical_Avg_Score': train['historical_avg_score'].values, 
        'ARIMA_Return': np.full(len(train), arima_getiri, dtype=np.float64), 
        'SARIMA_Return': np.full(len(train), sarima_getiri, dtype=np.float64), 
        'NNAR_Return': np.full(len(train), nnar_getiri, dtype=np.float64), 
        'GARCH_Volatility': np.full(len(train), garch_score_tr, dtype=np.float64), 
        'Vol_Signal': np.full(len(train), garch_signal, dtype=np.float64) 
    })
    
    meta_X = pd.concat([meta_X, hmm_prob_df.iloc[:len(meta_X)]], axis=1)

    # --- GÜVENLİK VE TEMİZLİK KATMANI ---
    meta_X = meta_X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta_features_to_scale = [
        'Heuristic', 'Historical_Avg_Score', 'ARIMA_Return', 'SARIMA_Return', 
        'NNAR_Return', 'GARCH_Volatility', 'Vol_Signal', 'HMM_R0', 'HMM_R1', 'HMM_R2'
    ]
    
    # Scaler hatasını önlemek için kontrol
    try:
        scaler_meta = StandardScaler()
        meta_X[meta_features_to_scale] = scaler_meta.fit_transform(meta_X[meta_features_to_scale])
    except ValueError: pass # Boş veya sabit veri gelirse scale etme

    # Tekrar Temizlik (Scaler sonrası NaN oluşabilir)
    meta_X = meta_X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta_model = LogisticRegression(C=0.1, solver='liblinear', penalty='l2').fit(meta_X, y_tr)
    weights = meta_model.coef_[0]
    
    # --- TEST VERİSİ ---
    arima_getiri_test = estimate_arima_models(test['close'], is_sarima=False)
    sarima_getiri_test = estimate_arima_models(test['close'], is_sarima=True)
    nnar_getiri_test = estimate_nnar_models(test['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    garch_score_test = estimate_arch_garch_models(test['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    scaled_range_test = scaler_vol.transform(np.array(test['range'].values).reshape(-1, 1)).flatten()
    garch_signal_test = float(-np.sign(scaled_range_test[-1])) if len(scaled_range_test) > 0 else 0.0
    
    X_hmm_t = scaler_hmm.transform(test[['log_ret','range_vol_delta']])
    hmm_p_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_prob_df_test = pd.DataFrame(hmm_p_t, columns=[f'HMM_R{i}' for i in range(hmm_p_t.shape[1])], index=test.index)
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'ETC': etc.predict_proba(X_test)[:,1],
        'XGB': xgb_c.predict_proba(X_test)[:,1],
        'Heuristic': test['heuristic'].values,
        'Historical_Avg_Score': test['historical_avg_score'].values,
        'ARIMA_Return': np.full(len(test), arima_getiri_test, dtype=np.float64), 
        'SARIMA_Return': np.full(len(test), sarima_getiri_test, dtype=np.float64),
        'NNAR_Return': np.full(len(test), nnar_getiri_test, dtype=np.float64), 
        'GARCH_Volatility': np.full(len(test), garch_score_test, dtype=np.float64),
        'Vol_Signal': np.full(len(test), garch_signal_test, dtype=np.float64)
    })
    
    mx_test = pd.concat([mx_test, hmm_prob_df_test.iloc[:len(mx_test)]], axis=1)
    
    # Test verisi temizliği
    mx_test = mx_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    try:
        mx_test[meta_features_to_scale] = scaler_meta.transform(mx_test[meta_features_to_scale])
    except ValueError: pass
    mx_test = mx_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    probs = meta_model.predict_proba(mx_test)[:,1] 
    
    for i in range(len(test)):
        p = test['close'].iloc[i]; s=(probs[i]-0.5)*2
        if s>0.25 and cash>0: coin=cash/p; cash=0
        elif s<-0.25 and coin>0: cash=coin*p; coin=0
        sim_eq.append(cash+coin*p); hodl_eq.append((100/p0)*p)
        
    final_signal=(probs[-1]-0.5)*2
    
    weights_names = [
        'RandomForest', 'ExtraTrees', 'XGBoost', 
        'Senin Kuralın (Heuristic)', 'Tarihsel Ortalamalar',
        'ARIMA Getiri Tahmini', 'SARIMA Getiri Tahmini', 
        'NNAR Getiri Tahmini', 'GARCH Oynaklık Skoru', 
        'Oynaklık Sinyali', 
        'HMM Rejim 0', 'HMM Rejim 1', 'HMM Rejim 2'
    ]
    
    info={'weights': weights, 'weights_names': weights_names, 'bot_eq': sim_eq[1:],'hodl_eq': hodl_eq[1:],'dates': test.index,'alpha': float(sim_eq[-1]-hodl_eq[-1]),'bot_roi': float(sim_eq[-1]-100),'hodl_roi': float(hodl_eq[-1]-100),'conf': probs[-1],'my_score': test['heuristic'].iloc[-1]}
    
    return final_signal, info

def analyze_ticker_tournament(ticker, status_placeholder):
    raw_df = get_raw_data(ticker)
    if raw_df is None: 
        status_placeholder.error("Veri Yok")
        return "HATA", 0.0, "YOK", None
    current_price = float(raw_df['close'].iloc[-1])
    timeframes={'GÜNLÜK':'D','HAFTALIK':'W','AYLIK':'M'}
    best_alpha=-9999; final_decision="BEKLE"; winning_tf="YOK"; best_info=None
    
    for tf_name, tf_code in timeframes.items():
        status_placeholder.text(f"Turnuva: {tf_name} grafiği test ediliyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        params = ga_optimize(df) if st.session_state.get('use_ga',True) else None
        sig, info = train_meta_learner(df, params)
        if info is None: continue
        if info['alpha']>best_alpha:
            best_alpha=info['alpha']; winning_tf=tf_name; best_info=info
            if sig>0.25: final_decision="AL"
            elif sig<-0.25: final_decision="SAT"
            else: final_decision="BEKLE"
    return final_decision, current_price, winning_tf, best_info

if st.button("🚀 PORTFÖYÜ CANLI ANALİZ ET", type="primary"):
    st.session_state['use_ga'] = use_ga
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    pf_df, sheet = load_and_fix_portfolio()
    
    if pf_df.empty:
        st.error("Hata: Portföy yüklenemedi.")
    else:
        updated = pf_df.copy(); prog = st.progress(0); sim_summary=[]
        for i,(idx,row) in enumerate(updated.iterrows()):
            ticker=row['Ticker']
            if len(str(ticker))<3: continue
            with st.expander(f"🧠 {ticker} Analiz Raporu", expanded=True):
                ph = st.empty()
                dec, prc, tf, info = analyze_ticker_tournament(ticker, ph)
                if dec!="HATA" and info:
                    sim_summary.append({"Coin":ticker,"Kazanan TF":tf,"Bot ROI":info['bot_roi'],"HODL ROI":info['hodl_roi'],"Alpha":info['alpha']})
                    w=info['weights']; w_names=info['weights_names']
                    w_abs=np.abs(w); w_norm=w_abs/(np.sum(w_abs)+1e-9)*100
                    w_df=pd.DataFrame({'Faktör':w_names,'Etki (%)':w_norm})
                    w_df=w_df.sort_values(by='Etki (%)', ascending=False)
                    c1,c2=st.columns([1,2])
                    with c1:
                        st.markdown(f"### Karar: **{dec}**"); st.caption(f"Seçilen Zaman Dilimi: {tf}"); st.markdown(f"**Senin Puanın:** {info['my_score']:.2f}"); st.markdown("**Model Etki Dağılımı:**")
                        st.dataframe(w_df, hide_index=True)
                    with c2:
                        fig=go.Figure(); fig.add_trace(go.Scatter(x=info['dates'],y=info['bot_eq'],name="Bot",line=dict(color='green',width=2)))
                        fig.add_trace(go.Scatter(x=info['dates'],y=info['hodl_eq'],name="HODL",line=dict(color='gray',dash='dot')))
                        color_ti="green" if info['alpha']>0 else "red"
                        fig.update_layout(title=f"Kazanan Strateji ({tf}) Alpha: ${info['alpha']:.2f}",title_font_color=color_ti,height=250,template="plotly_dark",margin=dict(t=30,b=0,l=0,r=0))
                        st.plotly_chart(fig, use_container_width=True)
                    
                    stt=row['Durum']
                    if stt=='COIN' and dec=='SAT':
                        amt=float(row['Miktar'])
                        if amt>0: updated.at[idx,'Durum']='CASH'; updated.at[idx,'Nakit_Bakiye_USD']=amt*prc; updated.at[idx,'Miktar']=0.0; updated.at[idx,'Son_Islem_Fiyati']=prc; updated.at[idx,'Son_Islem_Log']=f"SAT ({tf}) A:{info['alpha']:.1f}"; updated.at[idx,'Son_Islem_Zamani']=time_str
                    elif stt=='CASH' and dec=='AL':
                        cash=float(row['Nakit_Bakiye_USD'])
                        if cash>1: updated.at[idx,'Durum']='COIN'; updated.at[idx,'Miktar']=cash/prc; updated.at[idx,'Nakit_Bakiye_USD']=0.0; updated.at[idx,'Son_Islem_Fiyati']=prc; updated.at[idx,'Son_Islem_Log']=f"AL ({tf}) A:{info['alpha']:.1f}"; updated.at[idx,'Son_Islem_Zamani']=time_str
                    val=(float(updated.at[idx,'Miktar'])*prc) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])
                    updated.at[idx,'Kaydedilen_Deger_USD']=val
                    ph.success(f"Analiz Bitti. En iyi grafik: {tf}")
            prog.progress((i+1)/len(updated))
        save_portfolio(updated, sheet)
        st.divider(); st.subheader("🏆 Turnuva Sonuçları & Performans")
        if sim_summary:
            sum_df=pd.DataFrame(sim_summary)
            col1,col2,col3=st.columns(3)
            col1.metric("Ort. Bot Getirisi", f"%{sum_df['Bot ROI'].mean():.2f}")
            col2.metric("Ort. HODL Getirisi", f"%{sum_df['HODL ROI'].mean():.2f}")
            col3.metric("TOPLAM ALPHA", f"%{sum_df['Alpha'].mean():.2f}", delta_color="normal")
            st.dataframe(sum_df.style.format("{:.2f}", subset=["Bot ROI","HODL ROI","Alpha"]))
        st.success("✅ Canavar Motor Tamamlandı!")
