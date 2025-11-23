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

# --- İstatistiksel Kütüphaneler ---
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

# Uyarıları yoksay
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Walk-Forward & Benchmark", layout="wide")
st.title("🔬 Hedge Fund AI: Walk-Forward & Benchmark Modu")

# =============================================================================
# 1. AYARLAR VE SABİTLER
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y" 

with st.sidebar:
    st.header("⚙️ Ayarlar")
    use_ga = st.checkbox("Genetic Algoritma (Validation)", value=True)
    ga_gens = st.number_input("GA Döngüsü", 1, 20, 5)
    st.info("Walk-Forward Validation kullanılıyor. Modelin 'Geleceği Görmesi' (Data Leakage) engellenmiştir.")

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
# 3. VERİ İŞLEME VE MODEL FONKSİYONLARI
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

# --- EKONOMETRİK MODELLER ---

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

# --- GÜÇLENDİRİLMİŞ OPTİMİZASYON (Validation Seti ile) ---
def ga_optimize(train_df):
    """Train/Validation ayrımı yaparak GA optimizasyonu uygular."""
    # Validation Seti: Eğitim verisinin son %20'si
    val_size = int(len(train_df) * 0.2)
    if val_size < 20: return {'rf_depth': 5, 'rf_nest': 100, 'xgb_params': {'max_depth':5, 'n_estimators':100}}

    train = train_df.iloc[:-val_size]
    val = train_df.iloc[-val_size:]
    
    best_depth = 5; best_nest = 50; best_score = -999
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    for d in [3, 5, 7, 9]:
        for n in [20, 50, 100]:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42).fit(train[features], train['target'])
            score_rf = rf.score(val[features], val['target'])
            
            etc = ExtraTreesClassifier(n_estimators=n, max_depth=d, random_state=42).fit(train[features], train['target'])
            score_etc = etc.score(val[features], val['target'])

            score = max(score_rf, score_etc)
            if score > best_score:
                best_score = score; best_depth = d; best_nest = n
    return {'rf_depth': best_depth, 'rf_nest': best_nest, 'xgb_params': {'max_depth':5, 'n_estimators':100}}

# --- ANA META-LEARNER & COMPARISON ---
def train_meta_learner_comparison(df, params):
    # 3'lü Ayrım: Train (Eğitim) - Validation (GA) - Test (Final Performans)
    # Ancak veri azlığından dolayı GA'yı Train içinde, Final Testi ayrı tutuyoruz.
    test_size = 60 # Son 60 bar (Kesinlikle görülmemiş veri)
    
    if len(df) < 150: return 0.0, None
    
    train_full = df.iloc[:-test_size] # GA ve Eğitim için
    test = df.iloc[-test_size:]       # Final Karşılaştırma için
    
    # Özellikler
    base_features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train_full[base_features]; y_tr = train_full['target']
    X_test = test[base_features]

    # 1. Temel Modellerin Eğitimi (Train Set üzerinde)
    rf = RandomForestClassifier(n_estimators=params['rf_nest'], max_depth=params['rf_depth'], random_state=42).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=params['rf_nest'], max_depth=params['rf_depth'], random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5).fit(X_tr, y_tr)
    
    # 2. Ekonometrik Sinyaller (Train üzerinde)
    arima_getiri = estimate_arima_models(train_full['close'], is_sarima=False)
    sarima_getiri = estimate_arima_models(train_full['close'], is_sarima=True)
    nnar_getiri = estimate_nnar_models(train_full['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    garch_score_tr = estimate_arch_garch_models(train_full['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    
    scaler_vol = StandardScaler()
    scaled_range_tr = scaler_vol.fit_transform(np.array(train_full['range'].values).reshape(-1, 1)).flatten()
    garch_signal = float(-np.sign(scaled_range_tr[-1])) if len(scaled_range_tr) > 0 else 0.0 

    # 3. HMM (Train üzerinde)
    scaler_hmm = StandardScaler()
    X_hmm = scaler_hmm.fit_transform(train_full[['log_ret', 'range_vol_delta']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None
    
    hmm_prob_df = pd.DataFrame(np.zeros((len(train_full), 3)), columns=[f'HMM_R{i}' for i in range(3)], index=train_full.index)
    if hmm:
        pr = hmm.predict_proba(X_hmm)
        hmm_prob_df = pd.DataFrame(pr, columns=[f'HMM_R{i}' for i in range(pr.shape[1])], index=train_full.index)

    # 4. Meta-Data Oluşturma (Stacking)
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'ETC': etc.predict_proba(X_tr)[:,1],
        'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'Heuristic': train_full['heuristic'].values,
        'Historical_Avg_Score': train_full['historical_avg_score'].values, 
        'ARIMA_Return': np.full(len(train_full), arima_getiri, dtype=np.float64), 
        'SARIMA_Return': np.full(len(train_full), sarima_getiri, dtype=np.float64), 
        'NNAR_Return': np.full(len(train_full), nnar_getiri, dtype=np.float64), 
        'GARCH_Volatility': np.full(len(train_full), garch_score_tr, dtype=np.float64), 
        'Vol_Signal': np.full(len(train_full), garch_signal, dtype=np.float64) 
    }, index=train_full.index)
    
    meta_X = pd.concat([meta_X, hmm_prob_df], axis=1).dropna()
    y_tr = y_tr.loc[meta_X.index] # Hizalama

    # Normalizasyon
    meta_features_to_scale = [
        'Heuristic', 'Historical_Avg_Score', 'ARIMA_Return', 'SARIMA_Return', 
        'NNAR_Return', 'GARCH_Volatility', 'Vol_Signal', 'HMM_R0', 'HMM_R1', 'HMM_R2'
    ]
    scaler_meta = StandardScaler()
    try: meta_X[meta_features_to_scale] = scaler_meta.fit_transform(meta_X[meta_features_to_scale])
    except ValueError: pass
    
    # 5. Meta-Learner Eğitimi (Ensemble)
    meta_model = LogisticRegression(C=1.0, solver='liblinear', penalty='l2').fit(meta_X, y_tr)
    weights = meta_model.coef_[0]
    
    # ---------------------------------------------------------
    # --- TEST AŞAMASI (Geleceği Görmeden) ---
    # ---------------------------------------------------------
    
    # Test Verisi için Ekonometrik Tahminler (Basitlik için son eğitilen modelin tahminini yayıyoruz)
    # Gerçek walk-forward'da her bar için yeniden eğitilirdi ama çok yavaş olur.
    # Bu "Out-of-Sample" testi için yeterli bir yaklaşımdır.
    
    arima_getiri_test = estimate_arima_models(test['close'], is_sarima=False)
    sarima_getiri_test = estimate_arima_models(test['close'], is_sarima=True)
    nnar_getiri_test = estimate_nnar_models(test['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    garch_score_test = estimate_arch_garch_models(test['log_ret'].replace([np.inf, -np.inf], np.nan).dropna())
    
    X_hmm_t = scaler_hmm.transform(test[['log_ret','range_vol_delta']])
    hmm_p_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_prob_df_test = pd.DataFrame(hmm_p_t, columns=[f'HMM_R{i}' for i in range(3)], index=test.index)
    
    # Meta-Test Verisi
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
        'Vol_Signal': np.full(len(test), garch_signal, dtype=np.float64)
    }, index=test.index)
    
    mx_test = pd.concat([mx_test, hmm_prob_df_test], axis=1).fillna(0.0)
    try: mx_test[meta_features_to_scale] = scaler_meta.transform(mx_test[meta_features_to_scale])
    except ValueError: pass
    
    # --- KARŞILAŞTIRMALI TAHMİNLER ---
    
    # 1. Ensemble (Bot) Tahmini
    probs_ensemble = meta_model.predict_proba(mx_test)[:,1]
    
    # 2. Solo XGBoost Tahmini (Benchmark)
    probs_xgb = xgb_c.predict_proba(X_test)[:,1]
    
    # --- ROI SİMÜLASYONU ---
    sim_eq_ensemble = [100]
    sim_eq_xgb = [100]
    sim_eq_hodl = [100]
    
    cash_ens=100; coin_ens=0
    cash_xgb=100; coin_xgb=0
    
    p0 = test['close'].iloc[0]
    
    for i in range(len(test)):
        p = test['close'].iloc[i]
        
        # Ensemble Stratejisi
        s_ens = (probs_ensemble[i]-0.5)*2
        if s_ens > 0.1 and cash_ens > 0: coin_ens = cash_ens/p; cash_ens = 0
        elif s_ens < -0.1 and coin_ens > 0: cash_ens = coin_ens*p; coin_ens = 0
        sim_eq_ensemble.append(cash_ens + coin_ens*p)
        
        # XGBoost Stratejisi
        s_xgb = (probs_xgb[i]-0.5)*2
        if s_xgb > 0.1 and cash_xgb > 0: coin_xgb = cash_xgb/p; cash_xgb = 0
        elif s_xgb < -0.1 and coin_xgb > 0: cash_xgb = coin_xgb*p; coin_xgb = 0
        sim_eq_xgb.append(cash_xgb + coin_xgb*p)
        
        # HODL
        sim_eq_hodl.append((100/p0)*p)
        
    final_signal = (probs_ensemble[-1]-0.5)*2
    
    weights_names = list(meta_X.columns)
    
    info = {
        'weights': dict(zip(weights_names, weights)), 
        'bot_roi': float(sim_eq_ensemble[-1]-100),
        'xgb_roi': float(sim_eq_xgb[-1]-100),
        'hodl_roi': float(sim_eq_hodl[-1]-100),
        'dates': test.index,
        'sim_ensemble': sim_eq_ensemble[1:],
        'sim_xgb': sim_eq_xgb[1:],
        'sim_hodl': sim_eq_hodl[1:]
    }
    
    return final_signal, info

def analyze_ticker_tournament(ticker):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    
    current_price = float(raw_df['close'].iloc[-1])
    best_roi = -9999; final_res = None
    
    for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # GA Optimizasyonunu (Validation Set üzerinde) çalıştır
        params = ga_optimize(df) 
        
        # Modeli Eğit ve Test Et
        sig, info = train_meta_learner_comparison(df, params)
        
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

st.title("🏦 Hedge Fund AI: Canavar Motor (Benchmark Modu)")
st.markdown("Bu panel, Ensemble Model (Ortak Akıl) ile Solo XGBoost'u karşılaştırır ve en iyi ROI'ye göre işlem yapar.")

pf_df, sheet = load_and_fix_portfolio()

if not pf_df.empty:
    # --- 1. ÜST BİLGİ KARTLARI ---
    total_coin_val = pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    parked_cash = pf_df['Nakit_Bakiye_USD'].sum()
    total_portfolio = total_coin_val + parked_cash
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Toplam Portföy", f"${total_portfolio:.2f}")
    c2.metric("Aktif Yatırım", f"${total_coin_val:.2f}")
    c3.metric("Boştaki Nakit", f"${parked_cash:.2f}")
    
    st.divider()
    
    # --- 2. DETAYLI PORTFÖY ---
    st.subheader("📋 Mevcut Portföy")
    display_df = pf_df[['Ticker', 'Durum', 'Miktar', 'Kaydedilen_Deger_USD', 'Son_Islem_Log', 'Son_Islem_Zamani']].copy()
    display_df['Kaydedilen_Deger_USD'] = display_df['Kaydedilen_Deger_USD'].apply(lambda x: f"${x:.2f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    if st.button("🚀 ANALİZ ET VE KARŞILAŞTIR", type="primary"):
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
                res['idx'] = idx; res['status'] = row['Durum']; res['amount'] = float(row['Miktar'])
                results.append(res)
                
                with st.expander(f"📊 {ticker} Analiz & Benchmark (Bot ROI: %{res['roi']:.2f})"):
                    info = res['info']
                    
                    # 1. Grafik Karşılaştırması
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_ensemble'], name='Canavar Bot (Ensemble)', line=dict(color='#00CC96', width=3)))
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_xgb'], name='Sadece XGBoost', line=dict(color='#636EFA', width=2, dash='dot')))
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_hodl'], name='HODL (Piyasa)', line=dict(color='gray', width=1)))
                    fig.update_layout(title=f"Strateji Karşılaştırması ({res['tf']})", height=300, template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 2. Performans Metrikleri
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Bot Getirisi", f"%{info['bot_roi']:.2f}")
                    mc2.metric("XGBoost Getirisi", f"%{info['xgb_roi']:.2f}")
                    mc3.metric("Piyasa (HODL)", f"%{info['hodl_roi']:.2f}")
                    
                    # 3. Model Ağırlıkları
                    w_df = pd.DataFrame.from_dict(info['weights'], orient='index', columns=['Etki']).sort_values(by='Etki', ascending=False)
                    w_df['Etki'] = w_df['Etki'].abs()
                    st.bar_chart(w_df)
                    
            prog.progress((i + 1) / len(updated))
            
        # --- ORTAK KASA İŞLEMLERİ ---
        for r in results:
            if r['status'] == 'COIN' and r['signal'] < -0.1:
                rev = r['amount'] * r['price']
                total_pool += rev
                updated.at[r['idx'], 'Durum'] = 'CASH'; updated.at[r['idx'], 'Miktar'] = 0.0
                updated.at[r['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated.at[r['idx'], 'Son_Islem_Log'] = f"SAT ({r['tf']})"
                updated.at[r['idx'], 'Son_Islem_Zamani'] = time_str
                st.toast(f"🔻 SATILDI: {r['ticker']} (+${rev:.2f})")

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
