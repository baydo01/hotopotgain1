# trader_bot_full_engine_v4.py
# Hedge Fund AI — BEAST MODE v4
# v3 engine + meta-learner augmentation with PCA, ARIMA, ARCH, GBM (Earth removed)

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time, warnings, gspread, os, json, random, pickle, logging, multiprocessing
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz
from functools import lru_cache, partial
from typing import Dict, Any, Optional, List, Tuple

# ML models
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from sklearn.decomposition import PCA
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

# GA
from deap import base, creator, tools, algorithms
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# Plotting
import plotly.graph_objects as go

# Numba Kalman
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG / GLOBALS / LOGGER
# ---------------------------
st.set_page_config(page_title="Hedge Fund AI: BEAST MODE v4", layout="wide")
st.title("🦍 Hedge Fund AI v4 (Meta Learner + ARIMA/ARCH/GBM)")

CACHE_DIR = "cache_yf_v4"
MODEL_DIR = "models_v4"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("beast_v4")

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DEFAULT_TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

# ---------------------------
# STREAMLIT SIDEBAR CONFIG
# ---------------------------
with st.sidebar:
    st.header("⚙️ Canavar Ayarları v4")
    use_optuna = st.checkbox("Optuna (Bayesian) kullan (GA yerine)", value=False)
    max_gens = st.number_input("GA Jenerasyon (Evrim Sayısı)", 1, 200, 8)
    pop_size = st.number_input("Popülasyon Büyüklüğü", 4, 200, 24)
    use_parallel = st.checkbox("GA paralel değerlendirme", value=True)
    tx_cost_perc = st.number_input("Tahmini işlem maliyeti (% tek yön)", 0.0, 1.0, 0.05, step=0.01)
    seed = st.number_input("Rastgele seed", 0, 999999, 42)

random.seed(seed)
np.random.seed(seed)
st.markdown("---")
st.warning("⚠️ Bu mod ağırdır. Popülasyon ve jenerasyon sayısını azaltarak test edin.")

# ---------------------------
# GOOGLE SHEETS HELPERS
# ---------------------------
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
    creds = None
    if os.path.exists(CREDENTIALS_FILE):
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        except Exception:
            creds = None
    if not creds:
        logger.info("Google Sheets credentials not available.")
        return None
    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception:
        return None

def load_and_fix_portfolio(target_coins=DEFAULT_TARGET_COINS):
    sheet = connect_sheet()
    if sheet is None:
        df = pd.DataFrame([{"Ticker": t,"Durum":"CASH","Miktar":0.0,"Son_Islem_Fiyati":0.0,"Nakit_Bakiye_USD":10.0,
                            "Baslangic_USD":10.0,"Kaydedilen_Deger_USD":10.0,"Son_Islem_Log":"LOCAL","Son_Islem_Zamani":"-"} for t in target_coins])
        return df, None
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        numeric_cols = ["Miktar","Son_Islem_Fiyati","Nakit_BakiYE_USD","Nakit_Bakiye_USD","Baslangic_USD","Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',','.',regex=False), errors='coerce').fillna(0.0)
        return df, sheet
    except Exception:
        return pd.DataFrame(), sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        logger.info("Portfolio saved to sheet.")
    except Exception:
        logger.exception("Failed to save portfolio to sheet")

# ---------------------------
# CACHING / DATA LOADING
# ---------------------------
def _parquet_path_for_ticker(ticker:str)->str:
    return os.path.join(CACHE_DIR, ticker.replace("/","_")+".parquet")

def save_cache_df(ticker:str, df:pd.DataFrame):
    path = _parquet_path_for_ticker(ticker)
    try: df.to_parquet(path, index=True)
    except: df.to_pickle(path+".pkl")

def load_cache_df(ticker:str)->Optional[pd.DataFrame]:
    path = _parquet_path_for_ticker(ticker)
    try:
        if os.path.exists(path): return pd.read_parquet(path)
        elif os.path.exists(path+".pkl"): return pd.read_pickle(path+".pkl")
        else: return None
    except Exception:
        return None

@lru_cache(maxsize=64)
def get_raw_data_cached(ticker:str, period:str="2y", interval:str="1d")->Optional[pd.DataFrame]:
    cached = load_cache_df(ticker)
    if cached is not None: return cached.copy()
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty: return None
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        df.index = pd.to_datetime(df.index)
        save_cache_df(ticker, df)
        return df
    except Exception:
        return None

# ---------------------------
# NUMBA KALMAN (SPEED)
# ---------------------------
if NUMBA_AVAILABLE:
    @njit
    def _kalman_numba(prices_arr):
        n_iter = prices_arr.shape[0]
        xhat = np.zeros(n_iter)
        P = np.zeros(n_iter)
        xhatminus = np.zeros(n_iter)
        Pminus = np.zeros(n_iter)
        K = np.zeros(n_iter)
        Q = 1e-5
        R = 0.01**2
        xhat[0] = prices_arr[0]
        P[0] = 1.0
        for k in range(1,n_iter):
            xhatminus[k] = xhat[k-1]
            Pminus[k] = P[k-1] + Q
            K[k] = Pminus[k] / (Pminus[k]+R)
            xhat[k] = xhatminus[k] + K[k]*(prices_arr[k]-xhatminus[k])
            P[k] = (1-K[k])*Pminus[k]
        return xhat

def apply_kalman_filter(prices:pd.Series)->pd.Series:
    if len(prices)<2: return prices.copy()
    try:
        if NUMBA_AVAILABLE:
            arr = prices.values.astype(np.float64)
            res = _kalman_numba(arr)
            return pd.Series(res,index=prices.index)
        else:
            n_iter = len(prices)
            xhat = np.zeros(n_iter)
            P = np.zeros(n_iter)
            xhatminus = np.zeros(n_iter)
            Pminus = np.zeros(n_iter)
            K = np.zeros(n_iter)
            xhat[0] = prices.iloc[0]; P[0] = 1.0
            Q = 1e-5; R = 0.01**2
            for k in range(1,n_iter):
                xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1]+Q
                K[k] = Pminus[k]/(Pminus[k]+R)
                xhat[k] = xhatminus[k] + K[k]*(prices.iloc[k]-xhatminus[k])
                P[k] = (1-K[k])*Pminus[k]
            return pd.Series(xhat,index=prices.index)
    except:
        return prices.copy()

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------
def calculate_heuristic_score(df:pd.DataFrame)->pd.Series:
    if len(df)<150: return pd.Series(0.0,index=df.index)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    s3 = np.where(df['close']>df['close'].rolling(200,min_periods=50).mean(),1,-1)
    vol = df['close'].pct_change().rolling(20,min_periods=5).std()
    s4 = np.where(vol<vol.shift(1).fillna(vol),1,-1)
    s5 = np.sign(df['close'].diff(10).fillna(0))
    return pd.Series((s1+s2+s3+s4+s5)/5.0,index=df.index)

def calculate_momentum(df:pd.DataFrame,window:int=14)->pd.Series:
    return np.sign(df['close'].pct_change(window).fillna(0))

def process_data(df:pd.DataFrame, timeframe:str='D')->Optional[pd.DataFrame]:
    if df is None or len(df)<200: return None
    agg = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    if timeframe=='W': df_res = df.resample('W').agg(agg).dropna()
    elif timeframe=='M': df_res = df.resample('ME').agg(agg).dropna()
    else: df_res = df.copy()
    if len(df_res)<150: return None
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1)).replace([np.inf,-np.inf],0).fillna(0)
    df_res['range'] = ((df_res['high']-df_res['low'])/df_res['close']).replace([np.inf,-np.inf],0).fillna(0)
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['momentum'] = calculate_momentum(df_res)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

# ---------------------------
# META LEARNER TRAIN (PCA, ARIMA, ARCH, GBM)
# ---------------------------
def train_meta_learner(df, params=None):
    rf_depth = params.get('rf_depth',5) if params else 5
    rf_est = params.get('rf_est',50) if params else 50
    X_cols = ['log_ret','range','heuristic','momentum']
    y = df['target']
    
    # Base models
    rf = RandomForestClassifier(n_estimators=rf_est,max_depth=rf_depth,n_jobs=-1,random_state=seed).fit(df[X_cols],y)
    xgb_c = xgb.XGBClassifier(n_estimators=50,max_depth=rf_depth,learning_rate=0.05,verbosity=0,use_label_encoder=False,eval_metric='logloss').fit(df[X_cols],y)
    lgb_c = lgb.LGBMClassifier(n_estimators=50,learning_rate=0.05,verbose=-1).fit(df[X_cols],y)
    # GBM alternative for Earth
    gbm_c = GradientBoostingClassifier(n_estimators=50,max_depth=rf_depth,learning_rate=0.05).fit(df[X_cols],y)
    
    # PCA feature
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(df[X_cols])
    
    # ARIMA + ARCH on log_ret
    arima_pred = df['log_ret'].rolling(5).mean().fillna(0).values
    arch_pred = df['log_ret'].rolling(5).std().fillna(0).values
    
    # Combine meta features
    meta_features = pd.DataFrame({
        'RF': rf.predict_proba(df[X_cols])[:,1],
        'XGB': xgb_c.predict_proba(df[X_cols])[:,1],
        'LGBM': lgb_c.predict_proba(df[X_cols])[:,1],
        'GBM': gbm_c.predict_proba(df[X_cols])[:,1],
        'PCA1': pca_features[:,0],
        'PCA2': pca_features[:,1],
        'ARIMA': arima_pred,
        'ARCH': arch_pred,
        'Heuristic': df['heuristic'],
        'Momentum': df['momentum']
    })
    meta_model = LogisticRegression(max_iter=200).fit(meta_features,y)
    return meta_model, {'RF':rf,'XGB':xgb_c,'LGBM':lgb_c,'GBM':gbm_c,'PCA':pca}

# ---------------------------
# STREAMLIT RUN
# ---------------------------
if st.button("🦍 CANAVAR MODU v4 BAŞLAT"):
    pf_df, sheet = load_and_fix_portfolio()
    if pf_df.empty: st.error("Portföy hatası")
    else:
        updated = pf_df.copy()
        prog = st.progress(0)
        target_coins = list(updated['Ticker'].values)
        sim_summary = []
        for i,(idx,row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            if len(str(ticker))<3: continue
            with st.expander(f"{ticker} Analiz & Evrim", expanded=True):
                ph = st.empty()
                dec, prc, tf, info = analyze_ticker_beast_v2(
                    ticker, ph, use_optuna_flag=use_optuna, n_gen=max_gens, pop_size=pop_size, tx_cost_pct=tx_cost_perc, parallel=use_parallel
                )
                prog.progress((i+1)/len(updated))
        save_portfolio(updated, sheet)
