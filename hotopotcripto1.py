# trader_bot_full_engine_v2.py
# Trader Bot — Full Engine v2
# - Optimized, cached, walk-forward GA fitness, optional Optuna, numba Kalman, parallel GA
# - Keep model quality; add realistic costs; deterministic seeds; logging

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
import json
import random
import pickle
import logging
import multiprocessing
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz
from functools import lru_cache, partial
from typing import Tuple, Dict, Any, Optional, List

# ML
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

# GA
from deap import base, creator, tools, algorithms

# Optional Bayesian optimizer
try:
    import optuna
    OPTUNA_AVAILABLE = True
except Exception:
    OPTUNA_AVAILABLE = False

# Plotting
import plotly.graph_objects as go

# Numba for Kalman speedup
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

warnings.filterwarnings("ignore")

# ---------------------------
# CONFIG / GLOBALS / LOGGER
# ---------------------------

st.set_page_config(page_title="Hedge Fund AI: BEAST MODE v2", layout="wide")
st.title("🦍 Hedge Fund AI: Canavar Modu v2 (Optimized)")

CACHE_DIR = "cache_yf_v2"
MODEL_DIR = "models_v2"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("beast_v2")

# Default constants
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DEFAULT_TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]

# Streamlit sidebar config
with st.sidebar:
    st.header("⚙️ Canavar Ayarları v2")
    use_optuna = st.checkbox("Optuna (Bayesian) kullan (GA yerine)", value=False)
    max_gens = st.number_input("GA Jenerasyon (Evrim Sayısı)", 1, 200, 8, help="Daha yüksek = daha çok arama ama yavaş")
    pop_size = st.number_input("Popülasyon Büyüklüğü", 4, 200, 24, help="GA popülasyonu")
    use_parallel = st.checkbox("GA paralel değerlendirme", value=True)
    tx_cost_perc = st.number_input("Tahmini işlem maliyeti (% tek yön)", min_value=0.0, max_value=1.0, value=0.05, step=0.01, help="Spread+commission+slippage yaklaşık")
    seed = st.number_input("Rastgele seed", min_value=0, max_value=999999, value=42)
    st.markdown("---")
    st.warning("⚠️ Bu mod ağırdır. Başlatmadan önce pop/gens azaltmayı düşünün. Bilgisayar kaynaklarına dikkat edin.")

random.seed(seed)
np.random.seed(seed)

# ---------------------------
# Google Sheets helpers
# ---------------------------

def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
    except Exception:
        pass
    if creds is None and os.path.exists(CREDENTIALS_FILE):
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
    except Exception as e:
        logger.exception("Failed to connect to Google Sheets")
        return None

def load_and_fix_portfolio(target_coins=DEFAULT_TARGET_COINS):
    sheet = connect_sheet()
    if sheet is None:
        # Return a default DataFrame so UI still works offline
        df = pd.DataFrame([{
            "Ticker": t,
            "Durum": "CASH",
            "Miktar": 0.0,
            "Son_Islem_Fiyati": 0.0,
            "Nakit_Bakiye_USD": 10.0,
            "Baslangic_USD": 10.0,
            "Kaydedilen_Deger_USD": 10.0,
            "Son_Islem_Log": "LOCAL",
            "Son_Islem_Zamani": "-"
        } for t in target_coins])
        return df, None

    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                         "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                         "Son_Islem_Log", "Son_Islem_Zamani"]
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            defaults = []
            for t in target_coins:
                defaults.append([t, "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"])
            for d in defaults: sheet.append_row(d)
            time.sleep(2)
    except Exception:
        pass

    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        df = df[df['Ticker'].astype(str).str.len() > 3]
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_BakiYE_USD", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df, sheet
    except Exception:
        logger.exception("Failed to load portfolio from sheet")
        return pd.DataFrame(), sheet

def save_portfolio(df, sheet):
    if sheet is None:
        logger.info("No sheet connected, skipping save.")
        return
    try:
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        logger.info("Portfolio saved to sheet.")
    except Exception:
        logger.exception("Failed to save portfolio to sheet")

# ---------------------------
# CACHING / DATA LOADING
# ---------------------------

def _parquet_path_for_ticker(ticker: str) -> str:
    safe = ticker.replace("/", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, f"{safe}.parquet")

def save_cache_df(ticker: str, df: pd.DataFrame):
    path = _parquet_path_for_ticker(ticker)
    try:
        df.to_parquet(path, index=True)
    except Exception:
        try:
            df.to_pickle(path + ".pkl")
        except Exception:
            logger.exception("Failed to save cache for %s", ticker)

def load_cache_df(ticker: str) -> Optional[pd.DataFrame]:
    path = _parquet_path_for_ticker(ticker)
    try:
        if os.path.exists(path):
            return pd.read_parquet(path)
        elif os.path.exists(path + ".pkl"):
            return pd.read_pickle(path + ".pkl")
        else:
            return None
    except Exception:
        logger.exception("Failed to load cache for %s", ticker)
        return None

@lru_cache(maxsize=64)
def get_raw_data_cached(ticker: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Downloads via yfinance once and caches to disk."""
    cached = load_cache_df(ticker)
    if cached is not None:
        return cached.copy()
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        df.index = pd.to_datetime(df.index)
        save_cache_df(ticker, df)
        return df
    except Exception:
        logger.exception("Failed to download data for %s", ticker)
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
        R = 0.01 ** 2
        xhat[0] = prices_arr[0]
        P[0] = 1.0
        for k in range(1, n_iter):
            xhatminus[k] = xhat[k - 1]
            Pminus[k] = P[k - 1] + Q
            K[k] = Pminus[k] / (Pminus[k] + R)
            xhat[k] = xhatminus[k] + K[k] * (prices_arr[k] - xhatminus[k])
            P[k] = (1 - K[k]) * Pminus[k]
        return xhat

def apply_kalman_filter(prices: pd.Series) -> pd.Series:
    """Apply Kalman filter to a pandas Series. Uses numba if available."""
    if len(prices) < 2:
        return prices.copy()
    try:
        if NUMBA_AVAILABLE:
            arr = prices.values.astype(np.float64)
            res = _kalman_numba(arr)
            return pd.Series(res, index=prices.index)
        else:
            # fallback python implementation
            n_iter = len(prices)
            xhat = np.zeros(n_iter); P = np.zeros(n_iter)
            xhatminus = np.zeros(n_iter); Pminus = np.zeros(n_iter); K = np.zeros(n_iter)
            xhat[0] = prices.iloc[0]; P[0] = 1.0
            Q = 1e-5; R = 0.01 ** 2
            for k in range(1, n_iter):
                xhatminus[k] = xhat[k - 1]; Pminus[k] = P[k - 1] + Q
                K[k] = Pminus[k] / (Pminus[k] + R)
                xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
                P[k] = (1 - K[k]) * Pminus[k]
            return pd.Series(xhat, index=prices.index)
    except Exception:
        logger.exception("Kalman filter failed; returning original prices")
        return prices.copy()

# ---------------------------
# FEATURE ENGINEERING
# ---------------------------

def calculate_heuristic_score(df: pd.DataFrame) -> pd.Series:
    """5-step heuristic similar to original but robust to NaNs and uses past-only."""
    if len(df) < 150:
        return pd.Series(0.0, index=df.index)
    s1 = np.sign(df['close'].pct_change(5).fillna(0))
    s2 = np.sign(df['close'].pct_change(30).fillna(0))
    s3 = np.where(df['close'] > df['close'].rolling(200, min_periods=50).mean(), 1, -1)
    vol = df['close'].pct_change().rolling(20, min_periods=5).std()
    s4 = np.where(vol < vol.shift(1).fillna(vol), 1, -1)
    s5 = np.sign(df['close'].diff(10).fillna(0))
    return pd.Series((s1 + s2 + s3 + s4 + s5) / 5.0, index=df.index)

def calculate_momentum(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return np.sign(df['close'].pct_change(window).fillna(0))

def process_data(df: pd.DataFrame, timeframe: str='D') -> Optional[pd.DataFrame]:
    """Resample and compute features. timeframe: 'D','W','M'"""
    if df is None or len(df) < 200:
        return None
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W':
        df_res = df.resample('W').agg(agg).dropna()
    elif timeframe == 'M':
        df_res = df.resample('ME').agg(agg).dropna()
    else:
        df_res = df.copy()
    if len(df_res) < 150:
        return None
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1)).replace([np.inf, -np.inf], 0).fillna(0)
    df_res['range'] = ((df_res['high'] - df_res['low']) / df_res['close']).replace([np.inf, -np.inf], 0).fillna(0)
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['momentum'] = calculate_momentum(df_res)
    # target: next period up/down (binary) - use only past info for features
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

# ---------------------------
# BACKTEST & SIMULATION
# ---------------------------

def simulate_strategy_from_probs(test: pd.DataFrame, probs: np.ndarray, tx_cost_pct: float = 0.0) -> Dict[str, Any]:
    """
    Simulate simple all-in/all-out strategy:
    - if prob > 0.5+threshold buy (all-in), if prob < 0.5-threshold sell (all-out)
    - simulate ROI and HODL baseline
    - include tx_cost_pct (single side percent)
    """
    if len(test) == 0:
        return {"bot_eq": [], "hodl_eq": [], "alpha": -9999, "bot_roi": 0.0, "hodl_roi":0.0}
    threshold = 0.25  # same as original code
    sim_eq = []
    hodl_eq = []
    cash = 100.0
    coin = 0.0
    p0 = float(test['close'].iloc[0])
    for i in range(len(test)):
        p = float(test['close'].iloc[i])
        prob = float(probs[i])
        s = (prob - 0.5) * 2  # scaled to [-1,1]
        # buy
        if s > threshold and cash > 0:
            # pay tx cost when buying
            coin = (cash * (1 - tx_cost_pct)) / p
            cash = 0.0
        # sell
        elif s < -threshold and coin > 0:
            cash = coin * p * (1 - tx_cost_pct)
            coin = 0.0
        total = cash + coin * p
        sim_eq.append(total)
        hodl_eq.append((100.0 / p0) * p)
    bot_final = sim_eq[-1] if len(sim_eq)>0 else 100.0
    hodl_final = hodl_eq[-1] if len(hodl_eq)>0 else 100.0
    alpha = bot_final - hodl_final
    return {"bot_eq": sim_eq, "hodl_eq": hodl_eq, "alpha": alpha, "bot_roi": bot_final-100.0, "hodl_roi": hodl_final-100.0}

# ---------------------------
# WALK-FORWARD CV (for GA fitness)
# ---------------------------

def walk_forward_splits(df: pd.DataFrame, n_splits: int = 3, test_size: int = 60) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create walk-forward splits: sequential train/test windows.
    Returns list of (train, test) pairs.
    """
    total = len(df)
    if total < test_size * (n_splits + 1):
        # fallback to simple split
        split = int(total * 0.8)
        return [(df.iloc[:split], df.iloc[split:])]
    splits = []
    step = int((total - test_size) / (n_splits + 1))
    for i in range(n_splits):
        train_end = (i+1) * step
        train = df.iloc[:train_end]
        test = df.iloc[train_end: train_end + test_size]
        if len(test) < 10:
            continue
        splits.append((train, test))
    if not splits:
        split = int(total * 0.8)
        splits.append((df.iloc[:split], df.iloc[split:]))
    return splits

# ---------------------------
# FITNESS / GA / OPTUNA
# ---------------------------

# DEAP creator - ensure idempotent
if not hasattr(creator, 'FitnessMax'):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,))
    creator.create('Individual', list, fitness=creator.FitnessMax)

def evaluate_individual_alpha(individual, df: pd.DataFrame, tx_cost_pct: float, n_splits: int = 2, verbose=False):
    """
    Given an individual (hyperparams), compute average alpha across walk-forward splits.
    Individual layout: [rf_depth, rf_n, xgb_depth, xgb_lr, lgbm_lr]
    """
    try:
        rf_d, rf_n, xgb_d, xgb_l, lgb_l = individual
        splits = walk_forward_splits(df, n_splits=n_splits, test_size=60)
        alphas = []
        for train, test in splits:
            if len(train) < 30 or len(test) < 10:
                continue
            X_cols = ['log_ret', 'range', 'heuristic', 'momentum']
            y_train = train['target']
            # train models
            rf = RandomForestClassifier(n_estimators=int(max(10, rf_n)), max_depth=int(max(2, rf_d)), n_jobs=-1, random_state=seed).fit(train[X_cols], y_train)
            xgb_c = xgb.XGBClassifier(n_estimators=50, max_depth=int(max(2, xgb_d)), learning_rate=float(max(0.005, xgb_l)), use_label_encoder=False, eval_metric='logloss', n_jobs=-1, verbosity=0).fit(train[X_cols], y_train)
            lgb_c = lgb.LGBMClassifier(n_estimators=50, learning_rate=float(max(0.005, lgb_l)), n_jobs=-1, verbose=-1).fit(train[X_cols], y_train)
            # HMM - fit on train
            scaler = StandardScaler()
            X_hmm = scaler.fit_transform(train[['log_ret', 'range']])
            try:
                hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=seed)
                hmm.fit(X_hmm)
                probs_hmm_train = None
            except Exception:
                hmm = None
            # meta_X for test
            X_test = test[X_cols].copy()
            probs_rf = rf.predict_proba(X_test)[:, 1]
            probs_xgb = xgb_c.predict_proba(X_test)[:, 1]
            probs_lgb = lgb_c.predict_proba(X_test)[:, 1]
            hmm_s_t = np.zeros(len(test))
            if hmm is not None:
                pt = hmm.predict_proba(scaler.transform(test[['log_ret', 'range']]))
                hmm_s_t = pt[:, np.argmax(hmm.means_[:, 0])] - pt[:, np.argmin(hmm.means_[:, 0])]
            meta_X_test = pd.DataFrame({
                'RF': probs_rf,
                'XGB': probs_xgb,
                'LGBM': probs_lgb,
                'HMM': hmm_s_t,
                'Heuristic': test['heuristic'].values,
                'Momentum': test['momentum'].values
            })
            # For fitness calculation we need meta model weights - we approximate by training meta on train
            # TRAIN meta
            meta_X_train = pd.DataFrame({
                'RF': rf.predict_proba(train[X_cols])[:, 1],
                'XGB': xgb_c.predict_proba(train[X_cols])[:, 1],
                'LGBM': lgb_c.predict_proba(train[X_cols])[:, 1],
                'HMM': (hmm.predict_proba(X_hmm)[:, np.argmax(hmm.means_[:,0])] - hmm.predict_proba(X_hmm)[:, np.argmin(hmm.means_[:,0])]) if hmm is not None else np.zeros(len(train)),
                'Heuristic': train['heuristic'].values,
                'Momentum': train['momentum'].values
            })
            try:
                meta_model = LogisticRegression(max_iter=200).fit(meta_X_train, train['target'])
                f_probs = meta_model.predict_proba(meta_X_test)[:, 1]
            except Exception:
                # fallback simple ensemble average
                f_probs = (probs_rf + probs_xgb + probs_lgb) / 3.0
            # simulate
            sim = simulate_strategy_from_probs(test, f_probs, tx_cost_pct=tx_cost_pct/100.0)
            alphas.append(sim['alpha'])
        if len(alphas) == 0:
            return (-9999.0,)
        avg_alpha = float(np.mean(alphas))
        # DEAP expects a tuple
        return (avg_alpha,)
    except Exception:
        logger.exception("Error evaluating individual")
        return (-9999.0,)

def run_heavy_ga_improved(df: pd.DataFrame, n_gen: int = 8, pop_size: int = 24, tx_cost_pct: float = 0.05, parallel: bool = True):
    """
    Improved GA:
    - individuals encode RF/XGB/LGB params
    - fitness = average alpha across walk-forward splits
    - parallel evaluation via multiprocessing
    """
    X_cols = ['log_ret', 'range', 'heuristic', 'momentum']
    # Setup toolbox
    toolbox = base.Toolbox()
    toolbox.register('rf_depth', random.randint, 2, 16)
    toolbox.register('rf_est', random.randint, 10, 200)
    toolbox.register('xgb_depth', random.randint, 2, 12)
    toolbox.register('xgb_lr', random.uniform, 0.005, 0.3)
    toolbox.register('lgbm_lr', random.uniform, 0.005, 0.3)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.rf_depth, toolbox.rf_est, toolbox.xgb_depth, toolbox.xgb_lr, toolbox.lgbm_lr), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # evaluator wrapper that binds df and tx_cost
    eval_func = partial(evaluate_individual_alpha, df=df, tx_cost_pct=tx_cost_pct, n_splits=3)
    # register evaluate with optional parallelization
    if parallel:
        # Use multiprocessing pool map for evaluation
        pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 1))
        toolbox.register("map", pool.map)
        toolbox.register('evaluate', lambda ind: eval_func(ind))
    else:
        toolbox.register('evaluate', eval_func)
    toolbox.register('mate', tools.cxTwoPoint)
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)
    pop = toolbox.population(n=pop_size)
    # run evolution (use eaSimple but we manually evaluate with map if parallel)
    # We can't pass the pool into eaSimple easily so we'll implement a simple loop
    hof = tools.HallOfFame(3)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean); stats.register("max", np.max); stats.register("min", np.min)
    # initial evaluation
    if parallel:
        # prepare list for pool.map; pool.map calls our evaluate_indirectly
        # but easier: evaluate sequentially here (less code complexity) OR use map for toolbox.evaluate
        for ind in pop:
            ind.fitness.values = eval_func(ind)
    else:
        for ind in pop:
            ind.fitness.values = toolbox.evaluate(ind)
    for gen in range(n_gen):
        # selection
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        # crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values; del child2.fitness.values
        # mutation
        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        # evaluate invalid individuals
        invalids = [ind for ind in offspring if not ind.fitness.valid]
        if parallel and len(invalids)>0:
            # map eval_func across invalids in parallel
            results = pool.map(eval_func, invalids)
            for ind, fit in zip(invalids, results):
                ind.fitness.values = fit
        else:
            for ind in invalids:
                ind.fitness.values = toolbox.evaluate(ind)
        pop[:] = offspring
        hof.update(pop)
        record = stats.compile(pop)
        logger.info(f"GA gen {gen+1}/{n_gen} stats: {record}")
    # close pool
    try:
        if parallel:
            pool.close(); pool.join()
    except Exception:
        pass
    best = tools.selBest(pop, 1)[0]
    # map best to dict
    best_params = {
        'rf_depth': int(max(2, best[0])),
        'rf_est': int(max(10, best[1])),
        'xgb_depth': int(max(2, best[2])),
        'xgb_lr': max(0.005, float(best[3])),
        'lgbm_lr': max(0.005, float(best[4]))
    }
    return best_params, best

# OPTUNA objective for comparison / alternative
def optuna_objective(trial, df: pd.DataFrame, tx_cost_pct: float):
    rf_d = trial.suggest_int("rf_depth", 2, 16)
    rf_n = trial.suggest_int("rf_est", 10, 200)
    xgb_d = trial.suggest_int("xgb_depth", 2, 12)
    xgb_lr = trial.suggest_float("xgb_lr", 0.005, 0.3, log=True)
    lgb_lr = trial.suggest_float("lgbm_lr", 0.005, 0.3, log=True)
    indiv = [rf_d, rf_n, xgb_d, xgb_lr, lgb_lr]
    val = evaluate_individual_alpha(indiv, df=df, tx_cost_pct=tx_cost_pct, n_splits=3)
    return -val[0]  # optuna minimizes; we want max alpha so minimize -alpha

def run_optuna_search(df: pd.DataFrame, tx_cost_pct: float, n_trials: int = 30):
    if not OPTUNA_AVAILABLE:
        raise RuntimeError("Optuna not available")
    study = optuna.create_study(direction="minimize")
    func = lambda trial: optuna_objective(trial, df, tx_cost_pct)
    study.optimize(func, n_trials=n_trials, show_progress_bar=True)
    best = study.best_trial.params
    return {
        'rf_depth': int(best['rf_depth']),
        'rf_est': int(best['rf_est']),
        'xgb_depth': int(best['xgb_depth']),
        'xgb_lr': float(best['xgb_lr']),
        'lgbm_lr': float(best['lgbm_lr'])
    }

# ---------------------------
# TRAIN & SIMULATE FULL (final)
# ---------------------------

def train_and_simulate_full(df: pd.DataFrame, best_params: Dict[str, Any], tx_cost_pct: float = 0.05):
    """
    Train models on training set, stack, run simulation on last test window,
    return final signal and info (weights, sim eqs, alpha, etc.)
    """
    test_size = 60
    if len(df) < test_size + 50:
        return 0.0, None
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    X_cols = ['log_ret', 'range', 'heuristic', 'momentum']
    y_train = train['target']
    # Train base models
    rf = RandomForestClassifier(n_estimators=best_params['rf_est'], max_depth=best_params['rf_depth'], random_state=seed, n_jobs=-1).fit(train[X_cols], y_train)
    xgb_c = xgb.XGBClassifier(n_estimators=50, max_depth=best_params['xgb_depth'], learning_rate=best_params['xgb_lr'], use_label_encoder=False, eval_metric='logloss', n_jobs=-1, verbosity=0).fit(train[X_cols], y_train)
    lgb_c = lgb.LGBMClassifier(n_estimators=50, learning_rate=best_params['lgbm_lr'], n_jobs=-1, verbose=-1).fit(train[X_cols], y_train)
    # HMM
    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(train[['log_ret', 'range']])
    try:
        hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=seed)
        hmm.fit(X_hmm)
    except Exception:
        hmm = None
    # Prepare meta features for train
    hmm_pred = np.zeros(len(train))
    if hmm is not None:
        probs = hmm.predict_proba(X_hmm)
        bull = np.argmax(hmm.means_[:, 0]); bear = np.argmin(hmm.means_[:, 0])
        hmm_pred = probs[:, bull] - probs[:, bear]
    meta_X_train = pd.DataFrame({
        'RF': rf.predict_proba(train[X_cols])[:, 1],
        'XGB': xgb_c.predict_proba(train[X_cols])[:, 1],
        'LGBM': lgb_c.predict_proba(train[X_cols])[:, 1],
        'HMM': hmm_pred,
        'Heuristic': train['heuristic'].values,
        'Momentum': train['momentum'].values
    })
    meta_model = LogisticRegression(max_iter=200).fit(meta_X_train, y_train)
    weights = meta_model.coef_[0]
    # Build meta features for test
    X_test = test[X_cols]
    probs_rf = rf.predict_proba(X_test)[:, 1]
    probs_xgb = xgb_c.predict_proba(X_test)[:, 1]
    probs_lgb = lgb_c.predict_proba(X_test)[:, 1]
    hmm_s_t = np.zeros(len(test))
    if hmm is not None:
        pt = hmm.predict_proba(scaler.transform(test[['log_ret', 'range']]))
        hmm_s_t = pt[:, np.argmax(hmm.means_[:, 0])] - pt[:, np.argmin(hmm.means_[:, 0])]
    meta_X_test = pd.DataFrame({
        'RF': probs_rf,
        'XGB': probs_xgb,
        'LGBM': probs_lgb,
        'HMM': hmm_s_t,
        'Heuristic': test['heuristic'].values,
        'Momentum': test['momentum'].values
    })
    f_probs = meta_model.predict_proba(meta_X_test)[:, 1]
    # simulate
    sim = simulate_strategy_from_probs(test, f_probs, tx_cost_pct=tx_cost_pct/100.0)
    final_signal = (f_probs[-1] - 0.5) * 2
    info = {
        "weights": weights,
        "bot_eq": sim['bot_eq'], "hodl_eq": sim['hodl_eq'], "dates": test.index,
        "alpha": sim['alpha'],
        "bot_roi": sim['bot_roi'],
        "hodl_roi": sim['hodl_roi'],
        "conf": f_probs[-1],
        "scores": {
            "Heuristic": float(test['heuristic'].iloc[-1]),
            "Momentum": float(test['momentum'].iloc[-1])
        }
    }
    return final_signal, info

# ---------------------------
# ANALYZE TICKER (MAIN ENGINE)
# ---------------------------

def analyze_ticker_beast_v2(ticker: str, status_ph, use_optuna_flag: bool, n_gen: int, pop_size:int, tx_cost_pct: float, parallel: bool):
    raw_df = get_raw_data_cached(ticker)
    if raw_df is None:
        status_ph.error("Veri Yok")
        return "HATA", 0.0, "YOK", None
    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
    best_alpha = -99999.0
    final_decision = "BEKLE"
    winning_tf = "YOK"
    best_info = None
    for tf_name, tf_code in timeframes.items():
        status_ph.text(f"🧬 {tf_name} için veri işleniyor...")
        df = process_data(raw_df, tf_code)
        if df is None:
            status_ph.text(f"⚠️ {tf_name} için yeterli veri yok.")
            continue
        status_ph.text(f"🧬 {tf_name} için optimizasyon başlıyor...")
        # choose optimizer
        if use_optuna_flag and OPTUNA_AVAILABLE:
            try:
                best_params = run_optuna_search(df, tx_cost_pct=tx_cost_pct, n_trials=20)
                logger.info("Optuna found: %s", best_params)
            except Exception:
                best_params = None
        else:
            best_params, best_ind = run_heavy_ga_improved(df, n_gen=n_gen, pop_size=pop_size, tx_cost_pct=tx_cost_pct, parallel=parallel)
        if best_params is None:
            status_ph.text("Optimizasyon başarısız.")
            continue
        status_ph.text("Eğitim & simülasyon yapılıyor...")
        sig, info = train_and_simulate_full(df, best_params, tx_cost_pct=tx_cost_pct)
        if info and info['alpha'] > best_alpha:
            best_alpha = info['alpha']
            winning_tf = tf_name
            best_info = info
            if sig > 0.25:
                final_decision = "AL"
            elif sig < -0.25:
                final_decision = "SAT"
            else:
                final_decision = "BEKLE"
        status_ph.text(f"{tf_name} tamamlandı. Alpha: {info['alpha'] if info else 'N/A'}")
    return final_decision, current_price, winning_tf, best_info

# ---------------------------
# STREAMLIT UI (MAIN)
# ---------------------------

if st.button("🦍 CANAVAR MODU BAŞLAT (ANALİZ ET)", type="primary"):
    pf_df, sheet = load_and_fix_portfolio()
    if pf_df.empty:
        st.error("Portföy hatası.")
    else:
        updated = pf_df.copy()
        prog = st.progress(0)
        sim_summary = []
        target_coins = list(updated['Ticker'].values)
        for i, (idx, row) in enumerate(updated.iterrows()):
            ticker = row['Ticker']
            if len(str(ticker)) < 3:
                continue
            with st.expander(f"🧬 {ticker} Analiz & Evrim Raporu", expanded=True):
                ph = st.empty()
                dec, prc, tf, info = analyze_ticker_beast_v2(
                    ticker, ph, use_optuna, n_gen=max_gens, pop_size=pop_size,
                    tx_cost_pct=tx_cost_perc, parallel=use_parallel
                )
                if dec != "HATA" and info:
                    sim_summary.append({
                        "Coin": ticker, "TF": tf, "Alpha": info['alpha'],
                        "Bot": info['bot_roi'], "HODL": info['hodl_roi']
                    })
                    w = info['weights']
                    w_df = pd.DataFrame({
                        'Model': ['RF', 'XGB', 'LGBM', 'HMM', 'Heuristic', 'Momentum'],
                        'Etki': w
                    })
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.markdown(f"### Karar: **{dec}**")
                        st.caption(f"Seçilen Zaman: {tf}")
                        st.markdown(f"**Senin Puanın:** {info['scores']['Heuristic']:.2f}")
                        st.dataframe(w_df, hide_index=True)
                    with c2:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['bot_eq'], name="Bot"))
                        fig.add_trace(go.Scatter(x=info['dates'], y=info['hodl_eq'], name="HODL", line=dict(dash='dot')))
                        col_ti = "green" if info['alpha'] > 0 else "red"
                        fig.update_layout(title=f"Alpha: {info['alpha']:+.2f}", title_font_color=col_ti, height=300, template="plotly_dark", margin=dict(t=30,b=0,l=0,r=0))
                        st.plotly_chart(fig, use_container_width=True)
                    # İşlem kaydı
                    stt = row.get('Durum', 'CASH')
                    log_msg = row.get('Son_Islem_Log', '')
                    ts = datetime.now(pytz.timezone('Europe/Istanbul')).strftime("%d-%m %H:%M")
                    if stt == 'COIN' and dec == 'SAT':
                        amt = float(row.get('Miktar', 0.0)); cash_val = amt * prc
                        updated.at[idx, 'Durum'] = 'CASH'; updated.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                        updated.at[idx, 'Miktar'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                        log_msg = f"SAT ({tf}) A:{info['alpha']:.2f}"; updated.at[idx, 'Son_Islem_Zamani'] = ts
                    elif stt == 'CASH' and dec == 'AL':
                        cash = float(row.get('Nakit_Bakiye_USD', 0.0)); amt = 0.0
                        if cash > 1.0:
                            amt = cash / prc
                            updated.at[idx, 'Durum'] = 'COIN'; updated.at[idx, 'Miktar'] = amt
                            updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0; updated.at[idx, 'Son_Islem_Fiyati'] = prc
                            log_msg = f"AL ({tf}) A:{info['alpha']:.2f}"; updated.at[idx, 'Son_Islem_Zamani'] = ts
                    val = (float(updated.at[idx, 'Miktar']) * prc) if updated.at[idx, 'Durum'] == 'COIN' else float(updated.at[idx, 'Nakit_Bakiye_USD'])
                    updated.at[idx, 'Kaydedilen_Deger_USD'] = val
                    updated.at[idx, 'Son_Islem_Log'] = log_msg
                    ph.success(f"Tamamlandı. Alpha: {info['alpha']:+.2f}")
            prog.progress((i+1)/len(updated))
        save_portfolio(updated, sheet)
        st.divider()
        if sim_summary:
            sdf = pd.DataFrame(sim_summary)
            c1, c2, c3 = st.columns(3)
            c1.metric("Genel Alpha", f"{sdf['Alpha'].mean():.2f}")
            c2.metric("Bot ROI", f"{sdf['Bot'].mean():.2f}")
            c3.metric("HODL ROI", f"{sdf['HODL'].mean():.2f}")
            st.dataframe(sdf)
        st.success("✅ Canavar Analizi Tamamlandı!")

st.divider()
try:
    df_v, _ = load_and_fix_portfolio()
    if not df_v.empty:
        st.subheader("📋 Mevcut Portföy")
        st.dataframe(df_v)
        tot = df_v['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Varlık", f"${tot:.2f}")
except Exception:
    pass
