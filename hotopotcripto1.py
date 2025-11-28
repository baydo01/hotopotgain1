import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- ML LIBS ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import plotly.express as px

warnings.filterwarnings("ignore")

# UI CONFIG
st.set_page_config(page_title="Hedge Fund AI: V13 Final", layout="wide", page_icon="🏛️")
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #00ff88; margin-bottom: 25px;}
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #b0b0b0; margin-top: 5px;}
</style>
<div class="header-box">
    <div class="header-title">🏛️ Hedge Fund AI: V13 (Final Engine)</div>
    <div class="header-sub">Predictive Engine Restored • Risk Adjusted • Auto-Tuned Ensemble</div>
</div>
""", unsafe_allow_html=True)

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"

# --- CONNECT ---
def connect_sheet_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except: pass
    elif os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    if not creds: return None, None
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        try: hist = spreadsheet.worksheet("Gecmis")
        except: hist = spreadsheet.add_worksheet("Gecmis", 1000, 6)
        return spreadsheet.sheet1, hist
    except: return None, None

def load_portfolio():
    pf_sheet, _ = connect_sheet_services()
    if pf_sheet is None: return pd.DataFrame(), None
    try:
        data = pf_sheet.get_all_records()
        df = pd.DataFrame(data)
        cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, pf_sheet
    except: return pd.DataFrame(), None

# --- FEATURES & ML (Same as Bot) ---
def add_technical_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    df['sma_20'] = df['close'].rolling(20).mean()
    df['dist_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
    return df

def get_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def prepare_features(df):
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    df = add_technical_indicators(df)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['volatility_measure'] = df['close'].pct_change().rolling(window=14).std()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna(subset=['atr', 'rsi', 'volatility_measure'])
    future = df.iloc[[-1]].copy()
    hist = df.iloc[:-1].copy()
    return hist, future

class ImputationLab:
    def baydo_impute(self, df):
        filled = df.copy()
        num_cols = filled.select_dtypes(include=[np.number]).columns
        vol = filled['volatility_measure'].interpolate(method='linear').fillna(method='bfill')
        v_high = vol.quantile(0.7); v_low = vol.quantile(0.3)
        r_fast = filled[num_cols].rolling(3, center=True, min_periods=1).mean()
        r_mid = filled[num_cols].rolling(5, center=True, min_periods=1).mean()
        r_slow = filled[num_cols].rolling(9, center=True, min_periods=1).mean()
        base = r_mid.copy()
        base[vol > v_high] = r_fast[vol > v_high]
        base[vol < v_low] = r_slow[vol < v_low]
        filled[num_cols] = filled[num_cols].fillna(base)
        return filled.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    def apply_imputation(self, df_train, df_test, method):
        features = ['log_ret', 'range', 'rsi', 'dist_sma', 'atr', 'volatility_measure']
        features = [f for f in features if f in df_train.columns]
        X_tr = df_train[features].copy(); X_te = df_test[features].copy()
        
        if method == 'Baydo':
            X_tr = self.baydo_impute(X_tr); X_te = self.baydo_impute(X_te)
        elif method == 'MICE':
            try:
                imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
                X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=features, index=X_tr.index)
                X_te = pd.DataFrame(imp.transform(X_te), columns=features, index=X_te.index)
            except: X_tr = self.baydo_impute(X_tr); X_te = self.baydo_impute(X_te)
        elif method == 'KNN':
            try:
                imp = KNNImputer(n_neighbors=5)
                X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=features, index=X_tr.index)
                X_te = pd.DataFrame(imp.transform(X_te), columns=features, index=X_te.index)
            except: X_tr = self.baydo_impute(X_tr); X_te = self.baydo_impute(X_te)
        else:
            X_tr = X_tr.interpolate(method='linear').fillna(0)
            X_te = X_te.interpolate(method='linear').fillna(0)
            
        scaler = RobustScaler()
        X_tr_s = pd.DataFrame(scaler.fit_transform(X_tr), columns=features, index=X_tr.index)
        X_te_s = pd.DataFrame(scaler.transform(X_te), columns=features, index=X_te.index)
        return X_tr_s, X_te_s, scaler

class GrandLeagueBrain:
    def __init__(self):
        self.lab = ImputationLab()
        self.features = ['log_ret', 'range', 'rsi', 'dist_sma', 'atr', 'volatility_measure']
        
    def tune_xgboost(self, X_tr, y_tr):
        split = int(len(X_tr)*0.8)
        Xt, Xv = X_tr.iloc[:split], X_tr.iloc[split:]
        yt, yv = y_tr.iloc[:split], y_tr.iloc[split:]
        best_m = None; best_s = -1
        for d in [3, 5]:
            for lr in [0.05, 0.1]:
                m = xgb.XGBClassifier(n_estimators=80, max_depth=d, learning_rate=lr, n_jobs=1, random_state=42)
                m.fit(Xt, yt)
                s = accuracy_score(yv, m.predict(Xv))
                if s > best_s: best_m = m; best_s = s
        if best_m: best_m.fit(X_tr, y_tr)
        return best_m

    def run_league(self, df):
        impute_methods = ['Baydo', 'MICE', 'KNN', 'Linear']
        strategies = []
        wf_window = 30; steps = 3
        for imp in impute_methods:
            scores_xgb = []; scores_ens = []
            for i in range(steps):
                test_end = len(df) - (i * wf_window); test_start = test_end - wf_window
                if i==0: test_end = len(df)
                train_end = test_start
                if train_end < 200: break
                df_tr = df.iloc[:train_end]; df_val = df.iloc[train_end:test_end]
                X_tr, X_val, _ = self.lab.apply_imputation(df_tr, df_val, imp)
                y_tr, y_val = df_tr['target'], df_val['target']
                m_xgb = self.tune_xgboost(X_tr, y_tr)
                if m_xgb: scores_xgb.append(accuracy_score(y_val, m_xgb.predict(X_val)))
                rf = RandomForestClassifier(50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
                et = ExtraTreesClassifier(50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
                if m_xgb:
                    p = (rf.predict_proba(X_val)[:,1]*0.3 + et.predict_proba(X_val)[:,1]*0.3 + m_xgb.predict_proba(X_val)[:,1]*0.4)
                    scores_ens.append(accuracy_score(y_val, (p>0.5).astype(int)))
            
            avg_x = np.mean(scores_xgb) if scores_xgb else 0
            strategies.append({'name': f"{imp} + XGB", 'type': 'XGB', 'score': avg_x, 'imputer_name': imp})
            avg_e = np.mean(scores_ens) if scores_ens else 0
            strategies.append({'name': f"{imp} + ENS", 'type': 'ENS', 'score': avg_e, 'imputer_name': imp})
        winner = max(strategies, key=lambda x: x['score'])
        return winner, strategies

# --- EXECUTION ---
pf_df, sheet_pf = load_portfolio()
_, sheet_hist = connect_sheet_services()

tab1, tab2, tab3 = st.tabs(["🚀 Control", "📊 Data", "📜 Logs"])

if not pf_df.empty:
    with tab1:
        st.metric("Total Portfolio", f"${pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum():.2f}")
        if st.button("🔥 START ANALYSIS", type="primary"):
            brain = GrandLeagueBrain()
            tickers, scores, vols, sizes = [], [], [], []
            prog = st.progress(0)
            
            for i, (idx, row) in enumerate(pf_df.iterrows()):
                ticker = row['Ticker']
                df = get_data(ticker)
                if df is not None and len(df) > 200:
                    hist, future = prepare_features(df)
                    winner, table = brain.run_league(hist)
                    
                    # --- RESTORED PREDICTION ENGINE ---
                    # 1. Full Train Data Prep
                    X_full, X_future_scaled, _ = brain.lab.apply_imputation(hist, future, winner['imputer_name'])
                    y_full = hist['target']
                    
                    # 2. Retrain Winner & Predict
                    if winner['type'] == 'XGB':
                        model = brain.tune_xgboost(X_full, y_full)
                        prob = model.predict_proba(X_future_scaled)[:,1][0]
                    else: # ENS
                        m_xgb = brain.tune_xgboost(X_full, y_full)
                        rf = RandomForestClassifier(50, max_depth=5, n_jobs=1).fit(X_full, y_full)
                        et = ExtraTreesClassifier(50, max_depth=5, n_jobs=1).fit(X_full, y_full)
                        p1 = rf.predict_proba(X_future_scaled)[:,1]
                        p2 = et.predict_proba(X_future_scaled)[:,1]
                        p3 = m_xgb.predict_proba(X_future_scaled)[:,1]
                        prob = (p1*0.3 + p2*0.3 + p3*0.4)[0]
                    
                    # Risk Logic
                    vol = hist['volatility_measure'].iloc[-1]
                    target_vol = 0.04
                    if vol==0 or np.isnan(vol): risk_factor = 1.0
                    else: risk_factor = np.clip(target_vol / vol, 0.3, 2.0)
                    
                    # Decision
                    if prob > 0.58: signal = "BUY"
                    elif prob < 0.42: signal = "SELL"
                    else: signal = "HOLD"
                    
                    # Visualize
                    tickers.append(ticker); scores.append(winner['score']*100); vols.append(vol*100); sizes.append(risk_factor)
                    with st.expander(f"{ticker} | {signal} | {winner['name']} (Prob: {prob:.2f})"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Validation Acc", f"%{winner['score']*100:.1f}")
                        c2.metric("Probability", f"%{prob*100:.1f}")
                        c3.metric("Risk Factor", f"{risk_factor:.2f}x")
                        st.dataframe(pd.DataFrame(table)[['name', 'score']].sort_values('score', ascending=False).head(3))
                        if signal == "BUY": st.success(f"🟢 STRONG BUY SIGNAL (Adjusted Size: {risk_factor:.2f}x)")
                        elif signal == "SELL": st.error("🔴 SELL SIGNAL")
                        
                prog.progress((i+1)/len(pf_df))
                
            viz_df = pd.DataFrame({'Ticker': tickers, 'Model Acc (%)': scores, 'Volatility (%)': vols, 'Size Multiplier': sizes})
            fig = px.scatter(viz_df, x='Volatility (%)', y='Model Acc (%)', size='Size Multiplier', color='Ticker', title="📊 Risk vs Accuracy Map")
            st.plotly_chart(fig, use_container_width=True)
            st.success("Analysis Complete.")

    with tab2: st.dataframe(pf_df)
    with tab3: 
        if sheet_hist: st.dataframe(pd.DataFrame(sheet_hist.get_all_records()).iloc[::-1])
