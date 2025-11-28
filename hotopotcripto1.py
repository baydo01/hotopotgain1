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

# --- ML & AUTOML ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# UI CONFIG
st.set_page_config(page_title="Hedge Fund AI: V10 Chronos", layout="wide", page_icon="⏳")
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #fff; margin-bottom: 25px;}
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #f0f0f0; margin-top: 5px;}
</style>
<div class="header-box">
    <div class="header-title">⏳ Hedge Fund AI: V10 (Chronos Edition)</div>
    <div class="header-sub">BaydoImputation v2 • Walk-Forward Validation • Static Validation • Grand League</div>
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

def log_transaction(ticker, action, amount, price, model, sheet):
    if sheet:
        now = datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M')
        sheet.append_row([now, ticker, action, float(amount), float(price), model])

def save_portfolio(df, sheet):
    if sheet:
        df_exp = df.copy().astype(str)
        sheet.clear()
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())

# --- DATA & FEATURES ---
def get_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def prepare_raw_features(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df['kalman'] = df['close'].rolling(3).mean()
    df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
    
    # Volatility for Baydo
    df['volatility'] = df['close'].pct_change().rolling(window=10).std()
    
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    future_row = df.iloc[[-1]].copy()
    df_historic = df.iloc[:-1].copy()
    return df_historic, future_row

# --- IMPUTATION LAB (Baydo v2) ---
class ImputationLab:
    def baydo_impute(self, df):
        filled = df.copy()
        numeric_cols = filled.select_dtypes(include=[np.number]).columns
        
        # 1. Rolling Means
        roll_fast = filled[numeric_cols].rolling(window=3, center=True, min_periods=1).mean()
        roll_mid  = filled[numeric_cols].rolling(window=5, center=True, min_periods=1).mean()
        roll_slow = filled[numeric_cols].rolling(window=9, center=True, min_periods=1).mean()
        
        # 2. Volatility Mask
        vol_filled = filled['volatility'].interpolate(method='linear').fillna(method='bfill')
        vol_high = vol_filled.quantile(0.66)
        vol_low = vol_filled.quantile(0.33)
        
        final_fill = roll_mid.copy()
        mask_high = vol_filled > vol_high; final_fill[mask_high] = roll_fast[mask_high]
        mask_low = vol_filled < vol_low; final_fill[mask_low] = roll_slow[mask_low]
        
        filled[numeric_cols] = filled[numeric_cols].fillna(final_fill)
        return filled.interpolate(method='linear').fillna(method='bfill') # Fallback

    def apply_imputation(self, df_train, df_test, method):
        features = ['log_ret', 'range', 'heuristic', 'range_vol_delta', 'avg_ret_5m', 'avg_ret_3y']
        X_tr = df_train[features].copy()
        X_te = df_test[features].copy()
        
        if method == 'Baydo':
            X_tr = self.baydo_impute(df_train)[features]
            X_te = self.baydo_impute(df_test)[features]
        elif method == 'MICE':
            try:
                imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
                X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=features, index=X_tr.index)
                X_te = pd.DataFrame(imp.transform(X_te), columns=features, index=X_te.index)
            except: 
                X_tr = self.baydo_impute(df_train)[features]
                X_te = self.baydo_impute(df_test)[features]
        elif method == 'Linear':
            X_tr = X_tr.interpolate(method='linear').fillna(0)
            X_te = X_te.interpolate(method='linear').fillna(0)
            
        return X_tr, X_te

# --- GRAND LEAGUE ---
class GrandLeagueBrain:
    def __init__(self):
        self.lab = ImputationLab()
        self.features = ['log_ret', 'range', 'heuristic', 'range_vol_delta', 'avg_ret_5m', 'avg_ret_3y']
        
    def train_models(self, X_tr, y_tr):
        best_xgb, best_s = None, -1
        for d in [3, 5]:
            m = xgb.XGBClassifier(n_estimators=80, max_depth=d, learning_rate=0.1, n_jobs=1, random_state=42)
            m.fit(X_tr, y_tr)
            s = m.score(X_tr, y_tr)
            if s > best_s: best_xgb = m; best_s = s
        
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
        return best_xgb, (rf, etc, best_xgb)

    def predict_ens(self, models, X):
        rf, etc, xg = models
        return (rf.predict_proba(X)[:,1] + etc.predict_proba(X)[:,1] + xg.predict_proba(X)[:,1]) / 3

    def run_grand_league(self, df):
        impute_methods = ['Baydo', 'MICE', 'Linear']
        strategies = []
        
        # A. STATIC
        split = int(len(df)*0.85)
        df_tr_s, df_val_s = df.iloc[:split], df.iloc[split:]
        
        for imp in impute_methods:
            X_tr, X_val = self.lab.apply_imputation(df_tr_s, df_val_s, imp)
            y_tr, y_val = df_tr_s['target'], df_val_s['target']
            
            mx, me = self.train_models(X_tr, y_tr)
            strategies.append({'mode':'Static', 'imp':imp, 'type':'XGB', 'm':mx, 'score':accuracy_score(y_val, mx.predict(X_val))})
            strategies.append({'mode':'Static', 'imp':imp, 'type':'ENS', 'm':me, 'score':accuracy_score(y_val, (self.predict_ens(me, X_val)>0.5).astype(int))})
            
        # B. WALK-FORWARD
        wf_steps = 4; wf_win = 30
        for imp in impute_methods:
            sx, se = [], []
            for i in range(wf_steps):
                te_end = len(df) - (i*wf_win); te_st = te_end - wf_win
                if i==0: te_end = len(df)
                tr_end = te_st
                if tr_end < 200: break
                
                d_tr = df.iloc[:tr_end]; d_val = df.iloc[tr_end:te_end]
                X_tr, X_val = self.lab.apply_imputation(d_tr, d_val, imp)
                y_tr, y_val = d_tr['target'], d_val['target']
                
                mx, me = self.train_models(X_tr, y_tr)
                sx.append(accuracy_score(y_val, mx.predict(X_val)))
                se.append(accuracy_score(y_val, (self.predict_ens(me, X_val)>0.5).astype(int)))
                
            X_f, _ = self.lab.apply_imputation(df, df.iloc[-5:], imp)
            fx, fe = self.train_models(X_f, df['target'])
            strategies.append({'mode':'Walk-Fwd', 'imp':imp, 'type':'XGB', 'm':fx, 'score':np.mean(sx) if sx else 0})
            strategies.append({'mode':'Walk-Fwd', 'imp':imp, 'type':'ENS', 'm':fe, 'score':np.mean(se) if se else 0})
            
        strategies.sort(key=lambda x: x['score'], reverse=True)
        winner = strategies[0]
        
        # Test Curve for Winner
        if winner['mode'] == 'Static':
            _, X_test_viz = self.lab.apply_imputation(df_tr_s, df_val_s, winner['imp'])
            rets = df_val_s['close'].pct_change().fillna(0).values
            dates = df_val_s.index
        else:
             # Walk forward görselleştirmesi zor, son parçayı gösterelim
             _, X_test_viz = self.lab.apply_imputation(df.iloc[:-30], df.iloc[-30:], winner['imp'])
             rets = df.iloc[-30:]['close'].pct_change().fillna(0).values
             dates = df.iloc[-30:].index
             
        if winner['type'] == 'XGB': probs = winner['m'].predict_proba(X_test_viz)[:,1]
        else: probs = self.predict_ens(winner['m'], X_test_viz)
        
        sim = 100.0; eq = [100.0]
        for i in range(len(probs)):
            if probs[i]>0.55: sim *= (1+rets[i])
            eq.append(sim)
            
        return {'winner': winner, 'roi': sim-100, 'eq': eq, 'dates': dates, 'table': strategies}

# --- UI ---
pf_df, sheet_pf = load_portfolio()
_, sheet_hist = connect_sheet_services()

tab1, tab2, tab3 = st.tabs(["⏳ Chronos (Bot)", "📑 Veri", "📜 Loglar"])

if not pf_df.empty:
    with tab1:
        st.metric("Toplam Varlık", f"${pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum():.2f}")
        if st.button("🚀 ZAMAN YOLCULUĞUNU BAŞLAT", type="primary"):
            updated_pf = pf_df.copy()
            pool_cash = updated_pf['Nakit_Bakiye_USD'].sum(); updated_pf['Nakit_Bakiye_USD'] = 0.0
            brain = GrandLeagueBrain()
            buy_orders = []
            
            prog = st.progress(0)
            for i, (idx, row) in enumerate(updated_pf.iterrows()):
                ticker = row['Ticker']
                df = get_data(ticker)
                if df is not None:
                    df_h, df_f = prepare_raw_features(df)
                    res = brain.run_grand_league(df_h)
                    winner = res['winner']
                    
                    # Final Prediction
                    lookback = pd.concat([df_h.iloc[-50:], df_f])
                    lab = ImputationLab()
                    if winner['imp'] == 'Baydo': filled = lab.baydo_impute(lookback)
                    elif winner['imp'] == 'Linear': filled = lookback.interpolate(method='linear').fillna(method='bfill')
                    else: 
                        imp = KNNImputer(n_neighbors=5); nc = lookback.select_dtypes(include=[np.number]).columns
                        filled = lookback.copy(); filled[nc] = imp.fit_transform(lookback[nc])
                        
                    X_fin = filled[brain.features].iloc[[-1]]
                    if winner['type'] == 'XGB': prob = winner['m'].predict_proba(X_fin)[:,1][0]
                    else: prob = brain.predict_ens(winner['m'], X_fin)[0]
                    
                    decision = "HOLD"
                    if prob > 0.55: decision = "BUY"
                    elif prob < 0.45: decision = "SELL"
                    
                    with st.expander(f"{ticker} | {decision} | {winner['mode']} {winner['imp']} (Acc: %{winner['score']*100:.1f})"):
                        c1, c2 = st.columns(2)
                        c1.markdown(f"**Yöntem:** `{winner['mode']}` + `{winner['imp']}`")
                        c1.markdown(f"**Model:** `{winner['type']}`")
                        c1.metric("Score", f"%{winner['score']*100:.1f}")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(y=res['eq'], mode='lines', name='Equity', line=dict(color='#00ff88')))
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(pd.DataFrame(res['table']).head(5))
                        
                    current_p = df['close'].iloc[-1]
                    m_desc = f"{winner['mode']}|{winner['imp']}|{winner['type']}"
                    
                    if row['Durum']=='COIN' and decision=="SELL":
                        pool_cash += float(row['Miktar']) * current_p
                        updated_pf.at[idx,'Durum']='CASH'; updated_pf.at[idx,'Miktar']=0.0
                        log_transaction(ticker, "SAT", row['Miktar'], current_p, m_desc, sheet_hist)
                    elif row['Durum']=='CASH' and decision=="BUY":
                        buy_orders.append({'idx':idx, 'ticker':ticker, 'p':current_p, 'w':prob, 'm':m_desc})
                prog.progress((i+1)/len(updated_pf))
            
            if buy_orders and pool_cash > 5:
                tw = sum([b['w'] for b in buy_orders])
                for b in buy_orders:
                    s = (b['w']/tw)*pool_cash; amt = s/b['p']
                    updated_pf.at[b['idx'],'Durum']='COIN'; updated_pf.at[b['idx'],'Miktar']=amt
                    updated_pf.at[b['idx'],'Nakit_Bakiye_USD']=0.0
                    log_transaction(b['ticker'], "AL", amt, b['p'], b['m'], sheet_hist)
            elif pool_cash > 0: updated_pf.at[updated_pf.index[0], 'Nakit_Bakiye_USD'] += pool_cash
            save_portfolio(updated_pf, sheet_pf)
            st.success("Zaman Yolculuğu Tamamlandı!")
    
    with tab2: st.dataframe(pf_df)
    with tab3: 
        if sheet_hist: st.dataframe(pd.DataFrame(sheet_hist.get_all_records()).iloc[::-1])
