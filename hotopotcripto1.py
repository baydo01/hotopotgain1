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

# --- ML & QUANT LIBRARIES ---
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & UI
# =============================================================================
st.set_page_config(page_title="Hedge Fund AI: Ultimate V5", layout="wide", page_icon="🏛️")

st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {
        background: linear-gradient(135deg, #1c1e21 0%, #363b45 100%);
        padding: 25px; border-radius: 12px; border-left: 5px solid #00CC96;
        margin-bottom: 25px; box-shadow: 0 4px 20px rgba(0,0,0,0.6);
    }
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #b0b0b0; margin-top: 5px;}
    .winner-tag {padding: 5px 10px; border-radius: 5px; font-weight: bold; font-size: 12px;}
</style>
<div class="header-box">
    <div class="header-title">🏛️ Hedge Fund AI: Ultimate V5 (Tournament)</div>
    <div class="header-sub">Dynamic Model Selection (Solo XGB vs Ensemble) • Volatility Guard • Google Sheets Sync</div>
</div>
""", unsafe_allow_html=True)

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "730d"

# =============================================================================
# 2. BAĞLANTI KATMANI
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

def load_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, sheet
    except: return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_exp = df.copy().astype(str)
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())
    except: pass

# =============================================================================
# 3. FEATURE ENGINEERING
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

def get_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def process_data_advanced(df):
    if df is None or len(df)<150: return None
    df = df.copy()
    df['kalman_close'] = apply_kalman_filter(df['close'].fillna(method='ffill'))
    df['log_ret'] = np.log(df['kalman_close']/df['kalman_close'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = calculate_heuristic_score(df)
    
    # Historical Z-Scores
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    df['target'] = (df['close'].shift(-1)>df['close']).astype(int)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['target'], inplace=True)
    return df

def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0)
    imputer = KNNImputer(n_neighbors=5)
    try:
        df_imputed = df.copy()
        df_imputed[features] = imputer.fit_transform(df[features])
        return df_imputed
    except: return df.fillna(0)

# =============================================================================
# 4. QUANT & BRAIN (TOURNAMENT LOGIC)
# =============================================================================
def estimate_garch_vol(returns):
    if len(returns) < 200: return 0.0
    try:
        am = arch_model(100*returns, vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(disp='off')
        return float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100)
    except: return 0.0

class HedgeFundBrain:
    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0)
        
    def train_predict_tournament(self, df):
        # 1. Feature Prep
        features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        df = smart_impute(df, features)
        
        # 2. Train / Tournament Split (Last 60 days for Tournament)
        test_size = 60
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:]
        
        X_tr = train[features]; y_tr = train['target']
        X_test = test[features]
        
        # 3. Train Models
        rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3).fit(X_tr, y_tr)
        xgb_solo = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1).fit(X_tr, y_tr)
        
        # 4. Meta-Learner (Ensemble)
        scaler_hmm = StandardScaler()
        try:
            X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']].fillna(0))
            hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
            hmm_probs = hmm.predict_proba(X_hmm)
        except: hmm_probs = np.zeros((len(train), 3)); hmm = None
        
        meta_X_tr = pd.DataFrame({
            'RF': rf.predict_proba(X_tr)[:,1],
            'ETC': etc.predict_proba(X_tr)[:,1],
            'XGB': xgb_c.predict_proba(X_tr)[:,1],
            'Heuristic': train['heuristic'],
            'HMM_0': hmm_probs[:,0], 'HMM_1': hmm_probs[:,1], 'HMM_2': hmm_probs[:,2]
        }, index=train.index).fillna(0)
        
        self.meta_model.fit(meta_X_tr, y_tr)
        
        # 5. TOURNAMENT SIMULATION (Backtest on last 60 days)
        try:
            X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']].fillna(0))
            hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
        except: hmm_probs_t = np.zeros((len(test),3))
        
        meta_X_test = pd.DataFrame({
            'RF': rf.predict_proba(X_test)[:,1],
            'ETC': etc.predict_proba(X_test)[:,1],
            'XGB': xgb_c.predict_proba(X_test)[:,1],
            'Heuristic': test['heuristic'],
            'HMM_0': hmm_probs_t[:,0], 'HMM_1': hmm_probs_t[:,1], 'HMM_2': hmm_probs_t[:,2]
        }, index=test.index).fillna(0)
        
        probs_ens = self.meta_model.predict_proba(meta_X_test)[:,1]
        probs_solo = xgb_solo.predict_proba(X_test)[:,1]
        
        # ROI Calculation
        sim_ens = 100.0; sim_solo = 100.0
        rets = test['close'].pct_change().fillna(0).values
        equity_curve_ens = [100.0]; equity_curve_solo = [100.0]
        
        for i in range(len(test)):
            ret = rets[i]
            if probs_ens[i] > 0.55: sim_ens *= (1 + ret)
            if probs_solo[i] > 0.55: sim_solo *= (1 + ret)
            equity_curve_ens.append(sim_ens); equity_curve_solo.append(sim_solo)
            
        # 6. PICK WINNER
        garch_vol = estimate_garch_vol(df['log_ret'].dropna().iloc[-200:])
        
        if sim_solo > sim_ens:
            winner = "Solo XGBoost"
            final_prob = probs_solo[-1]
            winner_roi = sim_solo - 100
        else:
            winner = "Ensemble"
            final_prob = probs_ens[-1]
            winner_roi = sim_ens - 100
            
        return {
            'prob': final_prob,
            'winner': winner,
            'garch_vol': garch_vol,
            'winner_roi': winner_roi,
            'dates': test.index,
            'eq_ens': equity_curve_ens,
            'eq_solo': equity_curve_solo
        }

# =============================================================================
# 5. EXECUTION & DISPLAY
# =============================================================================
pf_df, sheet = load_portfolio()

if not pf_df.empty:
    with st.sidebar:
        st.header("💼 Portföy")
        total_val = pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Varlık", f"${total_val:.2f}")
        st.divider()
        st.write("Varlık Dağılımı")
        st.dataframe(pf_df[pf_df['Durum']=='COIN'][['Ticker', 'Miktar']], hide_index=True)

    if st.button("🏆 TURNUVAYI BAŞLAT VE ANALİZ ET", type="primary"):
        updated_pf = pf_df.copy()
        
        # --- KRİTİK DÜZELTME BAŞLANGICI ---
        # 1. Mevcut nakiti topla
        pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
        # 2. Tablodaki nakiti SIFIRLA (Çifte harcamayı önler)
        updated_pf['Nakit_Bakiye_USD'] = 0.0
        # --- KRİTİK DÜZELTME BİTİŞİ ---
        
        buy_orders = []
        session_log = [] # İşlem geçmişi tablosu için
        
        brain = HedgeFundBrain()
        prog = st.progress(0)
        
        for i, (idx, row) in enumerate(updated_pf.iterrows()):
            ticker = row['Ticker']
            df = get_data(ticker)
            
            if df is not None:
                df = process_data_advanced(df)
                res = brain.train_predict_tournament(df)
                
                prob = res['prob']
                winner = res['winner']
                
                # Karar
                decision = "HOLD"
                if prob > 0.55: decision = "BUY"
                elif prob < 0.45: decision = "SELL"
                
                # --- GÖRSELLEŞTİRME ---
                with st.expander(f"{ticker} | {decision} | Kazanan: {winner} (ROI: %{res['winner_roi']:.1f})"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Güven Skoru", f"%{prob*100:.1f}")
                    c2.metric("Volatilite (GARCH)", f"%{res['garch_vol']*100:.2f}")
                    c3.markdown(f"**Kazanan Model:** `{winner}`")
                    
                    # Turnuva Grafiği
                    fig = go.Figure()
                    dates = res['dates']
                    fig.add_trace(go.Scatter(x=dates, y=res['eq_ens'][1:], name='Ensemble', line=dict(color='#00CC96')))
                    fig.add_trace(go.Scatter(x=dates, y=res['eq_solo'][1:], name='Solo XGB', line=dict(color='#636EFA', dash='dot')))
                    fig.update_layout(title="Son 60 Gün Turnuva Performansı", height=300, margin=dict(l=0,r=0,t=30,b=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                # --- İŞLEM MANTIĞI ---
                current_p = df['close'].iloc[-1]
                
                # SATIŞ
                if row['Durum'] == 'COIN':
                    if decision == "SELL":
                        val = float(row['Miktar']) * current_p
                        pool_cash += val # Nakit havuza eklendi
                        updated_pf.at[idx, 'Durum'] = 'CASH'
                        updated_pf.at[idx, 'Miktar'] = 0.0
                        updated_pf.at[idx, 'Son_Islem_Log'] = f"SAT ({winner})"
                        st.toast(f"🛑 {ticker} Satıldı!")
                        session_log.append({'Zaman': datetime.now().strftime("%H:%M"), 'Ticker': ticker, 'İşlem': 'SAT', 'Fiyat': current_p, 'Detay': f"Model: {winner}"})
                
                # ALIM ADAYI
                elif row['Durum'] == 'CASH':
                    if decision == "BUY":
                        # Volatiliteye göre pozisyon ayarla
                        pos_scale = 0.5 if res['garch_vol'] > 0.05 else 1.0
                        buy_orders.append({
                            'idx': idx, 'ticker': ticker, 'price': current_p, 
                            'weight': prob * pos_scale, 'winner': winner
                        })
            
            prog.progress((i+1)/len(updated_pf))
            
        # --- ALIMLARI YAP ---
        if buy_orders and pool_cash > 10:
            total_w = sum([b['weight'] for b in buy_orders])
            for b in buy_orders:
                share = (b['weight'] / total_w) * pool_cash
                amt = share / b['price']
                
                updated_pf.at[b['idx'], 'Durum'] = 'COIN'
                updated_pf.at[b['idx'], 'Miktar'] = amt
                updated_pf.at[b['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated_pf.at[b['idx'], 'Son_Islem_Fiyati'] = b['price']
                log_msg = f"AL ({b['winner']})"
                updated_pf.at[b['idx'], 'Son_Islem_Log'] = log_msg
                st.toast(f"✅ {b['ticker']} Alındı!")
                session_log.append({'Zaman': datetime.now().strftime("%H:%M"), 'Ticker': b['ticker'], 'İşlem': 'AL', 'Fiyat': b['price'], 'Detay': f"Tutar: ${share:.1f}"})
        
        # --- KALAN NAKİTİ GERİ YAZ ---
        # Eğer alım yapılmadıysa veya para arttıysa, ilk satıra geri yükle
        elif pool_cash > 0:
            fidx = updated_pf.index[0]
            current_cash_in_row = float(updated_pf.at[fidx, 'Nakit_Bakiye_USD'])
            updated_pf.at[fidx, 'Nakit_Bakiye_USD'] = current_cash_in_row + pool_cash
                    
        # Değerleme
        for idx, row in updated_pf.iterrows():
            if row['Durum'] == 'COIN':
                try:
                    p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                    updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
                except: pass
            else:
                updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
        
        save_portfolio(updated_pf, sheet)
        
        # --- İŞLEM GEÇMİŞİ TABLOSU ---
        st.divider()
        st.subheader("📝 Bu Oturumdaki İşlem Geçmişi")
        if session_log:
            st.table(pd.DataFrame(session_log))
        else:
            st.info("Bu turda herhangi bir alım/satım işlemi yapılmadı.")
            
        st.success("✅ Tüm analizler tamamlandı ve Google Sheets güncellendi.")
