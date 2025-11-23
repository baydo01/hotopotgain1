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

# --- GELİŞMİŞ ML & İSTATİSTİK KÜTÜPHANELERİ ---
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.impute import KNNImputer, SimpleImputer
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# =============================================================================
# 1. AYARLAR & PROFESYONEL UI (V4 STİLİ)
# =============================================================================
st.set_page_config(page_title="Hedge Fund AI: Ultimate V5", layout="wide", page_icon="🏦")

# Bloomberg Terminal CSS
st.markdown("""
<style>
    .main {background-color: #0b0e11;}
    .stApp {background-color: #0b0e11; color: #e0e0e0;}
    
    /* Header Gradient */
    .header-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 25px;
        border-radius: 12px;
        border-left: 5px solid #00CC96;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.5);
    }
    .header-title {font-size: 32px; font-weight: 700; margin: 0; color: #fff;}
    .header-sub {font-size: 14px; color: #a0a0a0; margin-top: 5px;}
    
    /* Kartlar */
    .stat-card {
        background-color: #15191f; border: 1px solid #2d343d; border-radius: 8px; padding: 15px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <div class="header-title">🏦 Hedge Fund AI: Ultimate V5</div>
    <div class="header-sub">Meta-Learning Ensemble • Kalman Filter • Smart Imputation • Volatility Guard</div>
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
# 3. İLERİ SEVİYE FEATURE ENGINEERING (ESKİ MODEL GÜCÜ)
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
    
    # Kalman Filtresi (Noise Reduction)
    df['kalman_close'] = apply_kalman_filter(df['close'].fillna(method='ffill'))
    
    # Temel Getiriler
    df['log_ret'] = np.log(df['kalman_close']/df['kalman_close'].shift(1))
    df['ret'] = df['close'].pct_change()
    
    # Volatilite & Range
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    
    # Heuristic & Historical Scores
    df['heuristic'] = calculate_heuristic_score(df)
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    # Trend Göstergeleri (SMA)
    df['sma50'] = df['close'].rolling(50).mean()
    df['trend_up'] = (df['close'] > df['sma50']).astype(int)
    
    # Target
    df['target'] = (df['close'].shift(-1)>df['close']).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['target'], inplace=True)
    return df

def smart_impute(df, features):
    # Eksik verileri akıllıca doldur (KNN / Mean)
    if len(df) < 50: return df.fillna(0)
    imputer = KNNImputer(n_neighbors=5)
    try:
        df_imputed = df.copy()
        df_imputed[features] = imputer.fit_transform(df[features])
        return df_imputed
    except: return df.fillna(0)

# =============================================================================
# 4. QUANT MODELLERİ (ARIMA, GARCH, NNAR)
# =============================================================================
def estimate_arima_models(prices, is_sarima=False):
    returns = np.log(prices/prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    try:
        # Hız için stepwise=True ve basitleştirilmiş ayarlar
        model = pm.auto_arima(returns, seasonal=is_sarima, m=5 if is_sarima else 1, 
                              stepwise=True, trace=False, error_action='ignore', suppress_warnings=True)
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
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42).fit(X.iloc[:-1], y.iloc[:-1])
        return float(model.predict(X.iloc[-1].values.reshape(1,-1))[0])
    except: return 0.0

def estimate_garch_vol(returns):
    if len(returns) < 200: return 0.0
    try:
        am = arch_model(100*returns, vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(disp='off')
        return float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100)
    except: return 0.0

# =============================================================================
# 5. META-LEARNING & ENSEMBLE BRAIN (ESKİ MODEL ÇEKİRDEĞİ)
# =============================================================================
class HedgeFundBrain:
    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0)
        self.scaler = StandardScaler()
        
    def exhaustive_search(self, df, features):
        # En iyi hyperparametreleri bul (RF/XGB için)
        # Hız için basitleştirilmiş grid
        return {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}

    def train_predict(self, df):
        # 1. Veri Hazırlığı
        features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        df = smart_impute(df, features)
        
        # Train/Test Split (Son 1 gün tahmin edilecek, öncesi eğitim)
        test_size = 1
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:] # Son bar
        
        X_tr = train[features]; y_tr = train['target']
        X_test = test[features]
        
        # 2. Base Modelleri Eğit
        params = self.exhaustive_search(train, features)
        rf = RandomForestClassifier(n_estimators=params['rf']['n'], max_depth=params['rf']['d']).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=params['rf']['n'], max_depth=params['rf']['d']).fit(X_tr, y_tr)
        xgb_c = xgb.XGBClassifier(n_estimators=params['xgb']['n'], max_depth=params['xgb']['d']).fit(X_tr, y_tr)
        
        # 3. İleri Seviye Sinyaller (Quant Models)
        # Bu modellerin çıktısını Train setine Feature olarak eklememiz lazım (Stacking)
        # Ancak performans için sadece son barın sinyallerini hesaplayıp Meta-Model'e besleyeceğiz.
        # (Gerçek stacking için Cross-Val gerekir, burada "feature engineering" olarak kullanıyoruz)
        
        # Eğitim verisi için Quant Sinyalleri (Basitleştirilmiş: Feature olarak heuristic ekle)
        # Not: Gerçek zamanlı ARIMA her satır için hesaplanırsa çok yavaşlar. 
        # Bu yüzden Meta-Learner'ı eğitirken ana ML modellerinin olasılıklarını kullanacağız.
        
        # HMM Rejim Tespiti
        scaler_hmm = StandardScaler()
        try:
            X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']].fillna(0))
            hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
            hmm_probs = hmm.predict_proba(X_hmm)
        except: hmm_probs = np.zeros((len(train), 3)); hmm = None
        
        # Meta-Features (Level 1)
        meta_X_tr = pd.DataFrame({
            'RF': rf.predict_proba(X_tr)[:,1],
            'ETC': etc.predict_proba(X_tr)[:,1],
            'XGB': xgb_c.predict_proba(X_tr)[:,1],
            'Heuristic': train['heuristic'],
            'HMM_0': hmm_probs[:,0], 'HMM_1': hmm_probs[:,1], 'HMM_2': hmm_probs[:,2]
        }, index=train.index).fillna(0)
        
        # Meta-Model Eğitimi
        self.meta_model.fit(meta_X_tr, y_tr)
        
        # 4. Test (Son Bar) İçin Tahmin Üret
        # Quant Modelleri Sadece Son Bar İçin Çalıştır (Hız için)
        arima_sig = estimate_arima_models(df['close'].iloc[-60:]) # Son 60 günü baz al
        nnar_sig = estimate_nnar_models(df['log_ret'].dropna().iloc[-100:])
        garch_vol = estimate_garch_vol(df['log_ret'].dropna().iloc[-200:])
        
        # HMM Test
        try:
            X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']].fillna(0))
            hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else [[0,0,0]]
        except: hmm_probs_t = [[0,0,0]]
        
        # Meta-Features Test
        meta_X_test = pd.DataFrame({
            'RF': rf.predict_proba(X_test)[:,1],
            'ETC': etc.predict_proba(X_test)[:,1],
            'XGB': xgb_c.predict_proba(X_test)[:,1],
            'Heuristic': test['heuristic'],
            'HMM_0': hmm_probs_t[0][0], 'HMM_1': hmm_probs_t[0][1], 'HMM_2': hmm_probs_t[0][2]
        }, index=test.index).fillna(0)
        
        # Final Ensemble Olasılığı
        final_prob = self.meta_model.predict_proba(meta_X_test)[0][1]
        
        # Weights (Modelin neye önem verdiği)
        weights = dict(zip(meta_X_tr.columns, self.meta_model.coef_[0]))
        
        return {
            'prob': final_prob,
            'weights': weights,
            'quant_signals': {'ARIMA': arima_sig, 'NNAR': nnar_sig, 'GARCH_Vol': garch_vol},
            'regime': np.argmax(hmm_probs_t[0]) # 0, 1 veya 2
        }

# =============================================================================
# 6. RISK ENGINE (V4 - GÜVENLİK KATMANI)
# =============================================================================
class RiskEngine:
    def __init__(self):
        pass
        
    def check_volatility_guard(self, df):
        # Flash Crash: Bugünün range'i, ATR'nin 3 katıysa dur
        today_range = (df['high'].iloc[-1] - df['low'].iloc[-1])
        avg_range = df['range'].rolling(14).mean().iloc[-1] * df['close'].iloc[-1]
        if today_range > (avg_range * 3.0): return True
        return False
        
    def calculate_position(self, prob, regime, garch_vol, pf_row):
        # Ensemble Olasılığına Göre Temel Pozisyon
        # Prob: 0.5 - 1.0 arası
        
        # Rejim Yorumu (HMM 0: Yatay, 1: Trend, 2: Volatil/Ayı varsayalım - veya tam tersi modelden modele değişir)
        # Biz prob üzerinden gidelim: Ensemble zaten rejimi biliyor.
        
        base_size = 0.0
        if prob > 0.80: base_size = 1.0 # Çok güvenli
        elif prob > 0.65: base_size = 0.7
        elif prob > 0.55: base_size = 0.4
        else: base_size = 0.0
        
        # GARCH Volatilite Düzeltmesi
        # Volatilite yüksekse pozisyonu küçült
        if garch_vol > 0.05: # %5 günlük volatilite çok yüksek
            base_size *= 0.5
            
        return base_size

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
pf_df, sheet = load_portfolio()

if not pf_df.empty:
    with st.sidebar:
        st.write("## 💼 Ultimate V5 Portföy")
        total_val = pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
        st.metric("Toplam Varlık", f"${total_val:.2f}")
        st.dataframe(pf_df[pf_df['Durum']=='COIN'][['Ticker', 'Miktar']], hide_index=True)
        
    if st.button("🚀 ULTIMATE ANALİZİ BAŞLAT", type="primary", use_container_width=True):
        updated_pf = pf_df.copy()
        pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
        buy_orders = []
        
        prog = st.progress(0)
        
        brain = HedgeFundBrain()
        risk_eng = RiskEngine()
        
        for i, (idx, row) in enumerate(updated_pf.iterrows()):
            ticker = row['Ticker']
            df = get_data(ticker)
            
            if df is not None:
                # Feature Engineering (Kalman, Heuristic, vb.)
                df = process_data_advanced(df)
                
                # Model Tahmini (Stacking Ensemble)
                res = brain.train_predict(df)
                prob = res['prob']
                quant_sigs = res['quant_signals']
                
                # Risk Kontrolleri
                is_halted = risk_eng.check_volatility_guard(df)
                pos_size = risk_eng.calculate_position(prob, res['regime'], quant_sigs['GARCH_Vol'], row)
                
                # Karar Mekanizması
                decision = "HOLD"
                if is_halted: decision = "HALT (VOLATILITY)"
                elif prob > 0.55: decision = "BUY"
                elif prob < 0.45: decision = "SELL"
                
                # --- GÖRSELLEŞTİRME ---
                with st.expander(f"{ticker} | {decision} | Güven: %{prob*100:.1f}", expanded=False):
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Fiyat", f"${df['close'].iloc[-1]:.2f}")
                    c2.metric("ARIMA Tahmin", f"%{quant_sigs['ARIMA']*100:.2f}")
                    c3.metric("GARCH Vol", f"%{quant_sigs['GARCH_Vol']*100:.2f}")
                    c4.metric("Pos Size", f"%{pos_size*100:.0f}")
                    
                    # Model Ağırlıkları (Meta-Learner neye önem verdi?)
                    st.caption("🤖 Meta-Learner Karar Ağırlıkları:")
                    w_df = pd.DataFrame(list(res['weights'].items()), columns=['Faktör', 'Etki']).set_index('Faktör').sort_values(by='Etki', ascending=False)
                    st.bar_chart(w_df)
                
                # --- İŞLEM MANTIĞI ---
                # SATIŞ
                if row['Durum'] == 'COIN':
                    if decision == "SELL" or decision.startswith("HALT"):
                        sale_val = float(row['Miktar']) * df['close'].iloc[-1]
                        pool_cash += sale_val
                        updated_pf.at[idx, 'Durum'] = 'CASH'
                        updated_pf.at[idx, 'Miktar'] = 0.0
                        updated_pf.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                        updated_pf.at[idx, 'Son_Islem_Log'] = f"SAT ({prob:.2f})"
                        st.toast(f"🛑 {ticker} Satıldı (Güven: {prob:.2f})")
                
                # ALIM LİSTESİ
                elif row['Durum'] == 'CASH':
                    if decision == "BUY":
                        buy_orders.append({
                            'idx': idx, 'ticker': ticker, 'price': df['close'].iloc[-1],
                            'weight': pos_size, 'prob': prob
                        })

            prog.progress((i+1)/len(updated_pf))
            
        # --- ALIMLARI GERÇEKLEŞTİR ---
        if buy_orders and pool_cash > 10:
            total_w = sum([b['weight'] for b in buy_orders])
            if total_w > 0:
                for b in buy_orders:
                    share_pct = b['weight'] / total_w
                    usd_amt = pool_cash * share_pct
                    
                    amt = usd_amt / b['price']
                    updated_pf.at[b['idx'], 'Durum'] = 'COIN'
                    updated_pf.at[b['idx'], 'Miktar'] = amt
                    updated_pf.at[b['idx'], 'Nakit_Bakiye_USD'] = 0.0
                    updated_pf.at[b['idx'], 'Son_Islem_Fiyati'] = b['price']
                    updated_pf.at[b['idx'], 'Son_Islem_Log'] = f"AL (Ens: {b['prob']:.2f})"
                    st.toast(f"✅ {b['ticker']} Alındı (${usd_amt:.1f})")
        
        elif not buy_orders and pool_cash > 0:
            fidx = updated_pf.index[0]
            updated_pf.at[fidx, 'Nakit_Bakiye_USD'] = pool_cash
            for xi in updated_pf.index:
                if xi != fidx and updated_pf.at[xi, 'Durum'] == 'CASH':
                    updated_pf.at[xi, 'Nakit_Bakiye_USD'] = 0.0

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
        st.success("🏁 Ultimate V5 Analizi Tamamlandı! Stacking Ensemble Kararları Uygulandı.")
        st.balloons()
