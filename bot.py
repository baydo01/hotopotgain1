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

warnings.filterwarnings("ignore")

# --- AYARLAR ---
# GitHub Secrets'tan okunacak
SHEET_ID = os.environ.get("SHEET_ID")
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

# =============================================================================
# 1. BAĞLANTI VE VERİ YÖNETİMİ
# =============================================================================
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # GitHub Secrets Ortamı
    json_creds = os.environ.get("GCP_CREDENTIALS")
    
    creds = None
    if json_creds:
        try:
            creds_dict = json.loads(json_creds)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except Exception as e: print(f"Secret Hatası: {e}")
    
    # Yerel Ortam (Test için)
    elif os.path.exists("service_account.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    
    if not creds: return None
    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        print(f"Bağlantı Hatası: {e}")
        return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, sheet
    except: return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy().astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        print("✅ Google Sheets Güncellendi.")
    except Exception as e: print(f"Kayıt Hatası: {e}")

# =============================================================================
# 2. VERİ İŞLEME VE FEATURE ENGINEERING
# =============================================================================
def apply_kalman_filter(prices):
    xhat = np.zeros(len(prices)); P = np.zeros(len(prices)); xhatminus = np.zeros(len(prices)); Pminus = np.zeros(len(prices)); K = np.zeros(len(prices)); Q = 1e-5; R = 0.01**2
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, len(prices)):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k]/(Pminus[k]+R); xhat[k] = xhatminus[k]+K[k]*(prices.iloc[k]-xhatminus[k]); P[k] = (1-K[k])*Pminus[k]
    return pd.Series(xhat, index=prices.index)

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
    else: df_res=df.copy()
    if len(df_res)<60: return None
    
    # Feature Engineering
    df_res['kalman'] = apply_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['close']/df_res['close'].shift(1))
    
    # Volatilite Rejimi (ATR Bazlı)
    hl = df_res['high'] - df_res['low']
    tr = np.max(pd.concat([hl, np.abs(df_res['high'] - df_res['close'].shift())], axis=1), axis=1)
    atr = tr.rolling(14).mean()
    df_res['vol_regime'] = (atr / atr.rolling(50).mean()).fillna(1.0)
    
    # Basit Momentum
    df_res['momentum'] = (np.sign(df_res['close'].diff(5)) + np.sign(df_res['close'].diff(20)))
    
    # Trend Filtresi için SMA
    df_res['sma_50'] = df_res['close'].rolling(50).mean()
    df_res['trend_up'] = (df_res['close'] > df_res['sma_50']).astype(int)

    # Target (Gelecek)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res['ret'] = df_res['close'].pct_change() # Simülasyon için
    
    # Temizlik
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(subset=['target'], inplace=True)
    
    return df_res

# --- SMART IMPUTATION (Fixed Leakage) ---
def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0), "Simple-Zero"
    imputers = {'KNN': KNNImputer(n_neighbors=5), 'Mean': SimpleImputer(strategy='mean')}
    best_score = -999; best_df = df.fillna(0); best_m = "Zero"
    tr = df.iloc[:-20]; val = df.iloc[-20:]
    
    for name, imp in imputers.items():
        try:
            X_tr = imp.fit_transform(tr[features]); X_val = imp.transform(val[features])
            rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42).fit(X_tr, tr['target'])
            s = rf.score(X_val, val['target'])
            if s > best_score: 
                best_score=s; best_m=name
                full_imp = imp.fit_transform(df[features])
                best_df = pd.DataFrame(full_imp, columns=features, index=df.index)
                for c in df.columns: 
                    if c not in features and c != 'target': best_df[c] = df[c]
        except: continue
    return best_df, best_m

# --- EKONOMETRİK MODELLER (Hızlı & Güvenli) ---
def estimate_arima_models(prices):
    returns = np.log(prices/prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    try:
        model = pm.auto_arima(returns, seasonal=False, stepwise=True, trace=False, error_action='ignore', suppress_warnings=True, scoring='aic')
        return float(model.predict(n_periods=1)[0])
    except: return 0.0

def estimate_garch_vol(returns):
    if len(returns) < 100: return 0.0
    try:
        am = arch_model(100*returns, vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(disp='off')
        return float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100)
    except: return 0.0

def ga_optimize(df, features):
    return {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}

# --- ANA MOTOR (Meta-Learner) ---
def train_meta_learner(df, params):
    test_size = 60
    if len(df) < 150: return 0.0, None
    
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    features = ['log_ret', 'vol_regime', 'momentum'] 
    
    X_tr = train[features].replace([np.inf, -np.inf], np.nan).fillna(0); y_tr = train['target']
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if X_tr.empty: return 0.0, None

    # Level 1
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss').fit(X_tr, y_tr)
    
    # Ekonometrik Sinyaller
    arima_sig = estimate_arima_models(train['close'])
    garch_sig = estimate_garch_vol(train['log_ret'].dropna())
    
    # HMM
    scaler_hmm = StandardScaler()
    try:
        X_hmm = scaler_hmm.fit_transform(train[['log_ret']])
        hmm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=50).fit(X_hmm)
        hmm_probs = hmm.predict_proba(X_hmm)
    except: hmm_probs = np.zeros((len(train),2))
    
    # Meta-Data
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'ET': et.predict_proba(X_tr)[:,1],
        'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'HMM_0': hmm_probs[:,0],
        'ARIMA': np.full(len(train), arima_sig),
        'GARCH': np.full(len(train), garch_sig)
    }, index=train.index).fillna(0)
    
    # Level 2 XGBoost
    meta_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05).fit(meta_X, y_tr)
    
    # Test Sinyalleri
    arima_t = estimate_arima_models(test['close'])
    garch_t = estimate_garch_vol(test['log_ret'].dropna())
    try:
        X_hmm_t = scaler_hmm.transform(test[['log_ret']].fillna(0))
        hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),2))
    except: hmm_probs_t = np.zeros((len(test),2))
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'ET': et.predict_proba(X_test)[:,1],
        'XGB': xgb_c.predict_proba(X_test)[:,1],
        'HMM_0': hmm_probs_t[:,0],
        'ARIMA': np.full(len(test), arima_t),
        'GARCH': np.full(len(test), garch_t)
    }, index=test.index).fillna(0)
    
    probs_ens = meta_model.predict_proba(mx_test)[:,1]
    probs_xgb = xgb_c.predict_proba(X_test)[:,1]
    
    # Simülasyon (Trend Filtreli)
    sim_ens=[100]; sim_solo=[100]; p0=test['close'].iloc[0]
    
    for i in range(len(test)):
        ret = test['ret'].iloc[i]
        trend_up = test['trend_up'].iloc[i] == 1
        sell_thresh = -0.3 if trend_up else -0.1
        
        # Ensemble
        pos_ens = np.tanh(3 * (probs_ens[i]-0.5)*2)
        if pos_ens > 0.2: sim_ens.append(sim_ens[-1]*(1+ret*abs(pos_ens)))
        elif pos_ens < sell_thresh: sim_ens.append(sim_ens[-1]) # Satış
        else: sim_ens.append(sim_ens[-1]) # Bekle
        
        # Solo
        pos_solo = np.tanh(3 * (probs_xgb[i]-0.5)*2)
        if pos_solo > 0.2: sim_solo.append(sim_solo[-1]*(1+ret*abs(pos_solo)))
        elif pos_solo < sell_thresh: sim_solo.append(sim_solo[-1])
        else: sim_solo.append(sim_solo[-1])
        
    roi_ens = sim_ens[-1]-100; roi_solo = sim_solo[-1]-100
    
    if roi_solo > roi_ens:
        sig = (probs_xgb[-1]-0.5)*2
        if test['trend_up'].iloc[-1]==1 and -0.3 < sig < -0.1: sig=0.0
        return sig, {'bot_roi': roi_solo, 'method': 'Solo XGB'}
    else:
        sig = (probs_ens[-1]-0.5)*2
        if test['trend_up'].iloc[-1]==1 and -0.3 < sig < -0.1: sig=0.0
        return sig, {'bot_roi': roi_ens, 'method': 'Ensemble'}

# --- ÇALIŞTIRMA MANTIĞI ---
def run_bot_logic():
    print(f"🚀 Bot Başlatılıyor... {datetime.now()}")
    pf_df, sheet = load_and_fix_portfolio()
    if pf_df.empty: return

    updated = pf_df.copy()
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    total_cash = updated['Nakit_Bakiye_USD'].sum()
    signals = []
    
    for i, (idx, row) in enumerate(updated.iterrows()):
        ticker = row['Ticker']
        if len(str(ticker))<3: continue
        print(f"🧠 {ticker}...")
        
        raw_df = get_raw_data(ticker)
        if raw_df is None: continue
        current_p = float(raw_df['close'].iloc[-1])
        best_roi = -9999; final_sig = 0; winning_tf = "GÜNLÜK"
        
        for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
            df_raw = process_data(raw_df, tf_code)
            if df_raw is None: continue
            df_imp, _ = smart_impute(df_raw, ['log_ret', 'vol_regime', 'momentum'])
            sig, info = train_meta_learner(df_imp, ga_optimize(df_imp, []))
            
            if info and info['bot_roi'] > best_roi:
                best_roi = info['bot_roi']
                final_sig = sig
                winning_tf = tf_name
        
        signals.append({'idx':idx, 'ticker':ticker, 'price':current_p, 'signal':final_sig, 'roi':best_roi, 'tf':winning_tf, 'status':row['Durum'], 'amount':float(row['Miktar'])})
        print(f"   > {ticker}: ROI {best_roi:.2f}% ({winning_tf})")

    # Satış
    for s in signals:
        if s['status']=='COIN' and s['signal'] < 0.2:
            rev=s['amount']*s['price']; total_cash+=rev
            updated.at[s['idx'],'Durum']='CASH'; updated.at[s['idx'],'Miktar']=0.0
            updated.at[s['idx'],'Nakit_Bakiye_USD']=0.0
            updated.at[s['idx'],'Son_Islem_Log']=f"SAT ({s['tf']})"
            updated.at[s['idx'],'Son_Islem_Zamani']=time_str
            print(f"🔻 SATILDI: {s['ticker']}")
            
    # Alım (Orantılı)
    buy_cands = [s for s in signals if s['signal']>0.2 and s['roi']>0]
    total_roi = sum([s['roi'] for s in buy_cands])
    
    if buy_cands and total_cash>1.0:
        for w in buy_cands:
            weight = w['roi']/total_roi
            amt_usd = total_cash * weight
            if updated.at[w['idx'],'Durum']=='CASH':
                amt = amt_usd/w['price']
                updated.at[w['idx'],'Durum']='COIN'; updated.at[w['idx'],'Miktar']=amt
                updated.at[w['idx'],'Nakit_Bakiye_USD']=0.0
                updated.at[w['idx'],'Son_Islem_Fiyati']=w['price']
                updated.at[w['idx'],'Son_Islem_Log']=f"AL (%{weight*100:.0f})"
                updated.at[w['idx'],'Son_Islem_Zamani']=time_str
                print(f"🚀 ALINDI: {w['ticker']} (Pay: %{weight*100:.0f})")
    elif total_cash>0:
        f = updated.index[0]
        updated.at[f,'Nakit_Bakiye_USD'] += total_cash
        for ix in updated.index:
            if ix!=f and updated.at[ix,'Durum']=='CASH': updated.at[ix,'Nakit_Bakiye_USD']=0.0

    # Değerleme
    for idx, row in updated.iterrows():
        p = next((s['price'] for s in signals if s['idx']==idx), 0.0)
        if p>0: updated.at[idx,'Kaydedilen_Deger_USD'] = (float(updated.at[idx,'Miktar'])*p) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])

    save_portfolio(updated, sheet)
    print("🏁 Bitti.")

if __name__ == "__main__":
    run_bot_logic()
