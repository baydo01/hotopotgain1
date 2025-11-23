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

# İstatiksel ve ML Kütüphaneleri
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

warnings.filterwarnings("ignore")

# --- SABİTLER ---
SHEET_ID = os.environ.get("SHEET_ID") 
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

# --- GOOGLE SHEETS BAĞLANTISI ---
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    json_creds = os.environ.get("GCP_CREDENTIALS")
    creds = None
    if json_creds:
        try:
            creds_dict = json.loads(json_creds)
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except Exception as e: print(e)
    elif os.path.exists("service_account.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    
    if not creds: return None
    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except: return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df, sheet
    except: return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        print("✅ Google Sheets Güncellendi.")
    except Exception as e: print(f"Kayıt Hatası: {e}")

# --- ANALİZ FONKSİYONLARI ---
def apply_kalman_filter(prices):
    xhat = np.zeros(len(prices)); P = np.zeros(len(prices)); xhatminus = np.zeros(len(prices)); Pminus = np.zeros(len(prices)); K = np.zeros(len(prices)); Q = 1e-5; R = 0.01**2
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, len(prices)):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k]/(Pminus[k]+R); xhat[k] = xhatminus[k]+K[k]*(prices.iloc[k]-xhatminus[k]); P[k] = (1-K[k])*Pminus[k]
    return pd.Series(xhat, index=prices.index)

def calculate_heuristic_score(df):
    if len(df)<150: return pd.Series(0.0, index=df.index)
    return (np.sign(df['close'].pct_change(5).fillna(0)) + np.sign(df['close'].pct_change(30).fillna(0)) + np.where(df['close']>df['close'].rolling(150).mean(),1,-1) + np.where(df['close'].pct_change().rolling(20).std()<df['close'].pct_change().rolling(20).std().shift(1),1,-1) + np.sign(df['close'].diff(10).fillna(0)) + np.sign(df['close'].diff(20).fillna(0)))/6.0

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
    if timeframe=='W': df_res=df.resample('W').agg(agg).dropna()
    elif timeframe=='M': df_res=df.resample('ME').agg(agg).dropna()
    else: df_res=df.copy()
    if len(df_res)<100: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change()
    df_res['avg_ret_5m'] = df_res['ret'].rolling(100).mean()*100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(750).mean()*100
    df_res['day_score'] = df_res.index.dayofweek.map(df_res.groupby(df_res.index.dayofweek)['ret'].mean().fillna(0)).fillna(0)
    
    avg_feats = df_res[['avg_ret_5m','avg_ret_3y','day_score']].fillna(0)
    if not avg_feats.empty: df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    else: df_res['historical_avg_score'] = 0.0
    
    df_res['range_vol_delta'] = df_res['range'].pct_change(5).fillna(0)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

# --- EKONOMETRİK ---
def select_best_garch_model(returns):
    returns = returns.copy()
    if len(returns) < 200: return 0.0
    models = {'GARCH': {'vol':'GARCH','p':1,'o':0,'q':1}, 'GJR': {'vol':'GARCH','p':1,'o':1,'q':1}, 'APARCH': {'vol':'APARCH','p':1,'o':1,'q':1}}
    best_aic=np.inf; best_f=0.0
    for n, p in models.items():
        try:
            res = arch_model(100*returns, vol=p['vol'], p=p['p'], o=p['o'], q=p['q'], dist='StudentsT').fit(disp='off')
            if res.aic < best_aic:
                best_aic = res.aic
                best_f = np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100
        except: continue
    return best_f

def estimate_models(train, test):
    try:
        model = pm.auto_arima(np.log(train['close']/train['close'].shift(1)).dropna(), seasonal=False, trace=False, error_action='ignore')
        arima_ret = float((train['close'].iloc[-1] * np.exp(model.predict(1)[0]) / train['close'].iloc[-1]) - 1.0)
    except: arima_ret = 0.0
    
    try:
        garch_vol = select_best_garch_model(np.log(train['close']/train['close'].shift(1)).dropna())
    except: garch_vol = 0.0
    
    return arima_ret, garch_vol

# --- GÜÇLENDİRİLMİŞ OPTİMİZASYON (RF & XGB) ---
def ga_optimize(df):
    test_size = 30
    train = df.iloc[:-test_size]; val = df.iloc[-test_size:]
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    # 1. RF Optimize
    best_rf_score = -999; best_rf_params = {'d':5, 'n':100}
    for d in [3, 5, 7]:
        for n in [50, 100]:
            rf = RandomForestClassifier(n_estimators=n, max_depth=d, random_state=42).fit(train[features], train['target'])
            s = rf.score(val[features], val['target'])
            if s > best_rf_score: best_rf_score=s; best_rf_params={'d':d, 'n':n}
            
    # 2. XGBoost Optimize (YENİ)
    best_xgb_score = -999; best_xgb_params = {'d':3, 'lr':0.1, 'n':100}
    for d in [3, 5]:
        for lr in [0.01, 0.1, 0.2]:
            xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=d, learning_rate=lr, eval_metric='logloss').fit(train[features], train['target'])
            s = xgb_m.score(val[features], val['target'])
            if s > best_xgb_score: best_xgb_score=s; best_xgb_params={'d':d, 'lr':lr, 'n':100}
            
    return {'rf': best_rf_params, 'xgb': best_xgb_params}

def train_meta_learner_auto_select(df, params):
    test_size=30
    if len(df)<100: return 0.0, None
    train=df.iloc[:-test_size]; test=df.iloc[-test_size:]
    
    # Özellikler
    base_features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[base_features]; y_tr = train['target']
    X_test = test[base_features]
    
    if X_tr.empty: return 0.0, None
    
    # --- MODEL EĞİTİMİ ---
    
    # 1. XGBoost (Optimize Edilmiş)
    p_xgb = params['xgb']
    xgb_solo = xgb.XGBClassifier(n_estimators=p_xgb['n'], max_depth=p_xgb['d'], learning_rate=p_xgb['lr'], eval_metric='logloss').fit(X_tr, y_tr)
    
    # 2. Ensemble Üyeleri
    p_rf = params['rf']
    rf = RandomForestClassifier(n_estimators=p_rf['n'], max_depth=p_rf['d'], random_state=42).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=p_rf['n'], max_depth=p_rf['d'], random_state=42).fit(X_tr, y_tr)
    
    # HMM
    scaler_hmm = StandardScaler()
    X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm = None
    
    hmm_probs = hmm.predict_proba(X_hmm) if hmm else np.zeros((len(train),3))
    hmm_df = pd.DataFrame(hmm_probs, columns=['HMM_0','HMM_1','HMM_2'], index=train.index)
    
    # Meta-Learner Hazırlığı
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'ETC': etc.predict_proba(X_tr)[:,1],
        'XGB': xgb_solo.predict_proba(X_tr)[:,1],
        'Heuristic': train['heuristic'],
        'HMM_0': hmm_df['HMM_0'], 'HMM_1': hmm_df['HMM_1'], 'HMM_2': hmm_df['HMM_2']
    }, index=train.index).fillna(0)
    
    # Normalizasyon & Eğitim
    scaler_meta = StandardScaler()
    meta_X_scaled = scaler_meta.fit_transform(meta_X)
    meta_model = LogisticRegression(C=1.0, solver='liblinear').fit(meta_X_scaled, y_tr)
    
    # --- TEST VE SEÇİM ---
    X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']])
    hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    hmm_df_t = pd.DataFrame(hmm_probs_t, columns=['HMM_0','HMM_1','HMM_2'], index=test.index)
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'ETC': etc.predict_proba(X_test)[:,1],
        'XGB': xgb_solo.predict_proba(X_test)[:,1],
        'Heuristic': test['heuristic'],
        'HMM_0': hmm_df_t['HMM_0'], 'HMM_1': hmm_df_t['HMM_1'], 'HMM_2': hmm_df_t['HMM_2']
    }, index=test.index).fillna(0)
    
    mx_test_scaled = scaler_meta.transform(mx_test)
    
    # TAHMİNLER
    probs_ensemble = meta_model.predict_proba(mx_test_scaled)[:,1]
    probs_xgb = xgb_solo.predict_proba(X_test)[:,1]
    
    # ROI HESAPLAMA (KİM KAZANDI?)
    sim_ens=[100]; sim_xgb=[100]
    cash_e=100; coin_e=0
    cash_x=100; coin_x=0
    
    for i in range(len(test)):
        p=test['close'].iloc[i]
        # Ens
        s_e = (probs_ensemble[i]-0.5)*2
        if s_e>0.1 and cash_e>0: coin_e=cash_e/p; cash_e=0
        elif s_e<-0.1 and coin_e>0: cash_e=coin_e*p; coin_e=0
        sim_ens.append(cash_e+coin_e*p)
        # XGB
        s_x = (probs_xgb[i]-0.5)*2
        if s_x>0.1 and cash_x>0: coin_x=cash_x/p; cash_x=0
        elif s_x<-0.1 and coin_x>0: cash_x=coin_x*p; coin_x=0
        sim_xgb.append(cash_x+coin_x*p)
        
    roi_ens = sim_ens[-1]-100
    roi_xgb = sim_xgb[-1]-100
    
    # KAZANANI SEÇ
    if roi_xgb > roi_ens:
        final_signal = (probs_xgb[-1]-0.5)*2
        final_roi = roi_xgb
        method = "Solo XGBoost"
    else:
        final_signal = (probs_ensemble[-1]-0.5)*2
        final_roi = roi_ens
        method = "Ensemble"
        
    return final_signal, {'bot_roi': final_roi, 'method': method}

# --- ORTAK KASA ANA MOTOR ---
def run_bot_logic():
    print(f"🚀 Bot Başlatılıyor... {datetime.now()}")
    pf_df, sheet = load_and_fix_portfolio()
    if pf_df.empty: return

    updated = pf_df.copy()
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    total_cash_pool = updated['Nakit_Bakiye_USD'].sum()
    signals = []
    
    for i, (idx, row) in enumerate(updated.iterrows()):
        ticker = row['Ticker']
        if len(str(ticker))<3: continue
        print(f"🧠 {ticker}...")
        
        raw_df = get_raw_data(ticker)
        if raw_df is None: continue
        
        current_price = float(raw_df['close'].iloc[-1])
        best_roi = -9999; final_sig = 0; winning_tf = "GÜNLÜK"; chosen_method="-"
        
        for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
            df = process_data(raw_df, tf_code)
            if df is None: continue
            params = ga_optimize(df)
            sig, info = train_meta_learner_auto_select(df, params) # Oto Seçim
            
            if info and info['bot_roi'] > best_roi:
                best_roi = info['bot_roi']
                final_sig = sig
                winning_tf = tf_name
                chosen_method = info['method']
        
        signals.append({
            'idx': idx, 'ticker': ticker, 'price': current_price,
            'signal': 1 if final_sig > 0.1 else (-1 if final_sig < -0.1 else 0),
            'roi': best_roi, 'tf': winning_tf, 'method': chosen_method,
            'status': row['Durum'], 'amount': float(row['Miktar'])
        })

    # Satış
    for s in signals:
        if s['status'] == 'COIN' and s['signal'] == -1:
            rev = s['amount'] * s['price']
            total_cash_pool += rev
            updated.at[s['idx'], 'Durum'] = 'CASH'; updated.at[s['idx'], 'Miktar'] = 0.0
            updated.at[s['idx'], 'Nakit_Bakiye_USD'] = 0.0
            updated.at[s['idx'], 'Son_Islem_Fiyati'] = s['price']
            updated.at[s['idx'], 'Son_Islem_Log'] = f"SAT ({s['tf']}-{s['method']})"
            updated.at[s['idx'], 'Son_Islem_Zamani'] = time_str
            print(f"🔻 SAT: {s['ticker']}")

    # Alım
    buy_candidates = [s for s in signals if s['signal'] == 1]
    buy_candidates.sort(key=lambda x: x['roi'], reverse=True)
    
    if buy_candidates and total_cash_pool > 1.0:
        winner = buy_candidates[0]
        if updated.at[winner['idx'], 'Durum'] == 'CASH':
            amt = total_cash_pool / winner['price']
            updated.at[winner['idx'], 'Durum'] = 'COIN'; updated.at[winner['idx'], 'Miktar'] = amt
            updated.at[winner['idx'], 'Nakit_Bakiye_USD'] = 0.0
            updated.at[winner['idx'], 'Son_Islem_Fiyati'] = winner['price']
            updated.at[winner['idx'], 'Son_Islem_Log'] = f"AL ({winner['tf']}-{winner['method']})"
            updated.at[winner['idx'], 'Son_Islem_Zamani'] = time_str
            
            for idx in updated.index:
                if idx != winner['idx'] and updated.at[idx, 'Durum'] == 'CASH':
                    updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
            print(f"🚀 AL: {winner['ticker']} ({winner['method']})")
    
    elif total_cash_pool > 0:
        f_idx = updated.index[0]
        updated.at[f_idx, 'Nakit_Bakiye_USD'] += total_cash_pool
        for idx in updated.index:
            if idx != f_idx and updated.at[idx, 'Durum'] == 'CASH': updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0

    for idx, row in updated.iterrows():
        price = next((s['price'] for s in signals if s['idx'] == idx), 0.0)
        if price > 0:
            val = (float(updated.at[idx, 'Miktar']) * price) if updated.at[idx, 'Durum'] == 'COIN' else float(updated.at[idx, 'Nakit_Bakiye_USD'])
            updated.at[idx, 'Kaydedilen_Deger_USD'] = val

    save_portfolio(updated, sheet)
    print("🏁 Bitti.")

if __name__ == "__main__":
    run_bot_logic()
