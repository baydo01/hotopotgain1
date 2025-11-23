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

SHEET_ID = os.environ.get("SHEET_ID") 
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

# --- BAĞLANTI ---
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    json_creds = os.environ.get("GCP_CREDENTIALS")
    creds = None
    if json_creds:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(json_creds), scope)
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
        df_export = df.copy().astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        print("✅ Google Sheets Güncellendi.")
    except Exception as e: print(f"Kayıt Hatası: {e}")

# --- VERİ İŞLEME ---
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
    elif timeframe=='M': df_res=df.resample('ME').agg(agg)
    else: df_res=df.copy()
    if len(df_res)<100: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change()
    df_res['avg_ret_5m'] = df_res['ret'].rolling(100).mean()*100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(750).mean()*100
    
    avg_feats = df_res[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    df_res['range_vol_delta'] = df_res['range'].pct_change(5)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(subset=['target'], inplace=True)
    return df_res

def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0), "Simple-Zero"
    imputers = {'KNN': KNNImputer(n_neighbors=5), 'MICE': IterativeImputer(max_iter=10, random_state=42), 'Mean': SimpleImputer(strategy='mean')}
    best_score = -999; best_df = df.fillna(0); best_m = "Zero"
    val_size = 20
    tr = df.iloc[:-val_size]; val = df.iloc[-val_size:]
    y_tr = tr['target']; y_val = val['target']
    for name, imp in imputers.items():
        try:
            X_tr_imp = imp.fit_transform(tr[features])
            X_val_imp = imp.transform(val[features])
            rf = RandomForestClassifier(n_estimators=10, max_depth=3).fit(X_tr_imp, y_tr)
            s = rf.score(X_val_imp, y_val)
            if s > best_score:
                best_score = s; best_m = name
                full_imp = imp.fit_transform(df[features])
                best_df = pd.DataFrame(full_imp, columns=features, index=df.index)
                for c in df.columns: 
                    if c not in features: best_df[c] = df[c]
        except: continue
    return best_df, best_m

# --- MODEL FONKSİYONLARI ---
def select_best_garch_model(returns):
    returns = returns.copy()
    if len(returns) < 200: return 0.0
    models = {'GARCH': {'p':1,'o':0,'q':1}, 'GJR': {'p':1,'o':1,'q':1}}
    best_aic=np.inf; best_f=0.0
    for n, p in models.items():
        try:
            am = arch_model(100*returns, vol='GARCH', p=p['p'], o=p['o'], q=p['q'], dist='StudentsT')
            res = am.fit(disp='off')
            lb_p = acorr_ljungbox(res.resid**2, lags=[10], return_df=True)['lb_pvalue'].iloc[-1]
            if res.aic < best_aic and lb_p > 0.05:
                best_aic = res.aic; best_f = np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100
        except: continue
    return float(best_forecast) if best_forecast else 0.0

def estimate_arch_garch_models(returns): return select_best_garch_model(returns)

def estimate_arima_models(prices, is_sarima=False):
    returns = np.log(prices/prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    try:
        model = pm.auto_arima(returns, seasonal=is_sarima, m=5 if is_sarima else 1, stepwise=True, trace=False, error_action='ignore', suppress_warnings=True, scoring='aic')
        lb_p = acorr_ljungbox(model.resid(), lags=[10], return_df=True)['lb_pvalue'].iloc[-1]
        if lb_p < 0.05: return 0.0
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

# --- KAPSAMLI TARAMA (Exhaustive Search) ---
def exhaustive_search_optimize(df, features):
    """1'den 10'a kadar tüm öğrenme seviyelerini test eder."""
    test_size = 30
    train = df.iloc[:-test_size]; val = df.iloc[-test_size:]
    
    X_tr = train[features].replace([np.inf, -np.inf], np.nan).fillna(0); y_tr = train['target']
    X_val = val[features].replace([np.inf, -np.inf], np.nan).fillna(0); y_val = val['target']
    
    if X_tr.empty: return {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}, 0

    best_overall_score = -999
    best_overall_params = {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}
    best_gen = 5
    
    # Kapsamlı Tarama: 1, 3, 5, 8, 10
    for gen in [1, 3, 5, 8, 10]:
        # Bu seviye için RF
        b_rf = -999; p_rf = {'d':5, 'n':100}
        for d in [3, 5, 7]:
            m = RandomForestClassifier(n_estimators=gen*20, max_depth=d, random_state=42).fit(X_tr, y_tr)
            if m.score(X_val, y_val) > b_rf: b_rf=m.score(X_val, y_val); p_rf={'d':d, 'n':gen*20}
        
        # Bu seviye için XGB
        b_xgb = -999; p_xgb = {'d':3, 'lr':0.1, 'n':100}
        for d in [3, 5]:
            m = xgb.XGBClassifier(n_estimators=gen*20, max_depth=d, eval_metric='logloss').fit(X_tr, y_tr)
            if m.score(X_val, y_val) > b_xgb: b_xgb=m.score(X_val, y_val); p_xgb={'d':d, 'lr':0.1, 'n':gen*20}
            
        lvl_score = (b_rf + b_xgb) / 2
        if lvl_score > best_overall_score:
            best_overall_score = lvl_score
            best_overall_params = {'rf': p_rf, 'xgb': p_xgb}
            best_gen = gen
            
    return best_overall_params, best_gen

def train_meta_learner(df, params):
    test_size = 60
    if len(df) < test_size + 50: return 0.0, None, {}
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr = train[features].replace([np.inf, -np.inf], np.nan).fillna(0); y_tr = train['target']
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    arima_ret = estimate_arima_models(train['close'], False)
    sarima_ret = estimate_arima_models(train['close'], True)
    nnar_ret = estimate_nnar_models(train['log_ret'].dropna())
    garch_ret = estimate_arch_garch_models(train['log_ret'].dropna())
    
    scaler_vol = StandardScaler()
    try: garch_signal = float(-np.sign(scaler_vol.fit_transform(np.array(train['range'].values).reshape(-1, 1)).flatten()[-1]))
    except: garch_signal = 0.0

    rf = RandomForestClassifier(n_estimators=params['rf']['n'], max_depth=params['rf']['d']).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=params['rf']['n'], max_depth=params['rf']['d']).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(n_estimators=params['xgb']['n'], max_depth=params['xgb']['d']).fit(X_tr, y_tr)
    xgb_solo = xgb.XGBClassifier(n_estimators=params['xgb']['n'], max_depth=params['xgb']['d'], learning_rate=0.1).fit(X_tr, y_tr)
    
    scaler_hmm = StandardScaler()
    try:
        X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']].replace([np.inf, -np.inf], np.nan).fillna(0))
        hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
        hmm_probs = hmm.predict_proba(X_hmm)
    except: hmm_probs = np.zeros((len(train),3)); hmm = None
    
    hmm_df = pd.DataFrame(hmm_probs, columns=['HMM_0','HMM_1','HMM_2'], index=train.index)
    
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1], 'ETC': etc.predict_proba(X_tr)[:,1], 'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'Heuristic': train['heuristic'], 'HMM_0': hmm_df['HMM_0'], 'HMM_1': hmm_df['HMM_1'], 'HMM_2': hmm_df['HMM_2'],
        'ARIMA': np.full(len(train), arima_ret), 'SARIMA': np.full(len(train), sarima_ret),
        'NNAR': np.full(len(train), nnar_ret), 'GARCH': np.full(len(train), garch_ret), 'VolSig': np.full(len(train), garch_signal)
    }, index=train.index).fillna(0)
    
    scaler_meta = StandardScaler()
    try: meta_X_sc = scaler_meta.fit_transform(meta_X)
    except: meta_X_sc = meta_X.values
    
    meta_model = LogisticRegression(C=1.0).fit(meta_X_sc, y_tr)
    weights = meta_model.coef_[0]
    
    # Test Sinyalleri
    arima_t = estimate_arima_models(test['close'], False)
    sarima_t = estimate_arima_models(test['close'], True)
    nnar_t = estimate_nnar_models(test['log_ret'].dropna())
    garch_t = estimate_arch_garch_models(test['log_ret'].dropna())
    
    try:
        X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']].replace([np.inf, -np.inf], np.nan).fillna(0))
        hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
    except: hmm_probs_t = np.zeros((len(test),3))
    hmm_df_t = pd.DataFrame(hmm_probs_t, columns=['HMM_0','HMM_1','HMM_2'], index=test.index)
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1], 'ETC': etc.predict_proba(X_test)[:,1], 'XGB': xgb_c.predict_proba(X_test)[:,1],
        'Heuristic': test['heuristic'], 'HMM_0': hmm_df_t['HMM_0'], 'HMM_1': hmm_df_t['HMM_1'], 'HMM_2': hmm_df_t['HMM_2'],
        'ARIMA': np.full(len(test), arima_t), 'SARIMA': np.full(len(test), sarima_t),
        'NNAR': np.full(len(test), nnar_t), 'GARCH': np.full(len(test), garch_t), 'VolSig': np.full(len(test), garch_signal)
    }, index=test.index).fillna(0)
    
    try: mx_test_sc = scaler_meta.transform(mx_test)
    except: mx_test_sc = mx_test.values
    
    probs_ens = meta_model.predict_proba(mx_test_sc)[:,1]
    probs_xgb = xgb_solo.predict_proba(X_test)[:,1]
    
    sim_ens=[100]; sim_xgb=[100]; sim_hodl=[100]; p0=test['close'].iloc[0]
    for i in range(len(test)):
        p=test['close'].iloc[i]; ret=test['ret'].iloc[i]
        se=(probs_ens[i]-0.5)*2; sx=(probs_xgb[i]-0.5)*2
        if se>0.1: sim_ens.append(sim_ens[-1]*(1+ret))
        else: sim_ens.append(sim_ens[-1])
        
        if sx>0.1: sim_xgb.append(sim_xgb[-1]*(1+ret))
        else: sim_xgb.append(sim_xgb[-1])
        
        sim_hodl.append((100/p0)*p)
        
    roi_ens = sim_ens[-1]-100; roi_xgb = sim_xgb[-1]-100
    
    if roi_xgb > roi_ens:
        return (probs_xgb[-1]-0.5)*2, {'bot_roi': roi_xgb, 'method': 'Solo XGBoost', 'weights': weights, 'sim_ens': sim_ens, 'sim_xgb': sim_xgb, 'sim_hodl': sim_hodl, 'dates': test.index}
    else:
        return (probs_ens[-1]-0.5)*2, {'bot_roi': roi_ens, 'method': 'Ensemble', 'weights': weights, 'sim_ens': sim_ens, 'sim_xgb': sim_xgb, 'sim_hodl': sim_hodl, 'dates': test.index}

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
        
        raw_df = get_raw_data(ticker)
        if raw_df is None: continue
        
        current_p = float(raw_df['close'].iloc[-1])
        best_roi = -9999; final_sig = 0; winning_tf = "GÜNLÜK"; imp_method = "-"; best_gen = 0
        
        for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
            df_raw = process_data(raw_df, tf_code)
            if df_raw is None: continue
            
            feats = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
            df_imp, method_imp = smart_impute(df_raw, feats)
            
            best_params, best_g = exhaustive_search_optimize(df_imp, feats)
            sig, info = train_meta_learner(df_imp, best_params)
            
            if info and info['bot_roi'] > best_roi:
                best_roi = info['bot_roi']
                final_sig = sig
                winning_tf = tf_name
                imp_method = method_imp
                best_gen = best_g
        
        signals.append({'idx':idx, 'ticker':ticker, 'price':current_p, 'signal':final_sig, 'roi':best_roi, 'tf':winning_tf, 'method':imp_method, 'status':row['Durum'], 'amount':float(row['Miktar'])})
        print(f"{ticker}: ROI {best_roi:.2f}% (Gen {best_gen})")

    # Orantılı Dağıtım Mantığı
    # 1. Tüm nakiti topla (Satışlar dahil)
    for s in signals:
        if s['status']=='COIN' and s['signal']<-0.1:
            rev = s['amount']*s['price']; total_cash+=rev
            updated.at[s['idx'],'Durum']='CASH'; updated.at[s['idx'],'Miktar']=0.0
            updated.at[s['idx'],'Nakit_Bakiye_USD']=0.0
            updated.at[s['idx'],'Son_Islem_Log']=f"SAT ({s['tf']})"
            updated.at[s['idx'],'Son_Islem_Zamani']=time_str

    # 2. Pozitif ROI'si olanları bul
    buy_cands = [s for s in signals if s['signal']>0.1 and s['roi']>0]
    total_positive_roi = sum([s['roi'] for s in buy_cands])
    
    if buy_cands and total_cash > 1.0:
        for cand in buy_cands:
            # Ağırlık hesapla: ROI / Toplam_ROI
            weight = cand['roi'] / total_positive_roi
            allocation = total_cash * weight
            
            if updated.at[cand['idx'],'Durum'] == 'CASH':
                amt = allocation / cand['price']
                updated.at[cand['idx'],'Durum']='COIN'; updated.at[cand['idx'],'Miktar']=amt
                updated.at[cand['idx'],'Nakit_Bakiye_USD']=0.0
                updated.at[cand['idx'],'Son_Islem_Fiyati']=cand['price']
                updated.at[cand['idx'],'Son_Islem_Log']=f"AL (Pay: %{weight*100:.1f})"
                updated.at[cand['idx'],'Son_Islem_Zamani']=time_str
    elif total_cash > 0:
        # Park Et
        f_idx = updated.index[0]
        updated.at[f_idx,'Nakit_Bakiye_USD'] += total_cash
        for ix in updated.index:
            if ix!=f_idx and updated.at[ix,'Durum']=='CASH': updated.at[ix,'Nakit_Bakiye_USD']=0.0

    # Değerleme
    for idx, row in updated.iterrows():
        p = next((s['price'] for s in signals if s['idx']==idx), 0.0)
        if p>0: updated.at[idx,'Kaydedilen_Deger_USD'] = (float(updated.at[idx,'Miktar'])*p) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])
        
    save_portfolio(updated, sheet)
    print("🏁 Bitti.")

if __name__ == "__main__":
    run_bot_logic()
