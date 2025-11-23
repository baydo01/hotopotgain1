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

# --- YENİ IMPUTATION KÜTÜPHANELERİ ---
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # MICE için gerekli
from sklearn.impute import IterativeImputer

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
    if timeframe=='W': df_res=df.resample('W').agg(agg) # dropna kaldırıldı, impute edilecek
    elif timeframe=='M': df_res=df.resample('ME').agg(agg)
    else: df_res=df.copy()
    
    # Yetersiz veri kontrolü
    if len(df_res)<100: return None
    
    # NaN Temizliği Öncesi Temel Hesaplamalar (NaN üretebilir)
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change()
    df_res['avg_ret_5m'] = df_res['ret'].rolling(100).mean()*100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(750).mean()*100
    
    # Avg Feats Imputation (Basit mean ile doldur, sonra gelişmiş yapılacak)
    avg_feats = df_res[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    df_res['range_vol_delta'] = df_res['range'].pct_change(5)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    
    # Sonsuzları NaN yap ki Imputer düzeltsin
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Hedef değişkendeki NaN'lar için (Gelecek bilinmediği için son satır düşer)
    df_res.dropna(subset=['target'], inplace=True)
    
    return df_res

# --- IMPUTATION VE MODEL SEÇİMİ ---

def smart_impute(df, features):
    """En iyi imputation yöntemini seçer (KNN vs MICE vs Mean)."""
    # Veri seti çok küçükse basit methoda dön
    if len(df) < 50:
        return df.fillna(0), "Simple-Zero"
        
    # Test edilecek yöntemler
    imputers = {
        'KNN': KNNImputer(n_neighbors=5),
        'MICE': IterativeImputer(max_iter=10, random_state=42),
        'Mean': SimpleImputer(strategy='mean')
    }
    
    best_score = -np.inf
    best_imputed_df = df.fillna(0) # Fallback
    best_method = "Simple-Zero"
    
    # Basit bir validasyon: RF ile hangisi daha iyi accuracy veriyor?
    # Sadece son 20 veri üzerinde test et (Hız için)
    val_size = 20
    train_raw = df.iloc[:-val_size]
    val_raw = df.iloc[-val_size:]
    
    # Hedef değişkeni ayıralım (Target impute edilmez)
    y_train = train_raw['target']
    y_val = val_raw['target']
    
    for name, imputer in imputers.items():
        try:
            # Train üzerinde fit, hem train hem val üzerinde transform
            X_train_imp = imputer.fit_transform(train_raw[features])
            X_val_imp = imputer.transform(val_raw[features])
            
            # Basit bir RF ile test et
            rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
            rf.fit(X_train_imp, y_train)
            score = rf.score(X_val_imp, y_val)
            
            if score > best_score:
                best_score = score
                best_method = name
                # Tüm veriyi en iyi yöntemle doldur
                full_data = imputer.fit_transform(df[features])
                best_imputed_df = pd.DataFrame(full_data, columns=features, index=df.index)
                # Target'ı geri ekle
                best_imputed_df['target'] = df['target']
                # Diğer sütunları da koru (close vb.)
                for col in df.columns:
                    if col not in features and col != 'target':
                        best_imputed_df[col] = df[col]
        except:
            continue
            
    return best_imputed_df, best_method

# --- EKONOMETRİK MODELLER ---
# (Bu kısımlar aynı kaldı, sadece veri girişlerinde fillna(0) yerine smart imputed data kullanılacak)

def select_best_garch_model(returns):
    returns = returns.copy()
    if len(returns) < 200: return 0.0
    models_to_test = {'GARCH': {'p':1,'o':0,'q':1}, 'GJR': {'p':1,'o':1,'q':1}}
    best_aic=np.inf; best_f=0.0
    for n, p in models_to_test.items():
        try:
            am = arch_model(100*returns, vol='GARCH', p=p['p'], o=p['o'], q=p['q'], dist='StudentsT')
            res = am.fit(disp='off')
            if res.aic < best_aic:
                best_aic = res.aic
                best_f = np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100
        except: continue
    return best_f

def estimate_arima_models(prices, is_sarima=False):
    returns = np.log(prices/prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    try:
        model = pm.auto_arima(returns, seasonal=is_sarima, m=5 if is_sarima else 1, stepwise=True, 
                              trace=False, error_action='ignore', suppress_warnings=True, scoring='aic')
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
        model = MLPRegressor(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
        model.fit(X.iloc[:-1], y.iloc[:-1])
        return float(model.predict(X.iloc[-1].values.reshape(1,-1))[0])
    except: return 0.0

def ga_optimize(df, features):
    # Smart Imputed Data üzerinde çalışır
    test_size = 30
    train = df.iloc[:-test_size]; val = df.iloc[-test_size:]
    
    best_score = -999; best_params = {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}
    
    # RF Optimize
    for d in [3, 5, 7]:
        rf = RandomForestClassifier(n_estimators=100, max_depth=d, random_state=42).fit(train[features], train['target'])
        if rf.score(val[features], val['target']) > best_score:
            best_score = rf.score(val[features], val['target'])
            best_params['rf'] = {'d':d, 'n':100}
            
    # XGB Optimize
    for d in [3, 5]:
        xgb_m = xgb.XGBClassifier(n_estimators=100, max_depth=d).fit(train[features], train['target'])
        if xgb_m.score(val[features], val['target']) > best_score:
             best_params['xgb'] = {'d':d, 'n':100}
             
    return best_params

def train_meta_learner(df, params):
    test_size = 60
    if len(df) < test_size + 50: return 0.0, None
    
    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    # --- EKONOMETRİK SİNYALLER (Ham veriden) ---
    # Imputation öncesi ham veri gerektirenler (ARIMA vb. kendi içlerinde dropna yapar)
    arima_getiri = estimate_arima_models(train['close'], False)
    sarima_getiri = estimate_arima_models(train['close'], True)
    nnar_getiri = estimate_nnar_models(train['log_ret'].dropna())
    garch_score = select_best_garch_model(train['log_ret'].dropna())
    
    # --- ML MODELLERİ (Imputed veriden) ---
    X_tr = train[features]; y_tr = train['target']
    X_test = test[features]
    
    rf = RandomForestClassifier(n_estimators=params['rf']['n'], max_depth=params['rf']['d']).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=params['rf']['n'], max_depth=params['rf']['d']).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(n_estimators=params['xgb']['n'], max_depth=params['xgb']['d']).fit(X_tr, y_tr)
    
    # HMM
    hmm_cols = ['HMM_0', 'HMM_1', 'HMM_2']
    try:
        hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_tr[['log_ret', 'range_vol_delta']])
        hmm_tr = hmm.predict_proba(X_tr[['log_ret', 'range_vol_delta']])
        hmm_te = hmm.predict_proba(X_test[['log_ret', 'range_vol_delta']])
    except:
        hmm_tr = np.zeros((len(X_tr), 3)); hmm_te = np.zeros((len(X_test), 3))
        
    # Meta-Data Hazırlığı
    def create_meta(ml_X, hmm_probs, length):
        df_meta = pd.DataFrame(index=ml_X.index)
        df_meta['RF'] = rf.predict_proba(ml_X)[:,1]
        df_meta['ETC'] = etc.predict_proba(ml_X)[:,1]
        df_meta['XGB'] = xgb_c.predict_proba(ml_X)[:,1]
        for i in range(3): df_meta[f'HMM_{i}'] = hmm_probs[:, i]
        
        # Statik Sinyalleri Yay (Basitleştirilmiş Walk-Forward)
        df_meta['ARIMA'] = arima_getiri
        df_meta['SARIMA'] = sarima_getiri
        df_meta['NNAR'] = nnar_getiri
        df_meta['GARCH'] = garch_score
        return df_meta

    meta_X_tr = create_meta(X_tr, hmm_tr, len(X_tr))
    meta_X_te = create_meta(X_test, hmm_te, len(X_test))
    
    # Meta-Learner (Standardizasyon Önemli)
    scaler_meta = StandardScaler()
    meta_X_tr_sc = scaler_meta.fit_transform(meta_X_tr)
    meta_X_te_sc = scaler_meta.transform(meta_X_te)
    
    meta_model = LogisticRegression(C=1.0).fit(meta_X_tr_sc, y_tr)
    probs = meta_model.predict_proba(meta_X_te_sc)[:,1]
    
    # ROI Simülasyonu
    sim_eq = [100]
    for i in range(len(test)):
        ret = test['ret'].iloc[i]
        sig = (probs[i]-0.5)*2
        if sig > 0.1: sim_eq.append(sim_eq[-1]*(1+ret))
        else: sim_eq.append(sim_eq[-1])
        
    return (probs[-1]-0.5)*2, {'bot_roi': sim_eq[-1]-100}

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
        
        best_roi = -9999; final_sig = 0; winning_tf = "GÜNLÜK"; imp_method = "-"
        current_p = float(raw_df['close'].iloc[-1])
        
        for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
            df_raw = process_data(raw_df, tf_code)
            if df_raw is None: continue
            
            # AKILLI IMPUTATION (Sihir Burada)
            feats = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
            df_imp, method = smart_impute(df_raw, feats)
            
            params = ga_optimize(df_imp, feats)
            sig, info = train_meta_learner(df_imp, params)
            
            if info and info['bot_roi'] > best_roi:
                best_roi = info['bot_roi']
                final_sig = sig
                winning_tf = tf_name
                imp_method = method
        
        signals.append({
            'idx':idx, 'ticker':ticker, 'price':current_p, 'signal':final_sig, 
            'roi':best_roi, 'tf':winning_tf, 'status':row['Durum'], 'amount':float(row['Miktar'])
        })
        print(f"   > Imputation: {imp_method} | ROI: {best_roi:.2f}")

    # Satış ve Alım Mantığı (Ortak Kasa)
    for s in signals:
        if s['status']=='COIN' and s['signal']<-0.1:
            rev = s['amount']*s['price']; total_cash+=rev
            updated.at[s['idx'],'Durum']='CASH'; updated.at[s['idx'],'Miktar']=0.0
            updated.at[s['idx'],'Nakit_Bakiye_USD']=0.0
            updated.at[s['idx'],'Son_Islem_Log']=f"SAT ({s['tf']})"
            updated.at[s['idx'],'Son_Islem_Zamani']=time_str
            
    buy_cands = [s for s in signals if s['signal']>0.1]
    buy_cands.sort(key=lambda x: x['roi'], reverse=True)
    
    if buy_cands and total_cash>1.0:
        w = buy_cands[0]
        if updated.at[w['idx'],'Durum']=='CASH':
            amt = total_cash/w['price']
            updated.at[w['idx'],'Durum']='COIN'; updated.at[w['idx'],'Miktar']=amt
            updated.at[w['idx'],'Nakit_Bakiye_USD']=0.0
            updated.at[w['idx'],'Son_Islem_Log']=f"AL ({w['tf']}) Lider"
            updated.at[w['idx'],'Son_Islem_Zamani']=time_str
            for ix in updated.index:
                if ix!=w['idx'] and updated.at[ix,'Durum']=='CASH': updated.at[ix,'Nakit_Bakiye_USD']=0.0
    elif total_cash>0:
        f_idx = updated.index[0]
        updated.at[f_idx,'Nakit_Bakiye_USD'] += total_cash
        for ix in updated.index:
            if ix!=f_idx and updated.at[ix,'Durum']=='CASH': updated.at[ix,'Nakit_Bakiye_USD']=0.0
            
    # Değerleme
    for idx, row in updated.iterrows():
        p = next((s['price'] for s in signals if s['idx']==idx), 0.0)
        if p>0: updated.at[idx,'Kaydedilen_Deger_USD'] = (float(updated.at[idx,'Miktar'])*p) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])
        
    save_portfolio(updated, sheet)
    print("🏁 Tur Tamamlandı.")

if __name__ == "__main__":
    run_bot_logic()
