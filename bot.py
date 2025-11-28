import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import gspread
import logging
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- ML & AUTOML LIBRARIES ---
from sklearn.experimental import enable_iterative_imputer # MICE için şart
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from hmmlearn.hmm import GaussianHMM
import xgboost as xgb

warnings.filterwarnings("ignore")

# CONFIG
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"

# LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# --- BAĞLANTI KATMANI ---
def connect_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        try: hist = spreadsheet.worksheet("Gecmis")
        except: hist = spreadsheet.add_worksheet("Gecmis", 1000, 6)
        return spreadsheet.sheet1, hist
    except Exception as e:
        logger.error(f"Connection Error: {e}")
        return None, None

def load_portfolio(sheet):
    if not sheet: return pd.DataFrame()
    data = sheet.get_all_records()
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    if 'Son_Islem_Tarihi' not in df.columns: df['Son_Islem_Tarihi'] = "-"
    return df

def save_portfolio(df, sheet):
    if sheet:
        df_exp = df.copy().astype(str)
        sheet.clear()
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())

def log_transaction(sheet, ticker, action, amount, price, model):
    if sheet:
        now = datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M')
        try: sheet.append_row([now, ticker, action, float(amount), float(price), model])
        except: pass

# --- AUTOML CORE (HATA KORUMALI) ---
class DataArchitect:
    def find_best_imputer(self, df, features):
        """
        Zırhlı Imputer Seçicisi: MICE/KNN hata verirse Linear'a düşer.
        """
        # Sonsuz değerleri temizle
        df = df.replace([np.inf, -np.inf], np.nan)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Test seti oluştur (%10 gizle)
        try:
            test_df = df[features].copy()
            mask = np.random.choice([True, False], size=test_df.shape, p=[0.1, 0.9])
            ground_truth = test_df.values.copy()
            sim_df = test_df.copy()
            sim_df.values[mask] = np.nan
        except:
            # Veri çok küçükse direkt Linear dön
            return df.interpolate(method='linear').fillna(method='bfill'), "Linear (Fallback)"

        results = {}
        
        # 1. MICE Testi
        try:
            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
            # Sadece feature sütunları üzerinde test ediyoruz
            filled = imp.fit_transform(sim_df)
            if filled.shape == sim_df.shape:
                results['MICE'] = np.sqrt(mean_squared_error(ground_truth[mask], filled[mask]))
            else: results['MICE'] = 999.0
        except: results['MICE'] = 999.0

        # 2. KNN Testi
        try:
            imp = KNNImputer(n_neighbors=5)
            filled = imp.fit_transform(sim_df)
            if filled.shape == sim_df.shape:
                results['KNN'] = np.sqrt(mean_squared_error(ground_truth[mask], filled[mask]))
            else: results['KNN'] = 999.0
        except: results['KNN'] = 999.0
        
        # 3. Linear Testi
        try:
            filled = sim_df.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
            results['Linear'] = np.sqrt(mean_squared_error(ground_truth[mask], filled[mask]))
        except: results['Linear'] = 999.0
        
        winner = min(results, key=results.get)
        final_df = df.copy()
        
        # KAZANANI UYGULA (Hata Korumalı)
        try:
            if winner == 'MICE':
                imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10)
                filled_data = imputer.fit_transform(final_df[numeric_cols])
                if filled_data.shape[1] == len(numeric_cols): final_df[numeric_cols] = filled_data
                else: raise ValueError("MICE Shape Mismatch")
            elif winner == 'KNN':
                imputer = KNNImputer(n_neighbors=5)
                filled_data = imputer.fit_transform(final_df[numeric_cols])
                if filled_data.shape[1] == len(numeric_cols): final_df[numeric_cols] = filled_data
                else: raise ValueError("KNN Shape Mismatch")
            else:
                final_df[features] = final_df[features].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        except:
            winner = "Linear (Emergency)"
            final_df[features] = final_df[features].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            final_df = final_df.fillna(0)
            
        return final_df, winner

def process_data_automl(df):
    if len(df) < 150: return None, None
    df = df.copy()
    
    # 1. Temel Temizlik
    df = df.replace([np.inf, -np.inf], np.nan)
    df.fillna(method='ffill', inplace=True)
    
    # 2. Feature Engineering
    df['kalman'] = df['close'].rolling(3).mean()
    df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
    
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    
    try:
        avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
        df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    except: df['historical_avg_score'] = 0
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    # 3. AutoML Temizlik
    architect = DataArchitect()
    df_clean, imp_winner = architect.find_best_imputer(df, features)
    
    df_clean.dropna(subset=['target'], inplace=True)
    return df_clean, imp_winner

class Brain:
    def __init__(self):
        self.meta = LogisticRegression(C=1.0)
        
    def optimize_xgboost(self, X_tr, y_tr):
        """XGBoost Grid Search"""
        depths, lrs = [3, 5, 7], [0.01, 0.1, 0.2]
        best_score, best_model, best_desc = -1, None, ""
        v = int(len(X_tr)*0.8)
        X_t, X_v, y_t, y_v = X_tr.iloc[:v], X_tr.iloc[v:], y_tr.iloc[:v], y_tr.iloc[v:]
        
        for d in depths:
            for lr in lrs:
                m = xgb.XGBClassifier(n_estimators=100, max_depth=d, learning_rate=lr, random_state=42)
                m.fit(X_t, y_t)
                s = m.score(X_v, y_v)
                if s > best_score: best_score=s; best_model=m; best_desc=f"d={d},lr={lr}"
        best_model.fit(X_tr, y_tr)
        return best_model, best_desc

    def run_tournament(self, df, imp_info):
        features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        test_size = 60
        train, test = df.iloc[:-test_size], df.iloc[-test_size:]
        X_tr, y_tr = train[features], train['target']
        X_test = test[features]
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        opt_xgb, xgb_params = self.optimize_xgboost(X_tr, y_tr)
        
        try:
            scaler = StandardScaler()
            X_hmm = scaler.fit_transform(train[['log_ret', 'range_vol_delta']].fillna(0))
            hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
            hmm_probs = hmm.predict_proba(X_hmm)[:,0]
            hmm_probs_t = hmm.predict_proba(scaler.transform(test[['log_ret', 'range_vol_delta']].fillna(0)))[:,0]
        except: hmm_probs=np.zeros(len(train)); hmm_probs_t=np.zeros(len(test))
        
        meta_X = pd.DataFrame({'RF': rf.predict_proba(X_tr)[:,1], 'ETC': etc.predict_proba(X_tr)[:,1], 
                               'XGB': opt_xgb.predict_proba(X_tr)[:,1], 'HMM': hmm_probs}, index=train.index).fillna(0)
        self.meta.fit(meta_X, y_tr)
        
        meta_X_test = pd.DataFrame({'RF': rf.predict_proba(X_test)[:,1], 'ETC': etc.predict_proba(X_test)[:,1], 
                                    'XGB': opt_xgb.predict_proba(X_test)[:,1], 'HMM': hmm_probs_t}, index=test.index).fillna(0)
        
        p_ens = self.meta.predict_proba(meta_X_test)[:,1]
        p_solo = opt_xgb.predict_proba(X_test)[:,1]
        
        sim_ens, sim_solo = 100.0, 100.0
        rets = test['close'].pct_change().fillna(0).values
        for i in range(len(test)):
            if p_ens[i]>0.55: sim_ens*=(1+rets[i])
            if p_solo[i]>0.55: sim_solo*=(1+rets[i])
            
        winner = "Solo Optimized XGB" if sim_solo > sim_ens else "Ensemble"
        prob = p_solo[-1] if winner=="Solo Optimized XGB" else p_ens[-1]
        
        return prob, winner, imp_info, xgb_params

# --- PARALEL İŞLEME YARDIMCISI ---
def analyze_single_ticker(idx, row, brain, now_str):
    ticker = row['Ticker']
    result = {'idx': idx, 'action': None, 'log': None, 'val': 0.0, 'buy_info': None, 'ticker': ticker}
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df is None or df.empty: return result
        
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        
        df_processed, imp_info = process_data_automl(df)
        if df_processed is not None:
            prob, winner, imp_method, xgb_p = brain.run_tournament(df_processed, imp_info)
            
            decision = "HOLD"
            if prob > 0.55: decision = "BUY"
            elif prob < 0.45: decision = "SELL"
            
            log_msg = f"{winner}|{imp_method}|{xgb_p}"
            result['action'] = decision
            result['log'] = log_msg
            result['current_price'] = df['close'].iloc[-1]
            result['prob'] = prob
            
            if row['Durum']=='COIN' and decision=="SELL":
                val = float(row['Miktar']) * result['current_price']
                result['val'] = val
            elif row['Durum']=='CASH' and decision=="BUY":
                result['buy_info'] = {'idx':idx, 'ticker':ticker, 'p':result['current_price'], 'w':prob, 'model':log_msg}
    except Exception as e: logger.error(f"Err {ticker}: {e}")
    return result

# --- MAIN ---
if __name__ == "__main__":
    logger.info("Bot Started V8.2 (AutoML + Turbo).")
    pf_sheet, hist_sheet = connect_services()
    pf = load_portfolio(pf_sheet)
    if pf.empty: exit()
    
    updated = pf.copy()
    cash = updated['Nakit_Bakiye_USD'].sum()
    updated['Nakit_Bakiye_USD'] = 0.0 # Sonsuz para hatası fix
    
    buys = []
    brain = Brain()
    now_str = str(datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M'))
    
    # PARALEL İŞLEME (5 Coin aynı anda)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_single_ticker, idx, row, brain, now_str): idx for idx, row in updated.iterrows()}
        
        for future in as_completed(futures):
            res = future.result()
            idx = res['idx']
            
            if res['action'] == 'SELL':
                cash += res['val']
                updated.at[idx, 'Durum'] = 'CASH'
                updated.at[idx, 'Miktar'] = 0.0
                updated.at[idx, 'Son_Islem_Log'] = f"SAT ({res['log']})"
                updated.at[idx, 'Son_Islem_Tarihi'] = now_str
                log_transaction(hist_sheet, res['ticker'], "SAT", updated.at[idx, 'Miktar'], res['current_price'], res['log'])
                logger.info(f"{res['ticker']} SATILDI.")
                
            elif res['buy_info']:
                buys.append(res['buy_info'])
                logger.info(f"{res['ticker']} AL Adayı.")

    # ALIMLAR
    if buys and cash > 2.0:
        total_w = sum([b['w'] for b in buys])
        for b in buys:
            share = (b['w']/total_w)*cash; amt = share/b['p']
            updated.at[b['idx'], 'Durum']='COIN'; updated.at[b['idx'], 'Miktar']=amt
            updated.at[b['idx'], 'Nakit_Bakiye_USD']=0.0
            updated.at[b['idx'], 'Son_Islem_Fiyati']=b['p']
            updated.at[b['idx'], 'Son_Islem_Log']=f"AL ({b['model']})"
            updated.at[b['idx'], 'Son_Islem_Tarihi']=now_str
            log_transaction(hist_sheet, b['ticker'], "AL", amt, b['p'], b['model'])

    elif cash > 0:
        updated.at[updated.index[0], 'Nakit_Bakiye_USD'] += cash
        
    # Değerleme
    for idx, row in updated.iterrows():
        if row['Durum'] == 'COIN':
            try:
                p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                updated.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
            except: pass
        else: updated.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
        
    save_portfolio(updated, pf_sheet)
