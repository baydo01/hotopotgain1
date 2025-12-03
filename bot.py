import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import gspread
import logging
import time # EKLENDİ
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- ML LIBRARIES ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# CONFIG
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"
LOOP_INTERVAL_MINUTES = 15 # Normal döngü süresi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# --- CONNECT ---
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
    
    # Otomatik sütun ekleme
    if 'Son_Islem_Tarihi' not in df.columns: df['Son_Islem_Tarihi'] = "-"
    if 'Bot_Son_Kontrol' not in df.columns: df['Bot_Son_Kontrol'] = "-"
    # YENİ: Tetikleyici sütunu
    if 'Bot_Trigger' not in df.columns: df['Bot_Trigger'] = "FALSE"
    
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

# --- FEATURES & ML (Standart) ---
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
    df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
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
        return winner

def analyze_ticker(idx, row, now_str):
    brain = GrandLeagueBrain()
    ticker = row['Ticker']
    res = {'idx': idx, 'action': None, 'val': 0, 'buy': None, 'ticker': ticker}
    try:
        df = get_data(ticker)
        if df is None or len(df) < 200: return res
        hist, future = prepare_features(df)
        winner = brain.run_league(hist)
        lookback = pd.concat([hist.iloc[-50:], future])
        X_full, X_future_scaled, _ = brain.lab.apply_imputation(hist, future, winner['imputer_name'])
        y_full = hist['target']
        if winner['type'] == 'XGB':
            model = brain.tune_xgboost(X_full, y_full)
            prob = model.predict_proba(X_future_scaled)[:,1][0]
        else: 
            m_xgb = brain.tune_xgboost(X_full, y_full)
            rf = RandomForestClassifier(50, max_depth=5, n_jobs=1).fit(X_full, y_full)
            et = ExtraTreesClassifier(50, max_depth=5, n_jobs=1).fit(X_full, y_full)
            p1 = rf.predict_proba(X_future_scaled)[:,1]
            p2 = et.predict_proba(X_future_scaled)[:,1]
            p3 = m_xgb.predict_proba(X_future_scaled)[:,1]
            prob = (p1*0.3 + p2*0.3 + p3*0.4)[0]
        vol = hist['volatility_measure'].iloc[-1]
        target_vol = 0.04
        if vol==0 or np.isnan(vol): risk_factor = 1.0
        else: risk_factor = np.clip(target_vol / vol, 0.3, 2.0)
        decision = "HOLD"
        if prob > 0.58: decision = "BUY"
        elif prob < 0.42: decision = "SELL"
        log_msg = f"{winner['name']} (Acc:{winner['score']:.2f} | P:{prob:.2f} | R:{risk_factor:.2f}x)"
        res['action'] = decision; res['log'] = log_msg; res['price'] = df['close'].iloc[-1]
        if row['Durum']=='COIN' and decision=="SELL":
            res['val'] = float(row['Miktar']) * res['price']
        elif row['Durum']=='CASH' and decision=="BUY":
            res['buy'] = {'idx':idx, 'ticker':ticker, 'p':res['price'], 'w': prob * risk_factor, 'm':log_msg}
    except Exception as e: logger.error(f"Err {ticker}: {e}")
    return res

# --- MAIN LOOP (SONSUZ DÖNGÜ + UZAKTAN KUMANDA) ---
if __name__ == "__main__":
    logger.info("Bot Started V15 (Remote Control Enabled).")
    pf_sheet, _ = connect_services()
    if not pf_sheet: exit()
    
    last_run_time = datetime.now()
    
    while True:
        try:
            # 1. Sheets'i Kontrol Et (Sinyal Var mı?)
            pf_sheet, hist_sheet = connect_services()
            pf = load_portfolio(pf_sheet)
            
            # Tetikleyici Kontrolü
            manual_trigger = False
            if not pf.empty and 'Bot_Trigger' in pf.columns:
                # İlk satırdaki trigger'a bakmak yeterli
                if str(pf.iloc[0]['Bot_Trigger']).upper() == 'TRUE':
                    manual_trigger = True
                    logger.info("🚨 TETİKLEYİCİ ALGILANDI! Bot çalıştırılıyor...")
            
            # Zaman Kontrolü (15 dakika geçti mi?)
            time_diff = (datetime.now() - last_run_time).total_seconds() / 60
            should_run = manual_trigger or (time_diff >= LOOP_INTERVAL_MINUTES)
            
            if should_run:
                logger.info("--- Analiz Başlıyor ---")
                
                # --- ANA İŞLEM BLOKU ---
                updated = pf.copy()
                # Tetikleyiciyi sıfırla (Loop tekrar tetiklenmesin)
                updated['Bot_Trigger'] = "FALSE"
                
                cash = updated['Nakit_Bakiye_USD'].sum()
                updated['Nakit_Bakiye_USD'] = 0.0
                buys = []
                now_str = str(datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M'))
                updated['Bot_Son_Kontrol'] = now_str
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(analyze_ticker, idx, row, now_str): idx for idx, row in updated.iterrows()}
                    for future in as_completed(futures):
                        r = future.result()
                        idx = r['idx']
                        if r['action'] == 'SELL':
                            cash += r['val']
                            updated.at[idx,'Durum']='CASH'; updated.at[idx,'Miktar']=0.0
                            updated.at[idx,'Son_Islem_Log'] = f"SAT {r['log']}"
                            updated.at[idx,'Son_Islem_Tarihi'] = now_str
                            log_transaction(hist_sheet, r['ticker'], "SAT", updated.at[idx,'Miktar'], r['price'], r['log'])
                            logger.info(f"{r['ticker']} SATILDI.")
                        elif r['buy']:
                            buys.append(r['buy'])
                            logger.info(f"{r['ticker']} AL Adayı.")

                if buys and cash > 2.0:
                    total_w = sum([b['w'] for b in buys])
                    for b in buys:
                        share = (b['w']/total_w)*cash; amt = share/b['p']
                        updated.at[b['idx'],'Durum']='COIN'; updated.at[b['idx'],'Miktar']=amt
                        updated.at[b['idx'],'Nakit_Bakiye_USD']=0.0; updated.at[b['idx'],'Son_Islem_Fiyati']=b['p']
                        updated.at[b['idx'],'Son_Islem_Log']=f"AL {b['m']}"
                        updated.at[b['idx'],'Son_Islem_Tarihi']=now_str
                        log_transaction(hist_sheet, b['ticker'], "AL", amt, b['p'], b['m'], sheet_hist)
                elif cash > 0: updated.at[updated.index[0], 'Nakit_Bakiye_USD'] += cash
                
                for idx, row in updated.iterrows():
                    if row['Durum'] == 'COIN':
                        try:
                            p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                            updated.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
                        except: pass
                    else: updated.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
                
                save_portfolio(updated, pf_sheet)
                last_run_time = datetime.now()
                logger.info(f"✅ Tur Tamamlandı. Saat: {now_str}")
                
            else:
                # Çalışma zamanı değilse bekle
                pass

        except Exception as e:
            logger.error(f"Döngü Hatası: {e}")
            
        # Kısa uyku (Sinyal kontrolü için sık sık uyanmalı)
        # Her 30 saniyede bir Sheets'i kontrol eder
        time.sleep(30)
