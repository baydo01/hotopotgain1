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
    if 'Son_Islem_Tarihi' not in df.columns: df['Son_Islem_Tarihi'] = "-"
    if 'Bot_Son_Kontrol' not in df.columns: df['Bot_Son_Kontrol'] = "-"
    if 'Bot_Trigger' not in df.columns: df['Bot_Trigger'] = "FALSE"
    if 'Bot_Durum' not in df.columns: df['Bot_Durum'] = "Beklemede"
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

# --- FEATURES ---
def add_technical_indicators(df):
    df = df.copy()
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['atr'] = (df['high']-df['low']).rolling(14).mean()
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
    return df.iloc[:-1].copy(), df.iloc[[-1]].copy()

class ImputationLab:
    def baydo_impute(self, df):
        filled = df.copy()
        num = filled.select_dtypes(include=[np.number]).columns
        vol = filled['volatility_measure'].interpolate(method='linear').fillna(method='bfill')
        v_h = vol.quantile(0.7); v_l = vol.quantile(0.3)
        base = filled[num].rolling(5, center=True, min_periods=1).mean()
        base[vol > v_h] = filled[num].rolling(3, center=True, min_periods=1).mean()[vol > v_h]
        base[vol < v_l] = filled[num].rolling(9, center=True, min_periods=1).mean()[vol < v_l]
        filled[num] = filled[num].fillna(base)
        return filled.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    def apply_imputation(self, df_tr, df_te, method):
        features = ['log_ret', 'range', 'rsi', 'dist_sma', 'atr', 'volatility_measure']
        X_tr = df_tr[features].copy(); X_te = df_te[features].copy()
        if method == 'Baydo': X_tr = self.baydo_impute(X_tr); X_te = self.baydo_impute(X_te)
        elif method == 'Linear': X_tr.interpolate(limit_direction='both', inplace=True); X_te.interpolate(limit_direction='both', inplace=True)
        else: # Fallback KNN
             try: 
                 imp = KNNImputer(n_neighbors=5)
                 X_tr = pd.DataFrame(imp.fit_transform(X_tr), columns=features, index=X_tr.index)
                 X_te = pd.DataFrame(imp.transform(X_te), columns=features, index=X_te.index)
             except: X_tr = self.baydo_impute(X_tr); X_te = self.baydo_impute(X_te)
        scaler = RobustScaler()
        return pd.DataFrame(scaler.fit_transform(X_tr), columns=features, index=X_tr.index), \
               pd.DataFrame(scaler.transform(X_te), columns=features, index=X_te.index)

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
            m = xgb.XGBClassifier(n_estimators=80, max_depth=d, learning_rate=0.1, n_jobs=1, random_state=42)
            m.fit(Xt, yt)
            s = accuracy_score(yv, m.predict(Xv))
            if s > best_s: best_m = m; best_s = s
        if best_m: best_m.fit(X_tr, y_tr)
        return best_m
    def run_league(self, df):
        impute_methods = ['Baydo', 'Linear']
        strategies = []
        split = int(len(df)*0.85)
        df_tr = df.iloc[:split]; df_val = df.iloc[split:]
        for imp in impute_methods:
            X_tr, X_val = self.lab.apply_imputation(df_tr, df_val, imp)
            y_tr, y_val = df_tr['target'], df_val['target']
            m_xgb = self.tune_xgboost(X_tr, y_tr)
            s_xgb = accuracy_score(y_val, m_xgb.predict(X_val))
            strategies.append({'name': f"{imp}+XGB", 'type': 'XGB', 'score': s_xgb, 'imputer': imp})
        winner = max(strategies, key=lambda x: x['score'])
        return winner

def analyze_ticker(idx, row):
    brain = GrandLeagueBrain()
    res = {'idx': idx, 'action': None, 'val': 0, 'buy': None, 'ticker': row['Ticker']}
    try:
        df = get_data(row['Ticker'])
        if df is None or len(df) < 200: return res
        hist, future = prepare_features(df)
        winner = brain.run_league(hist)
        last_win = pd.concat([hist.iloc[-30:], future])
        X_full, X_fut_sc = brain.lab.apply_imputation(hist, future, winner['imputer'])
        model = brain.tune_xgboost(X_full, hist['target'])
        prob = model.predict_proba(X_fut_sc)[:,1][0]
        vol = hist['volatility_measure'].iloc[-1]
        risk = np.clip(0.04/vol, 0.3, 2.0) if vol > 0 else 1.0
        decision = "HOLD"
        if prob > 0.58: decision = "BUY"
        elif prob < 0.42: decision = "SELL"
        log = f"{winner['name']} (P:{prob:.2f}|R:{risk:.2f}x)"
        res.update({'action': decision, 'log': log, 'price': df['close'].iloc[-1]})
        if row['Durum']=='COIN' and decision=="SELL":
            res['val'] = float(row['Miktar']) * res['price']
        elif row['Durum']=='CASH' and decision=="BUY":
            res['buy'] = {'idx':idx, 'ticker':row['Ticker'], 'p':res['price'], 'w': prob*risk, 'm':log}
    except Exception as e: logger.error(f"Err {row['Ticker']}: {e}")
    return res

if __name__ == "__main__":
    logger.info("GitHub Action Triggered.")
    pf_sheet, hist_sheet = connect_services()
    if not pf_sheet: exit()
    
    pf = load_portfolio(pf_sheet)
    if pf.empty: exit()
    
    # 1. DURUM GÜNCELLE
    now_str = str(datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M'))
    pf['Bot_Durum'] = "☁️ Bulut Çalışıyor..."
    pf['Bot_Trigger'] = "FALSE" # Sinyali söndür
    pf['Bot_Son_Kontrol'] = now_str
    save_portfolio(pf, pf_sheet)
    
    # 2. İŞLEM
    updated = pf.copy()
    cash = updated['Nakit_Bakiye_USD'].sum()
    updated['Nakit_Bakiye_USD'] = 0.0
    buys = []
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_ticker, idx, row): idx for idx, row in updated.iterrows()}
        for i, future in enumerate(as_completed(futures)):
            r = future.result()
            idx = r['idx']
            if r['action'] == 'SELL':
                cash += r['val']
                updated.at[idx,'Durum']='CASH'; updated.at[idx,'Miktar']=0.0
                updated.at[idx,'Son_Islem_Log'] = f"SAT {r['log']}"
                updated.at[idx,'Son_Islem_Tarihi'] = now_str
                log_transaction(hist_sheet, r['ticker'], "SAT", updated.at[idx,'Miktar'], r['price'], r['log'])
            elif r['buy']: buys.append(r['buy'])

    # 3. ALIMLAR
    if buys and cash > 2.0:
        total_w = sum([b['w'] for b in buys])
        for b in buys:
            amt = ((b['w']/total_w)*cash) / b['p']
            updated.at[b['idx'],'Durum']='COIN'; updated.at[b['idx'],'Miktar']=amt
            updated.at[b['idx'],'Nakit_Bakiye_USD']=0.0; updated.at[b['idx'],'Son_Islem_Log']=f"AL {b['m']}"
            updated.at[b['idx'],'Son_Islem_Tarihi']=now_str
            log_transaction(hist_sheet, b['ticker'], "AL", amt, b['p'], b['m'], sheet_hist)
    elif cash > 0: updated.at[updated.index[0], 'Nakit_Bakiye_USD'] += cash
    
    # 4. DEĞERLEME
    for idx, row in updated.iterrows():
        if row['Durum'] == 'COIN':
            try:
                p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                updated.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
            except: pass
        else: updated.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
    
    updated['Bot_Durum'] = "✅ Hazır (Cloud)"
    save_portfolio(updated, pf_sheet)
    logger.info("GitHub Action Completed.")
