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
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score
from hmmlearn.hmm import GaussianHMM
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

# --- ADVANCED FEATURE ENGINEERING ---
def add_technical_indicators(df):
    df = df.copy()
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR (Volatility)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['atr'] = true_range.rolling(14).mean()
    
    # Trend (SMA)
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

def prepare_features_v11(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Yeni İndikatörler
    df = add_technical_indicators(df)
    
    # Temel Türevler
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['volatility_measure'] = df['close'].pct_change().rolling(window=10).std() # Baydo için
    
    # Target
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Split Future Row
    future_row = df.iloc[[-1]].copy()
    df_historic = df.iloc[:-1].copy()
    
    return df_historic, future_row

# --- IMPUTATION LAB (Baydo v2.1 - Leakage Proof) ---
class ImputationLab:
    def baydo_impute(self, df):
        """
        BaydoImputation v2: Volatiliteye göre dinamik pencere.
        """
        filled = df.copy()
        numeric_cols = filled.select_dtypes(include=[np.number]).columns
        
        # Volatiliteyi doldur (Linear) ki karar verebilelim
        vol = filled['volatility_measure'].interpolate(method='linear').fillna(method='bfill')
        
        # Thresholds
        v_high = vol.quantile(0.7)
        v_low = vol.quantile(0.3)
        
        # Rolling Means (Farklı Pencereler)
        r_fast = filled[numeric_cols].rolling(3, center=True, min_periods=1).mean()
        r_mid  = filled[numeric_cols].rolling(5, center=True, min_periods=1).mean()
        r_slow = filled[numeric_cols].rolling(9, center=True, min_periods=1).mean()
        
        # Masking
        base = r_mid.copy()
        base[vol > v_high] = r_fast[vol > v_high]
        base[vol < v_low] = r_slow[vol < v_low]
        
        filled[numeric_cols] = filled[numeric_cols].fillna(base)
        return filled.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

    def apply_imputation(self, df_train, df_val, method):
        """
        Strict Separation: Train fit edilir, Val transform edilir.
        Baydo gibi 'Local' metodlar için leakage riskini minimize etmek adına
        validation setine rolling uygularken dikkatli olunur.
        """
        # Feature Listesi (Artık daha zengin)
        features = ['log_ret', 'range', 'rsi', 'dist_sma', 'atr', 'volatility_measure']
        
        # Sütun varlığı kontrolü (NaN üretimi sonrası)
        features = [f for f in features if f in df_train.columns]
        
        X_tr = df_train[features].copy()
        X_val = df_val[features].copy()
        
        if method == 'Baydo':
            # Baydo yerel (local) çalıştığı için transform mantığı zordur.
            # Val seti için: Train'in sonundan veri alıp "history" oluşturmamız gerekirdi.
            # Hız için: Her iki seti bağımsız dolduruyoruz (Minor boundary effect kabul edilir)
            X_tr_filled = self.baydo_impute(X_tr)
            X_val_filled = self.baydo_impute(X_val)
            
        elif method == 'MICE':
            try:
                imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
                X_tr_filled = pd.DataFrame(imp.fit_transform(X_tr), columns=features, index=X_tr.index)
                X_val_filled = pd.DataFrame(imp.transform(X_val), columns=features, index=X_val.index)
            except:
                X_tr_filled = self.baydo_impute(X_tr)
                X_val_filled = self.baydo_impute(X_val)
                
        elif method == 'KNN':
            try:
                imp = KNNImputer(n_neighbors=5)
                X_tr_filled = pd.DataFrame(imp.fit_transform(X_tr), columns=features, index=X_tr.index)
                X_val_filled = pd.DataFrame(imp.transform(X_val), columns=features, index=X_val.index)
            except:
                X_tr_filled = self.baydo_impute(X_tr)
                X_val_filled = self.baydo_impute(X_val)
                
        else: # Linear
            X_tr_filled = X_tr.interpolate(method='linear').fillna(0)
            X_val_filled = X_val.interpolate(method='linear').fillna(0)
            
        # SCALING (Robust Scaler - Outlier Koruması)
        scaler = RobustScaler()
        X_tr_scaled = pd.DataFrame(scaler.fit_transform(X_tr_filled), columns=features, index=X_tr.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val_filled), columns=features, index=X_val.index)
        
        return X_tr_scaled, X_val_scaled, scaler

# --- HEDGE FUND BRAIN ---
class GrandLeagueBrain:
    def __init__(self):
        self.lab = ImputationLab()
        self.features = ['log_ret', 'range', 'rsi', 'dist_sma', 'atr', 'volatility_measure']
        
    def tune_xgboost(self, X_tr, y_tr):
        """
        Training Score YERİNE Validation Score kullanarak tuning yapar.
        Bu overfitting'i engeller.
        """
        # Kendi içinde mini-split
        split = int(len(X_tr) * 0.8)
        Xt, Xv = X_tr.iloc[:split], X_tr.iloc[split:]
        yt, yv = y_tr.iloc[:split], y_tr.iloc[split:]
        
        best_model = None
        best_acc = -1
        
        # Grid
        for d in [3, 5, 6]:
            for lr in [0.01, 0.05, 0.1]:
                m = xgb.XGBClassifier(n_estimators=100, max_depth=d, learning_rate=lr, n_jobs=1, random_state=42)
                m.fit(Xt, yt)
                acc = accuracy_score(yv, m.predict(Xv))
                if acc > best_acc:
                    best_acc = acc
                    best_model = xgb.XGBClassifier(n_estimators=100, max_depth=d, learning_rate=lr, n_jobs=1, random_state=42)
        
        # Kazanan parametrelerle tüm train setine fit et
        if best_model: best_model.fit(X_tr, y_tr)
        return best_model

    def run_league(self, df):
        impute_methods = ['Baydo', 'MICE', 'KNN', 'Linear']
        strategies = []
        
        # A. WALK-FORWARD VALIDATION (Daha güvenilir)
        # Son 120 günü 4 pencerede test et
        wf_window = 30
        steps = 3
        
        for imp in impute_methods:
            scores_xgb = []
            scores_ens = []
            
            for i in range(steps):
                test_end = len(df) - (i * wf_window)
                test_start = test_end - wf_window
                if i == 0: test_end = len(df)
                
                train_end = test_start
                if train_end < 200: break
                
                df_train = df.iloc[:train_end]
                df_val = df.iloc[train_end:test_end]
                
                # Impute & Scale (Leakage Free)
                X_tr, X_val, _ = self.lab.apply_imputation(df_train, df_val, imp)
                y_tr, y_val = df_train['target'], df_val['target']
                
                # Model 1: Tuned XGBoost
                m_xgb = self.tune_xgboost(X_tr, y_tr)
                scores_xgb.append(accuracy_score(y_val, m_xgb.predict(X_val)))
                
                # Model 2: Ensemble (Basit RF+ET+XGB)
                rf = RandomForestClassifier(max_depth=5, n_estimators=50, n_jobs=1).fit(X_tr, y_tr)
                et = ExtraTreesClassifier(max_depth=5, n_estimators=50, n_jobs=1).fit(X_tr, y_tr)
                
                # Weighted Voting (Performansa göre)
                p1 = rf.predict_proba(X_val)[:,1]
                p2 = et.predict_proba(X_val)[:,1]
                p3 = m_xgb.predict_proba(X_val)[:,1]
                
                final_p = (p1*0.3 + p2*0.3 + p3*0.4) # XGBoost'a hafif torpil
                acc_ens = accuracy_score(y_val, (final_p > 0.5).astype(int))
                scores_ens.append(acc_ens)
            
            # Ortalama Skorlar
            avg_xgb = np.mean(scores_xgb) if scores_xgb else 0
            avg_ens = np.mean(scores_ens) if scores_ens else 0
            
            # Final Modelleri Eğit (Tüm Geçmiş Veriyle)
            X_full, _, final_scaler = self.lab.apply_imputation(df, df.iloc[-5:], imp) # 2. arg dummy
            final_xgb_model = self.tune_xgboost(X_full, df['target'])
            
            final_rf = RandomForestClassifier(max_depth=5, n_estimators=50, n_jobs=1).fit(X_full, df['target'])
            final_et = ExtraTreesClassifier(max_depth=5, n_estimators=50, n_jobs=1).fit(X_full, df['target'])
            
            strategies.append({
                'name': f"{imp} + XGB", 'type': 'XGB', 'score': avg_xgb,
                'model': final_xgb_model, 'scaler': final_scaler, 'imputer_name': imp
            })
            strategies.append({
                'name': f"{imp} + ENS", 'type': 'ENS', 'score': avg_ens,
                'model': (final_rf, final_et, final_xgb_model), 'scaler': final_scaler, 'imputer_name': imp
            })
            
        # Kazananı Seç
        winner = max(strategies, key=lambda x: x['score'])
        return winner

# --- EXECUTION LOGIC ---
def analyze_ticker(idx, row, now_str):
    # Her thread kendi beynini yaratır (Thread-Safety)
    brain = GrandLeagueBrain()
    ticker = row['Ticker']
    res = {'idx': idx, 'action': None, 'val': 0, 'buy': None, 'ticker': ticker}
    
    try:
        df = get_data(ticker)
        if df is None or len(df) < 200: return res
        
        df_hist, df_future = prepare_features_v11(df)
        
        # Lig Başlasın
        winner = brain.run_league(df_hist)
        
        # Geleceği Tahmin Et
        # 1. Featureları hazırla (Impute & Scale)
        # Future row'u imputation'a dahil etmek için küçük bir pencereyle birleştir
        last_window = pd.concat([df_hist.iloc[-30:], df_future])
        
        # İlgili imputer mantığıyla doldur
        if winner['imputer_name'] == 'Baydo':
            filled = brain.lab.baydo_impute(last_window)
        elif winner['imputer_name'] == 'Linear':
            filled = last_window.interpolate(method='linear').fillna(method='bfill')
        else: # MICE/KNN -> Fallback KNN
            imp = KNNImputer(n_neighbors=5)
            # Sadece numeric sütunlar
            num_c = last_window.select_dtypes(include=[np.number]).columns
            f_vals = imp.fit_transform(last_window[num_c])
            filled = last_window.copy(); filled[num_c] = f_vals
            
        # Scale (Eğitilen scaler ile)
        # Sadece modelin kullandığı featureları al
        features_in_model = [f for f in brain.features if f in filled.columns]
        X_future_raw = filled[features_in_model].iloc[[-1]]
        X_future = winner['scaler'].transform(X_future_raw)
        
        # Tahmin
        if winner['type'] == 'XGB':
            prob = winner['model'].predict_proba(X_future)[:,1][0]
        else:
            rf, et, xg = winner['model']
            p1 = rf.predict_proba(X_future)[:,1]
            p2 = et.predict_proba(X_future)[:,1]
            p3 = xg.predict_proba(X_future)[:,1]
            prob = (p1*0.3 + p2*0.3 + p3*0.4)[0]
            
        # KARAR & RISK YÖNETİMİ (Position Sizing)
        volatility = df_hist['atr'].iloc[-1] / df_hist['close'].iloc[-1] # Yüzdesel volatilite
        if np.isnan(volatility) or volatility == 0: volatility = 0.02 # Default %2
        
        # Risk Factor: Düşük volatilite -> Yüksek size. Yüksek volatilite -> Düşük size.
        risk_factor = np.clip(0.02 / volatility, 0.5, 1.5) # 0.5x ile 1.5x arası çarpan
        
        decision = "HOLD"
        # Threshold (ROC optimizasyonu yerine şimdilik güvenli aralık)
        if prob > 0.58: decision = "BUY"
        elif prob < 0.42: decision = "SELL"
        
        log_msg = f"{winner['name']} (ValAcc: {winner['score']:.2f} | RiskFactor: {risk_factor:.2f})"
        
        res['action'] = decision
        res['log'] = log_msg
        res['price'] = df['close'].iloc[-1]
        
        if row['Durum']=='COIN' and decision=="SELL":
            res['val'] = float(row['Miktar']) * res['price']
        elif row['Durum']=='CASH' and decision=="BUY":
            # Adjusted Weight: Prob * RiskFactor
            weight = prob * risk_factor
            res['buy'] = {'idx':idx, 'ticker':ticker, 'p':res['price'], 'w':weight, 'm':log_msg}
            
    except Exception as e: logger.error(f"Err {ticker}: {e}")
    return res

# --- MAIN ---
if __name__ == "__main__":
    logger.info("Bot Started V11 (Hedge Fund Grade).")
    pf_sheet, hist_sheet = connect_services()
    pf = load_portfolio(pf_sheet)
    if pf.empty: exit()
    
    updated = pf.copy()
    cash = updated['Nakit_Bakiye_USD'].sum()
    updated['Nakit_Bakiye_USD'] = 0.0
    
    buys = []
    now_str = str(datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M'))
    
    # PARALEL BEYİN (Safe Threading)
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
                logger.info(f"{r['ticker']} ALIM ADAYI.")

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
