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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit # Walk-Forward için
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

# --- DATA PREP & FEATURES ---
def get_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def prepare_raw_features(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Teknik İndikatörler
    df['kalman'] = df['close'].rolling(3).mean()
    df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
    
    # Volatilite (BaydoImputation için gerekli)
    df['volatility'] = df['close'].pct_change().rolling(window=10).std()
    
    # Historical Stats
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Son satırı ayır
    future_row = df.iloc[[-1]].copy()
    df_historic = df.iloc[:-1].copy()
    
    return df_historic, future_row

# --- IMPUTATION LAB (Baydo v2) ---
class ImputationLab:
    def baydo_impute(self, df):
        """
        BaydoImputation (Baybebek v2):
        Volatiliteye göre dinamik pencere (Window) seçer.
        Yüksek Volatilite -> Dar Pencere (3) (Hızlı tepki)
        Düşük Volatilite  -> Geniş Pencere (7) (Yumuşatma)
        """
        filled = df.copy()
        numeric_cols = filled.select_dtypes(include=[np.number]).columns
        
        # 1. Farklı pencerelerle ortalamaları hazırla
        roll_fast = filled[numeric_cols].rolling(window=3, center=True, min_periods=1).mean()
        roll_mid  = filled[numeric_cols].rolling(window=5, center=True, min_periods=1).mean()
        roll_slow = filled[numeric_cols].rolling(window=9, center=True, min_periods=1).mean()
        
        # 2. Volatilite seviyesini belirle (Median'a göre)
        # Volatilite sütunu da boş olabilir, onu lineer doldur önce
        vol_filled = filled['volatility'].interpolate(method='linear').fillna(method='bfill')
        
        vol_high_thresh = vol_filled.quantile(0.66)
        vol_low_thresh = vol_filled.quantile(0.33)
        
        # 3. Maskeleme ve Birleştirme
        # Varsayılan: Mid
        final_fill = roll_mid.copy()
        
        # Volatilite yüksekse Fast kullan
        mask_high = vol_filled > vol_high_thresh
        final_fill[mask_high] = roll_fast[mask_high]
        
        # Volatilite düşükse Slow kullan
        mask_low = vol_filled < vol_low_thresh
        final_fill[mask_low] = roll_slow[mask_low]
        
        # 4. Boşlukları Doldur
        filled[numeric_cols] = filled[numeric_cols].fillna(final_fill)
        
        # Uç noktalar için fallback
        filled = filled.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        return filled

    def apply_imputation(self, df_train, df_test_chunk, method):
        """Train verisine göre strateji belirle, Test/Val verisine uygula"""
        features = ['log_ret', 'range', 'heuristic', 'range_vol_delta', 'avg_ret_5m', 'avg_ret_3y']
        
        # NaN koruması
        X_train = df_train[features].copy()
        X_test = df_test_chunk[features].copy()
        
        if method == 'Baydo':
            X_train_filled = self.baydo_impute(df_train)[features] # Tüm df'i gönder volatility için
            # Test setini doldururken Train'in son verileriyle birleştirmek lazım (Lookahead bias olmaması için)
            # Ama basitlik için kendi içinde dolduruyoruz (Baydo lokal çalışır)
            X_test_filled = self.baydo_impute(df_test_chunk)[features]
            
        elif method == 'MICE':
            try:
                imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
                X_train_filled = pd.DataFrame(imp.fit_transform(X_train), columns=features, index=X_train.index)
                X_test_filled = pd.DataFrame(imp.transform(X_test), columns=features, index=X_test.index)
            except: # Hata varsa Baydo'ya dön
                X_train_filled = self.baydo_impute(df_train)[features]
                X_test_filled = self.baydo_impute(df_test_chunk)[features]
        
        elif method == 'Linear':
             X_train_filled = X_train.interpolate(method='linear').fillna(0)
             X_test_filled = X_test.interpolate(method='linear').fillna(0)
             
        else: # Default KNN
             try:
                imp = KNNImputer(n_neighbors=5)
                X_train_filled = pd.DataFrame(imp.fit_transform(X_train), columns=features, index=X_train.index)
                X_test_filled = pd.DataFrame(imp.transform(X_test), columns=features, index=X_test.index)
             except:
                X_train_filled = self.baydo_impute(df_train)[features]
                X_test_filled = self.baydo_impute(df_test_chunk)[features]
        
        return X_train_filled, X_test_filled

# --- GRAND LEAGUE BRAIN (Walk-Forward & Static) ---
class GrandLeagueBrain:
    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0)
        self.lab = ImputationLab()
        self.features = ['log_ret', 'range', 'heuristic', 'range_vol_delta', 'avg_ret_5m', 'avg_ret_3y']
        
    def train_models(self, X_tr, y_tr):
        # XGBoost (Fast Tuning)
        best_xgb = None; best_score = -1
        # Hız için limitli grid
        for d in [3, 5]:
            m = xgb.XGBClassifier(n_estimators=80, max_depth=d, learning_rate=0.1, n_jobs=1, random_state=42)
            # Basit CV yerine direkt fit ediyoruz (Zaten dış döngüdeyiz)
            m.fit(X_tr, y_tr)
            score = m.score(X_tr, y_tr) # Training score (prox)
            if score > best_score: best_xgb = m; best_score = score
            
        # Ensemble
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
        
        return best_xgb, (rf, etc, best_xgb)

    def predict_ensemble(self, models, X_in):
        rf, etc, xg = models
        p1 = rf.predict_proba(X_in)[:,1]
        p2 = etc.predict_proba(X_in)[:,1]
        p3 = xg.predict_proba(X_in)[:,1]
        # Basit ortalama (Soft Voting) - Hız için LR yerine
        return (p1 + p2 + p3) / 3

    def run_grand_league(self, df):
        # 1. SETUP
        impute_methods = ['Baydo', 'MICE', 'Linear']
        strategies = [] # (Validation Mode, Imputer, Model Type, Acc)
        
        # --- A. STATIC VALIDATION (Eski Yöntem: Son %15) ---
        split_idx = int(len(df) * 0.85)
        df_train_s = df.iloc[:split_idx]
        df_val_s = df.iloc[split_idx:]
        
        for imp in impute_methods:
            X_tr, X_val = self.lab.apply_imputation(df_train_s, df_val_s, imp)
            y_tr, y_val = df_train_s['target'], df_val_s['target']
            
            # Train
            model_xgb, models_ens = self.train_models(X_tr, y_tr)
            
            # Eval XGB
            acc_xgb = accuracy_score(y_val, model_xgb.predict(X_val))
            strategies.append({'mode': 'Static', 'imp': imp, 'type': 'XGB', 'm': model_xgb, 'score': acc_xgb})
            
            # Eval Ens
            preds_ens = (self.predict_ensemble(models_ens, X_val) > 0.5).astype(int)
            acc_ens = accuracy_score(y_val, preds_ens)
            strategies.append({'mode': 'Static', 'imp': imp, 'type': 'ENS', 'm': models_ens, 'score': acc_ens})

        # --- B. WALK-FORWARD VALIDATION (Yeni Yöntem: Kayan Pencere) ---
        # Son 120 günü 4 parçaya böl (30'ar gün)
        wf_steps = 4
        wf_window = 30
        
        for imp in impute_methods:
            scores_xgb = []
            scores_ens = []
            
            # Walk Forward Loop
            for i in range(wf_steps):
                # Test aralığı: [End - (i+1)*W : End - i*W]
                test_end = len(df) - (i * wf_window)
                test_start = test_end - wf_window
                if i == 0: test_end = len(df) # Son parça sona kadar
                
                train_end = test_start
                
                if train_end < 200: break # Yetersiz veri
                
                df_tr_wf = df.iloc[:train_end]
                df_val_wf = df.iloc[train_end:test_end]
                
                X_tr, X_val = self.lab.apply_imputation(df_tr_wf, df_val_wf, imp)
                y_tr, y_val = df_tr_wf['target'], df_val_wf['target']
                
                # Train & Eval
                m_xgb, m_ens = self.train_models(X_tr, y_tr)
                scores_xgb.append(accuracy_score(y_val, m_xgb.predict(X_val)))
                scores_ens.append(accuracy_score(y_val, (self.predict_ensemble(m_ens, X_val) > 0.5).astype(int)))
            
            # Ortalama Skorlar
            avg_xgb = np.mean(scores_xgb) if scores_xgb else 0
            avg_ens = np.mean(scores_ens) if scores_ens else 0
            
            # Walk-Forward Modelleri (En son veriye göre eğitip saklayalım)
            # Final karar için tüm geçmiş veriyi kullan
            X_full, _ = self.lab.apply_imputation(df, df.iloc[-5:], imp) # Dummy test
            final_xgb, final_ens = self.train_models(X_full, df['target'])
            
            strategies.append({'mode': 'Walk-Fwd', 'imp': imp, 'type': 'XGB', 'm': final_xgb, 'score': avg_xgb})
            strategies.append({'mode': 'Walk-Fwd', 'imp': imp, 'type': 'ENS', 'm': final_ens, 'score': avg_ens})

        # 3. PICK WINNER
        winner = max(strategies, key=lambda x: x['score'])
        
        # 4. FINAL SIGNAL
        # Kazanan stratejiyi kullanarak son veriye (Future Row) tahmin üret
        # Future Row için imputation yapmamız lazım.
        # Basitlik için: Son 100 satırı al, future row'u ekle, impute et, son satırı tahmin et.
        
        return {
            'winner': winner,
            'prob': 0.0, # Bot.py için sadece winner yeterli şimdilik, detaylar Streamlit'te
            'desc': f"{winner['mode']} | {winner['imp']} | {winner['type']}"
        }

def analyze_single_ticker(idx, row, brain, now_str):
    ticker = row['Ticker']
    res_obj = {'idx': idx, 'action': None, 'log': None, 'val': 0.0, 'buy_info': None, 'ticker': ticker}
    try:
        df = get_data(ticker)
        if df is None: return res_obj
        
        df_hist, df_future = prepare_raw_features(df)
        
        # Turnuva
        res = brain.run_grand_league(df_hist)
        winner = res['winner']
        
        # Final Tahmin (Prediction)
        # Kazanan imputer ile son veriyi hazırla
        # Lookback penceresi oluştur
        lookback = pd.concat([df_hist.iloc[-50:], df_future])
        lab = ImputationLab()
        
        # Impute (Sadece son satır için)
        # Train kısmı önemli değil, sadece son satırın dolması lazım
        # Baydo/MICE lokal çalıştığı veya modele ihtiyaç duyduğu için tüm pencereyi veriyoruz
        if winner['imp'] == 'Baydo':
            filled = lab.baydo_impute(lookback)
        elif winner['imp'] == 'Linear':
            filled = lookback.interpolate(method='linear').fillna(method='bfill')
        else: # MICE/KNN (Basit fallback: KNN)
             imp = KNNImputer(n_neighbors=5)
             num_cols = lookback.select_dtypes(include=[np.number]).columns
             filled_mat = imp.fit_transform(lookback[num_cols])
             filled = lookback.copy(); filled[num_cols] = filled_mat
             
        X_final = filled[brain.features].iloc[[-1]] # Son satır (Future)
        
        if winner['type'] == 'XGB':
            prob = winner['m'].predict_proba(X_final)[:,1][0]
        else:
            prob = brain.predict_ensemble(winner['m'], X_final)[0]
            
        decision = "HOLD"
        if prob > 0.55: decision = "BUY"
        elif prob < 0.45: decision = "SELL"
        
        log_msg = f"{res['desc']} (Acc:{winner['score']:.2f})"
        
        res_obj['action'] = decision
        res_obj['log'] = log_msg
        res_obj['prob'] = prob
        res_obj['current_price'] = df['close'].iloc[-1]
        
        if row['Durum']=='COIN' and decision=="SELL":
            res_obj['val'] = float(row['Miktar']) * res_obj['current_price']
        elif row['Durum']=='CASH' and decision=="BUY":
            res_obj['buy_info'] = {'idx':idx, 'ticker':ticker, 'p':res_obj['current_price'], 'w':prob, 'm':log_msg}
            
    except Exception as e: logger.error(f"Err {ticker}: {e}")
    return res_obj

# --- MAIN ---
if __name__ == "__main__":
    logger.info("Bot Started V10 (Chronos - Baydo & WalkForward).")
    pf_sheet, hist_sheet = connect_services()
    pf = load_portfolio(pf_sheet)
    if pf.empty: exit()
    
    updated = pf.copy()
    cash = updated['Nakit_Bakiye_USD'].sum()
    updated['Nakit_Bakiye_USD'] = 0.0
    
    buys = []
    brain = GrandLeagueBrain()
    now_str = str(datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M'))
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(analyze_single_ticker, idx, row, brain, now_str): idx for idx, row in updated.iterrows()}
        for future in as_completed(futures):
            r = future.result()
            idx = r['idx']
            if r['action'] == 'SELL':
                cash += r['val']
                updated.at[idx,'Durum']='CASH'; updated.at[idx,'Miktar']=0.0
                updated.at[idx,'Son_Islem_Log'] = f"SAT ({r['log']})"
                updated.at[idx,'Son_Islem_Tarihi'] = now_str
                log_transaction(hist_sheet, r['ticker'], "SAT", updated.at[idx,'Miktar'], r['current_price'], r['log'])
                logger.info(f"{r['ticker']} SAT.")
            elif r['buy_info']:
                buys.append(r['buy_info'])
                logger.info(f"{r['ticker']} AL Adayı.")

    if buys and cash > 2.0:
        total_w = sum([b['w'] for b in buys])
        for b in buys:
            share = (b['w']/total_w)*cash; amt = share/b['p']
            updated.at[b['idx'],'Durum']='COIN'; updated.at[b['idx'],'Miktar']=amt
            updated.at[b['idx'],'Nakit_Bakiye_USD']=0.0; updated.at[b['idx'],'Son_Islem_Fiyati']=b['p']
            updated.at[b['idx'],'Son_Islem_Log']=f"AL ({b['m']})"
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
