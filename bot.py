import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import gspread
import logging
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# ML Libraries
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb

warnings.filterwarnings("ignore")

# CONFIG
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"

# LOGGING
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# --- CONNECT ---
def connect_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        
        # 1. Ana Portföy Sayfası
        pf_sheet = spreadsheet.sheet1
        
        # 2. Geçmiş Log Sayfası (Yoksa oluşturur)
        try:
            hist_sheet = spreadsheet.worksheet("Gecmis")
        except:
            hist_sheet = spreadsheet.add_worksheet(title="Gecmis", rows="1000", cols="6")
            hist_sheet.append_row(["Tarih", "Ticker", "Islem", "Miktar", "Fiyat", "Model"])
            logger.info("ℹ️ 'Gecmis' sayfası oluşturuldu.")
            
        return pf_sheet, hist_sheet
    except Exception as e:
        logger.error(f"Connection Error: {e}")
        return None, None

def load_portfolio(sheet):
    if sheet is None: return pd.DataFrame()
    try:
        data = sheet.get_all_records()
        if not data: return pd.DataFrame()
        
        df = pd.DataFrame(data)
        
        # Sayısal sütunları temizle
        num_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for c in num_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        
        # İstenen Sütun: Son_Islem_Tarihi (Eskiden Zamanı idi, Tarihi yaptık)
        if 'Son_Islem_Tarihi' not in df.columns:
            df['Son_Islem_Tarihi'] = "-"
            
        return df
    except Exception as e:
        logger.error(f"Load Error: {e}")
        return pd.DataFrame()

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_exp = df.copy().fillna("")
        df_exp = df_exp.astype(str)
        sheet.clear() 
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())
        logger.info("✅ Portföy Güncellendi.")
    except Exception as e:
        logger.error(f"❌ Save Error: {e}")

def log_transaction(sheet, ticker, action, amount, price, model):
    if sheet is None: return
    try:
        now_str = datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M')
        # Satır sırası: Tarih, Ticker, İşlem, Miktar, Fiyat, Model
        sheet.append_row([now_str, ticker, action, float(amount), float(price), model])
        logger.info(f"📝 Geçmişe Loglandı: {ticker} {action}")
    except Exception as e:
        logger.error(f"Log Error: {e}")

# --- LOGIC ---
def process_data(df):
    if len(df) < 150: return None
    df = df.copy()
    # Feature Engineering
    df['kalman'] = df['close'].rolling(3).mean() 
    df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
    
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df.dropna(inplace=True)
    return df

def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0)
    try: return pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df[features]), columns=features, index=df.index)
    except: return df.fillna(0)

class Brain:
    def __init__(self):
        self.meta = LogisticRegression(C=1.0)
        
    def run_tournament(self, df):
        features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        df = smart_impute(df, features)
        
        test_size = 60
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:]
        
        X_tr = train[features]; y_tr = train['target']
        X_test = test[features]
        
        # Modeller
        rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3).fit(X_tr, y_tr)
        xgb_solo = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1).fit(X_tr, y_tr)
        
        try:
            scaler = StandardScaler()
            X_hmm = scaler.fit_transform(train[['log_ret', 'range_vol_delta']].fillna(0))
            hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
            hmm_probs = hmm.predict_proba(X_hmm)
        except: 
            hmm_probs = np.zeros((len(train),3))
            hmm = None
        
        meta_X = pd.DataFrame({
            'RF': rf.predict_proba(X_tr)[:,1], 
            'ETC': etc.predict_proba(X_tr)[:,1], 
            'XGB': xgb_c.predict_proba(X_tr)[:,1], 
            'Heuristic': train['heuristic'],
            'HMM_0': hmm_probs[:,0]
        }, index=train.index).fillna(0)
        
        self.meta.fit(meta_X, y_tr)
        
        try:
            X_hmm_t = scaler.transform(test[['log_ret', 'range_vol_delta']].fillna(0))
            hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),3))
        except: hmm_probs_t = np.zeros((len(test),3))
        
        meta_X_test = pd.DataFrame({
            'RF': rf.predict_proba(X_test)[:,1], 
            'ETC': etc.predict_proba(X_test)[:,1], 
            'XGB': xgb_c.predict_proba(X_test)[:,1], 
            'Heuristic': test['heuristic'],
            'HMM_0': hmm_probs_t[:,0]
        }, index=test.index).fillna(0)
        
        p_ens = self.meta.predict_proba(meta_X_test)[:,1]
        p_solo = xgb_solo.predict_proba(X_test)[:,1]
        
        sim_ens = 100.0; sim_solo = 100.0
        rets = test['close'].pct_change().fillna(0).values
        for i in range(len(test)):
            if p_ens[i] > 0.55: sim_ens *= (1+rets[i])
            if p_solo[i] > 0.55: sim_solo *= (1+rets[i])
            
        if sim_solo > sim_ens: return p_solo[-1], "Solo XGB"
        return p_ens[-1], "Ensemble"

# --- MAIN ---
if __name__ == "__main__":
    logger.info("Bot Started.")
    pf_sheet, hist_sheet = connect_services()
    pf = load_portfolio(pf_sheet)
    
    if pf.empty: 
        logger.error("Portfolio Empty or Connection Failed.")
        exit()
    
    updated = pf.copy()
    
    # 1. Havuz Hesabı ve Sıfırlama (Kritik Düzeltme)
    cash = updated['Nakit_Bakiye_USD'].sum()
    updated['Nakit_Bakiye_USD'] = 0.0
    
    buys = []
    brain = Brain()
    
    # Tarih formatı
    now_str = str(datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M'))
    
    # 2. ANALİZ & SATIŞ
    for i, (idx, row) in enumerate(updated.iterrows()):
        ticker = row['Ticker']
        try:
            df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
            if df.empty: continue
            
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower() for c in df.columns]
            
            df = process_data(df)
            prob, winner = brain.run_tournament(df)
            
            decision = "HOLD"
            if prob > 0.55: decision = "BUY"
            elif prob < 0.45: decision = "SELL"
            
            logger.info(f"{ticker}: {decision} ({winner}) P:{prob:.2f}")
            
            # --- SATIŞ ---
            if row['Durum'] == 'COIN' and decision == "SELL":
                val = float(row['Miktar']) * df['close'].iloc[-1]
                cash += val
                
                updated.at[idx, 'Durum'] = 'CASH'
                updated.at[idx, 'Miktar'] = 0.0
                updated.at[idx, 'Son_Islem_Log'] = f"SAT ({winner})"
                updated.at[idx, 'Son_Islem_Tarihi'] = now_str
                
                # Loglama
                log_transaction(hist_sheet, ticker, "SAT", row['Miktar'], df['close'].iloc[-1], winner)
                
            # --- ALIM LİSTESİ ---
            elif row['Durum'] == 'CASH' and decision == "BUY":
                buys.append({'idx':idx, 'ticker':ticker, 'price':df['close'].iloc[-1], 'weight':prob, 'winner':winner})
                
        except Exception as e: logger.error(f"Err {ticker}: {e}")
            
    # 3. ALIM İŞLEMİ
    if buys and cash > 2.0:
        total_w = sum([b['weight'] for b in buys])
        for b in buys:
            share = (b['weight'] / total_w) * cash
            amt = share / b['price']
            
            updated.at[b['idx'], 'Durum'] = 'COIN'
            updated.at[b['idx'], 'Miktar'] = amt
            updated.at[b['idx'], 'Nakit_Bakiye_USD'] = 0.0
            updated.at[b['idx'], 'Son_Islem_Fiyati'] = b['price']
            updated.at[b['idx'], 'Son_Islem_Log'] = f"AL ({b['winner']})"
            updated.at[b['idx'], 'Son_Islem_Tarihi'] = now_str
            
            # Loglama
            log_transaction(hist_sheet, b['ticker'], "AL", amt, b['price'], b['winner'])

    # 4. KALAN NAKİT YÖNETİMİ
    elif cash > 0:
        fidx = updated.index[0]
        current_cash_in_row = float(updated.at[fidx, 'Nakit_Bakiye_USD'])
        updated.at[fidx, 'Nakit_Bakiye_USD'] = current_cash_in_row + cash
            
    # 5. DEĞERLEME
    for idx, row in updated.iterrows():
        if row['Durum'] == 'COIN':
            try:
                td = yf.download(row['Ticker'], period="1d", interval="1m", progress=False)
                if not td.empty:
                    p = td['Close'].iloc[-1]
                    if isinstance(p, pd.Series): p = p.iloc[0]
                    updated.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
            except: pass
        else: updated.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
        
    # KAYDET
    save_portfolio(updated, pf_sheet)
