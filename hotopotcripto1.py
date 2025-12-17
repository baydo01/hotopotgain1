import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import gspread
import logging
import time
import os
import sys
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- ML LIBRARIES ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

warnings.filterwarnings("ignore")

# CONFIG
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger()

# --- CONNECT ---
def connect_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    if not os.path.exists(CREDENTIALS_FILE):
        logger.error(f"KRİTİK HATA: {CREDENTIALS_FILE} yok! Secrets ayarını kontrol et.")
        sys.exit(1)
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        try: hist = spreadsheet.worksheet("Gecmis")
        except: hist = spreadsheet.add_worksheet("Gecmis", 1000, 6)
        logger.info("✅ Google Sheets Bağlantısı BAŞARILI.")
        return spreadsheet.sheet1, hist
    except Exception as e:
        logger.error(f"Bağlantı Hatası: {e}")
        sys.exit(1)

def load_portfolio(sheet):
    data = sheet.get_all_records()
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", "Volatilite"]
    for c in cols:
        if c in df.columns: 
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
    
    if 'Volatilite' not in df.columns: df['Volatilite'] = 0.0
    if 'Son_Islem_Tarihi' not in df.columns: df['Son_Islem_Tarihi'] = "-"
    if 'Bot_Son_Kontrol' not in df.columns: df['Bot_Son_Kontrol'] = "-"
    if 'Bot_Trigger' not in df.columns: df['Bot_Trigger'] = "FALSE"
    if 'Bot_Durum' not in df.columns: df['Bot_Durum'] = "Beklemede"
    return df

def save_portfolio(df, sheet):
    if sheet:
        try:
            df_exp = df.copy().astype(str)
            sheet.clear()
            sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())
            logger.info("💾 Veriler Sheets'e kaydedildi.")
        except Exception as e: logger.error(f"Kaydetme Hatası: {e}")

def log_transaction(sheet, ticker, action, amount, price, model):
    if sheet:
        now = datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M')
        try: sheet.append_row([now, str(ticker), action, float(amount), float(price), str(model)])
        except: pass

# --- ML & FEATURES (GÜNCELLENDİ) ---
def get_data(ticker):
    try: 
        time.sleep(0.5) 
        # yfinance düzeltmesi: auto_adjust=True bazen veri yapısını basitleştirir
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False, auto_adjust=True)
        
        # MultiIndex Kontrolü ve Düzeltmesi (KRİTİK KISIM)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        if df.empty:
            return None
            
        return df
    except Exception as e: 
        logger.error(f"{ticker} verisi çekilirken hata: {e}")
        return None

def prepare_features(df):
    df = df.copy().replace([np.inf, -np.inf], np.nan)
    
    # Sütun isimlerini standartlaştırma
    df.columns = [c.capitalize() for c in df.columns]
    
    # Gerekli sütun kontrolü
    required_cols = ['Close', 'High', 'Low']
    if not all(col in df.columns for col in required_cols):
        return None, None

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    
    df['rsi'] = 100 - (100 / (1 + gain/loss))
    df['atr'] = (df['High']-df['Low']).rolling(14).mean()
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['dist_sma'] = (df['Close'] - df['sma_20']) / df['sma_20']
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['range'] = (df['High'] - df['Low']) / df['Close']
    df['volatility_measure'] = df['Close'].pct_change().rolling(window=14).std()
    
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna(subset=['atr', 'rsi', 'volatility_measure'])
    
    if len(df) < 50: # Yetersiz veri kontrolü
        return None, None

    return df.iloc[:-1].copy(), df.iloc[[-1]].copy()

class ImputationLab:
    def apply_imputation(self, df_tr, df_te, method):
        features = ['log_ret', 'range', 'rsi', 'dist_sma', 'atr', 'volatility_measure']
        X_tr = df_tr[features].copy()
        X_te = df_te[features].copy()
        
        X_tr.interpolate(limit_direction='both', inplace=True)
        X_te.interpolate(limit_direction='both', inplace=True)
        
        # PANDAS GÜNCELLEMESİ: method='bfill' yerine .bfill()
        X_tr = X_tr.bfill().ffill()
        X_te = X_te.bfill().ffill()
        
        scaler = RobustScaler()
        return pd.DataFrame(scaler.fit_transform(X_tr), columns=features, index=X_tr.index), \
               pd.DataFrame(scaler.transform(X_te), columns=features, index=X_te.index)

class GrandLeagueBrain:
    def __init__(self): self.lab = ImputationLab()
    def tune_xgboost(self, X_tr, y_tr):
        m = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1, random_state=42)
        m.fit(X_tr, y_tr)
        return m
    def run_league(self, df):
        imp = 'Linear'
        split = int(len(df)*0.85)
        df_tr = df.iloc[:split]
        df_val = df.iloc[split:]
        
        X_tr, X_val = self.lab.apply_imputation(df_tr, df_val, imp)
        m_xgb = self.tune_xgboost(X_tr, df_tr['target'])
        
        # Prediction sırasında index hatasını önlemek için güvenli predict
        try:
            preds = m_xgb.predict(X_val)
            score = accuracy_score(df_val['target'], preds)
        except:
            score = 0.5
            
        return {'name': f"{imp}+XGB", 'type': 'XGB', 'score': score, 'imputer': imp}

def analyze_ticker(idx, row):
    brain = GrandLeagueBrain()
    res = {'idx': idx, 'action': None, 'val': 0, 'buy': None, 'ticker': row['Ticker'], 'vol': 0.0}
    try:
        ticker_name = str(row['Ticker']).strip()
        if "USD" not in ticker_name: return res
        
        df = get_data(ticker_name)
        if df is None or len(df) < 200: 
            # logger.warning(f"{ticker_name} verisi yetersiz veya boş.")
            return res
        
        hist, future = prepare_features(df)
        if hist is None: return res # Özellik çıkarma başarısızsa dön

        winner = brain.run_league(hist)
        
        X_full, X_fut_sc = brain.lab.apply_imputation(hist, future, winner['imputer'])
        model = brain.tune_xgboost(X_full, hist['target'])
        
        prob = model.predict_proba(X_fut_sc)[:,1][0]
        
        vol = hist['volatility_measure'].iloc[-1]
        res['vol'] = vol
        risk = np.clip(0.04/vol, 0.3, 2.0) if vol > 0 else 1.0
        
        buy_threshold = 0.62 if vol > 0.05 else 0.58
        sell_threshold = 0.42
        
        decision = "HOLD"
        if prob > buy_threshold: decision = "BUY"
        elif prob < sell_threshold: decision = "SELL"
        
        log = f"{winner['name']} (P:{prob:.2f}|R:{risk:.2f}x)"
        current_price = df['Close'].iloc[-1]
        
        res.update({'action': decision, 'log': log, 'price': current_price})
        
        if row['Durum']=='COIN' and decision=="SELL":
            res['val'] = float(row['Miktar']) * res['price']
        elif row['Durum']=='CASH' and decision=="BUY":
            res['buy'] = {'idx':idx, 'ticker':ticker_name, 'p':current_price, 'w': prob*risk, 'm':log}
            
    except Exception as e: 
        logger.error(f"Err {row['Ticker']}: {e}")
    return res

if __name__ == "__main__":
    logger.info("🚀 CLOUD BOT BAŞLATILIYOR (V18.3 Fixed)...")
    try:
        pf_sheet, hist_sheet = connect_services()
        pf = load_portfolio(pf_sheet)
        if pf.empty: 
            logger.error("Portföy boş geldi, işlem durduruluyor.")
            sys.exit(1)
        
        now_str = str(datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M'))
        pf['Bot_Durum'] = "☁️ Analiz Ediliyor..."
        pf['Bot_Trigger'] = "FALSE"
        pf['Bot_Son_Kontrol'] = now_str
        save_portfolio(pf, pf_sheet)
        
        logger.info("Analiz başlıyor (Sıralı Mod)...")
        updated = pf.copy()
        cash = updated['Nakit_Bakiye_USD'].sum()
        updated['Nakit_Bakiye_USD'] = 0.0
        buys = []
        
        for idx, row in updated.iterrows():
            r = analyze_ticker(idx, row)
            
            if 'vol' in r and r['vol'] > 0:
                updated.at[idx, 'Volatilite'] = r['vol']
            
            if r['action']:
                logger.info(f"Analiz: {r['ticker']} -> {r['action']} ({r.get('log', '-')}) | Fiyat: {r.get('price', 0):.4f}")
            
            if r['action'] == 'SELL':
                cash += r['val']
                updated.at[idx,'Durum']='CASH'; updated.at[idx,'Miktar']=0.0
                updated.at[idx,'Son_Islem_Log'] = f"SAT {r['log']}"
                updated.at[idx,'Son_Islem_Tarihi'] = now_str
                log_transaction(hist_sheet, r['ticker'], "SAT", updated.at[idx,'Miktar'], r['price'], r['log'])
            elif r['buy']: 
                buys.append(r['buy'])
            
            time.sleep(1)

        if buys and cash > 2.0:
            total_w = sum([b['w'] for b in buys])
            for b in buys:
                amt = ((b['w']/total_w)*cash) / b['p']
                updated.at[b['idx'],'Durum']='COIN'; updated.at[b['idx'],'Miktar']=amt
                updated.at[b['idx'],'Nakit_Bakiye_USD']=0.0; updated.at[b['idx'],'Son_Islem_Log']=f"AL {b['m']}"
                updated.at[b['idx'],'Son_Islem_Tarihi']=now_str
                log_transaction(hist_sheet, b['ticker'], "AL", amt, b['p'], b['m']) 
                logger.info(f"✅ ALIM YAPILDI: {b['ticker']} (Fiyat: {b['p']})")
        elif cash > 0: 
            idx_target = updated.index[0]
            updated.at[idx_target, 'Nakit_Bakiye_USD'] += cash
        
        # Güncel Fiyatla Değerleme
        for idx, row in updated.iterrows():
            if row['Durum'] == 'COIN':
                try:
                    df_temp = yf.download(row['Ticker'], period="1d", progress=False, auto_adjust=True)
                    if isinstance(df_temp.columns, pd.MultiIndex):
                        df_temp.columns = df_temp.columns.get_level_values(0)
                    current_p = df_temp['Close'].iloc[-1]
                    updated.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(current_p)
                    if row['Son_Islem_Fiyati'] == 0:
                         updated.at[idx, 'Son_Islem_Fiyati'] = float(current_p)
                except: pass
            else: 
                updated.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
        
        updated['Bot_Durum'] = "✅ Analiz Bitti"
        save_portfolio(updated, pf_sheet)
        logger.info("🏁 İŞLEM TAMAM.")
        
    except Exception as main_e:
        logger.error(f"ANA DÖNGÜ HATASI: {main_e}")
        # Hata durumunda durumu güncellemeye çalış
        try:
            pf['Bot_Durum'] = f"❌ HATA: {str(main_e)[:20]}"
            save_portfolio(pf, pf_sheet)
        except: pass
