import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
import json
import logging
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- AĞIR SİKLET ML KÜTÜPHANELERİ ---
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from arch import arch_model
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import xgboost as xgb

# Uyarıları kapat
warnings.filterwarnings("ignore")

# =============================================================================
# 1. KONFİGÜRASYON & LOGLAMA
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE" # Senin Sheet ID'n
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d" # Model Eğitimi için veri uzunluğu

# Loglama Ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_activity.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# =============================================================================
# 2. BAĞLANTI KATMANI (GOOGLE SHEETS)
# =============================================================================
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        if os.path.exists(CREDENTIALS_FILE):
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        else:
            logger.error(f"'{CREDENTIALS_FILE}' dosyası bulunamadı!")
            return None
        
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        logger.error(f"Google Sheets Bağlantı Hatası: {e}")
        return None

def load_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        # Sayısal Dönüşümler
        cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, sheet
    except Exception as e:
        logger.error(f"Portföy Okuma Hatası: {e}")
        return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_exp = df.copy().astype(str)
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())
        logger.info("✅ Google Sheets Başarıyla Güncellendi.")
    except Exception as e:
        logger.error(f"Kayıt Hatası: {e}")

# =============================================================================
# 3. FEATURE ENGINEERING (ULTIMATE V5 CORE)
# =============================================================================
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

def get_data(ticker):
    try:
        # Ticker düzeltmesi (boşluk vs varsa)
        ticker = ticker.strip()
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"{ticker} verisi çekilemedi: {e}")
        return None

def process_data_advanced(df):
    if df is None or len(df)<150: return None
    df = df.copy()
    
    # Kalman
    df['kalman_close'] = apply_kalman_filter(df['close'].fillna(method='ffill'))
    
    # Returns
    df['log_ret'] = np.log(df['kalman_close']/df['kalman_close'].shift(1))
    df['ret'] = df['close'].pct_change()
    
    # Volatility
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    
    # Heuristic
    df['heuristic'] = calculate_heuristic_score(df)
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    
    # Target
    df['target'] = (df['close'].shift(-1)>df['close']).astype(int)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['target'], inplace=True)
    return df

def smart_impute(df, features):
    if len(df) < 50: return df.fillna(0)
    imputer = KNNImputer(n_neighbors=5)
    try:
        df_imputed = df.copy()
        df_imputed[features] = imputer.fit_transform(df[features])
        return df_imputed
    except: return df.fillna(0)

# =============================================================================
# 4. QUANT MODELLERİ (OPTIMIZED FOR BOT)
# =============================================================================
def estimate_arima_models(prices, is_sarima=False):
    returns = np.log(prices/prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    try:
        model = pm.auto_arima(returns, seasonal=is_sarima, m=5 if is_sarima else 1, 
                              stepwise=True, trace=False, error_action='ignore', suppress_warnings=True)
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

def estimate_garch_vol(returns):
    if len(returns) < 200: return 0.0
    try:
        am = arch_model(100*returns, vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(disp='off')
        return float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100)
    except: return 0.0

# =============================================================================
# 5. BRAIN & RISK ENGINE
# =============================================================================
class HedgeFundBrain:
    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0)
        
    def train_predict(self, df):
        features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        df = smart_impute(df, features)
        
        # Son 1 bar test, kalanı train
        test_size = 1
        train = df.iloc[:-test_size]
        test = df.iloc[-test_size:]
        
        X_tr = train[features]; y_tr = train['target']
        X_test = test[features]
        
        # Base Models
        rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3).fit(X_tr, y_tr)
        
        # HMM
        scaler_hmm = StandardScaler()
        try:
            X_hmm = scaler_hmm.fit_transform(train[['log_ret', 'range_vol_delta']].fillna(0))
            hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
            hmm_probs = hmm.predict_proba(X_hmm)
        except: hmm_probs = np.zeros((len(train), 3)); hmm = None
        
        # Meta-Features
        meta_X_tr = pd.DataFrame({
            'RF': rf.predict_proba(X_tr)[:,1],
            'ETC': etc.predict_proba(X_tr)[:,1],
            'XGB': xgb_c.predict_proba(X_tr)[:,1],
            'Heuristic': train['heuristic'],
            'HMM_0': hmm_probs[:,0], 'HMM_1': hmm_probs[:,1], 'HMM_2': hmm_probs[:,2]
        }, index=train.index).fillna(0)
        
        self.meta_model.fit(meta_X_tr, y_tr)
        
        # Test Prediction (Son bar)
        # Hız için quant modelleri sadece son 100-200 bar üzerinde çalıştırıyoruz
        arima_sig = estimate_arima_models(df['close'].iloc[-100:])
        garch_vol = estimate_garch_vol(df['log_ret'].dropna().iloc[-200:])
        
        try:
            X_hmm_t = scaler_hmm.transform(test[['log_ret', 'range_vol_delta']].fillna(0))
            hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else [[0,0,0]]
        except: hmm_probs_t = [[0,0,0]]
        
        meta_X_test = pd.DataFrame({
            'RF': rf.predict_proba(X_test)[:,1],
            'ETC': etc.predict_proba(X_test)[:,1],
            'XGB': xgb_c.predict_proba(X_test)[:,1],
            'Heuristic': test['heuristic'],
            'HMM_0': hmm_probs_t[0][0], 'HMM_1': hmm_probs_t[0][1], 'HMM_2': hmm_probs_t[0][2]
        }, index=test.index).fillna(0)
        
        final_prob = self.meta_model.predict_proba(meta_X_test)[0][1]
        
        return {
            'prob': final_prob,
            'garch_vol': garch_vol,
            'arima_sig': arima_sig
        }

class RiskEngine:
    def check_volatility_guard(self, df):
        today_range = (df['high'].iloc[-1] - df['low'].iloc[-1])
        avg_range = df['range'].rolling(14).mean().iloc[-1] * df['close'].iloc[-1]
        if today_range > (avg_range * 3.0): return True
        return False
        
    def calculate_position(self, prob, garch_vol):
        base_size = 0.0
        if prob > 0.80: base_size = 1.0
        elif prob > 0.65: base_size = 0.7
        elif prob > 0.55: base_size = 0.4
        
        if garch_vol > 0.05: base_size *= 0.5
        return base_size

# =============================================================================
# 6. ANA ÇALIŞMA MANTIĞI (EXECUTION LOOP)
# =============================================================================
def run_bot():
    logger.info("🤖 Hedge Fund AI Bot Başlatılıyor...")
    pf_df, sheet = load_portfolio()
    
    if pf_df.empty:
        logger.error("Portföy yüklenemedi. Bot durduruluyor.")
        return
        
    updated_pf = pf_df.copy()
    pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
    buy_orders = []
    
    brain = HedgeFundBrain()
    risk_eng = RiskEngine()
    
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    # 1. Adım: Tüm Coinleri Analiz Et ve SATIŞLARI Yap
    logger.info("📊 Piyasa analizi ve satış kontrolleri yapılıyor...")
    
    for i, (idx, row) in enumerate(updated_pf.iterrows()):
        ticker = row['Ticker']
        if pd.isna(ticker) or len(str(ticker)) < 3: continue
        
        logger.info(f"Analiz ediliyor: {ticker}")
        df = get_data(ticker)
        
        if df is not None:
            try:
                # Veri İşleme
                df = process_data_advanced(df)
                
                # Model
                res = brain.train_predict(df)
                prob = res['prob']
                
                # Risk
                is_halted = risk_eng.check_volatility_guard(df)
                pos_size = risk_eng.calculate_position(prob, res['garch_vol'])
                
                # Karar
                decision = "HOLD"
                if is_halted: decision = "HALT"
                elif prob > 0.55: decision = "BUY"
                elif prob < 0.45: decision = "SELL"
                
                logger.info(f"   -> Sinyal: {decision} | Güven: {prob:.2f} | VolGuard: {is_halted}")
                
                # --- SATIŞ MANTIĞI ---
                if row['Durum'] == 'COIN':
                    if decision == "SELL" or decision == "HALT":
                        current_val = float(row['Miktar']) * df['close'].iloc[-1]
                        pool_cash += current_val
                        
                        updated_pf.at[idx, 'Durum'] = 'CASH'
                        updated_pf.at[idx, 'Miktar'] = 0.0
                        updated_pf.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                        updated_pf.at[idx, 'Son_Islem_Log'] = f"SAT ({prob:.2f})"
                        updated_pf.at[idx, 'Son_Islem_Zamani'] = time_str
                        logger.info(f"   🚨 SATILDI: {ticker} (+${current_val:.2f})")
                
                # --- ALIM ADAYI ---
                elif row['Durum'] == 'CASH':
                    if decision == "BUY":
                        buy_orders.append({
                            'idx': idx, 'ticker': ticker, 'price': df['close'].iloc[-1],
                            'weight': pos_size, 'prob': prob
                        })
                        
            except Exception as e:
                logger.error(f"Hata ({ticker}): {e}")
                continue

    # 2. Adım: ALIMLARI Gerçekleştir (Toplanan nakit ile)
    logger.info(f"💰 Mevcut Nakit Havuzu: ${pool_cash:.2f}")
    
    if buy_orders and pool_cash > 10:
        total_weight = sum([b['weight'] for b in buy_orders])
        if total_weight > 0:
            for b in buy_orders:
                share_pct = b['weight'] / total_weight
                usd_amt = pool_cash * share_pct
                
                # İşlem
                amt = usd_amt / b['price']
                updated_pf.at[b['idx'], 'Durum'] = 'COIN'
                updated_pf.at[b['idx'], 'Miktar'] = amt
                updated_pf.at[b['idx'], 'Nakit_Bakiye_USD'] = 0.0
                updated_pf.at[b['idx'], 'Son_Islem_Fiyati'] = b['price']
                updated_pf.at[b['idx'], 'Son_Islem_Log'] = f"AL (Prob: {b['prob']:.2f})"
                updated_pf.at[b['idx'], 'Son_Islem_Zamani'] = time_str
                
                logger.info(f"   ✅ ALINDI: {b['ticker']} (${usd_amt:.2f})")
    
    elif not buy_orders and pool_cash > 0:
        # Kimse alınmadıysa nakiti ilk boş satırda veya ilk satırda tut
        # Basitlik için ilk satırın nakit bakiyesine ekle, diğerlerini sıfırla
        first_idx = updated_pf.index[0]
        updated_pf.at[first_idx, 'Nakit_Bakiye_USD'] = pool_cash
        for xi in updated_pf.index:
            if xi != first_idx and updated_pf.at[xi, 'Durum'] == 'CASH':
                updated_pf.at[xi, 'Nakit_Bakiye_USD'] = 0.0
        logger.info("   ℹ️ Alım sinyali yok. Nakit park edildi.")

    # 3. Adım: Değerleme Güncellemesi
    for idx, row in updated_pf.iterrows():
        try:
            if row['Durum'] == 'COIN':
                # Hızlı son fiyat çekimi
                p = yf.download(row['Ticker'], period="1d", progress=False)['Close'].iloc[-1]
                updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = float(row['Miktar']) * float(p)
            else:
                updated_pf.at[idx, 'Kaydedilen_Deger_USD'] = row['Nakit_Bakiye_USD']
        except: pass

    # 4. Adım: Kaydet
    save_portfolio(updated_pf, sheet)
    logger.info("🏁 Bot turu tamamlandı. Uyku moduna geçiliyor (veya kapanıyor).")

if __name__ == "__main__":
    # Bu scripti bir Task Scheduler ile çalıştıracağın için döngüye sokmadım.
    # Her çalıştırıldığında 1 tur atar ve kapanır.
    run_bot()
