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

# --- ANALİZ FONKSİYONLARI (Aynen Korundu - Özetlendi) ---
def apply_kalman_filter(prices):
    xhat = np.zeros(len(prices)); P = np.zeros(len(prices)); xhatminus = np.zeros(len(prices)); Pminus = np.zeros(len(prices)); K = np.zeros(len(prices)); Q = 1e-5; R = 0.01**2
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, len(prices)):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k]/(Pminus[k]+R); xhat[k] = xhatminus[k]+K[k]*(prices.iloc[k]-xhatminus[k]); P[k] = (1-K[k])*Pminus[k]
    return pd.Series(xhat, index=prices.index)

def calculate_heuristic_score(df):
    if len(df)<150: return pd.Series(0.0, index=df.index)
    return (np.sign(df['close'].pct_change(5).fillna(0)) + np.sign(df['close'].pct_change(30).fillna(0)) + np.where(df['close']>df['close'].rolling(150).mean(),1,-1) + np.where(df['close'].pct_change().rolling(20).std()<df['close'].pct_change().rolling(20).std().shift(1),1,-1) + np.sign(df['close'].diff(10).fillna(0)) + np.sign(df['close'].diff(20).fillna(0)))/6.0

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df)<150: return None
    agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    if timeframe=='W': df_res=df.resample('W').agg(agg).dropna()
    elif timeframe=='M': df_res=df.resample('ME').agg(agg).dropna()
    else: df_res=df.copy()
    if len(df_res)<100: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high']-df_res['low'])/df_res['close']
    df_res['heuristic'] = calculate_heuristic_score(df_res)
    df_res['ret'] = df_res['close'].pct_change()
    df_res['avg_ret_5m'] = df_res['ret'].rolling(100).mean()*100
    df_res['avg_ret_3y'] = df_res['ret'].rolling(750).mean()*100
    df_res['day_score'] = df_res.index.dayofweek.map(df_res.groupby(df_res.index.dayofweek)['ret'].mean().fillna(0)).fillna(0)
    
    avg_feats = df_res[['avg_ret_5m','avg_ret_3y','day_score']].fillna(0)
    if not avg_feats.empty: df_res['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    else: df_res['historical_avg_score'] = 0.0
    
    df_res['range_vol_delta'] = df_res['range'].pct_change(5).fillna(0)
    df_res['target'] = (df_res['close'].shift(-1)>df_res['close']).astype(int)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

def estimate_models(train, test):
    # Basitleştirilmiş Model Tahminleri (Hata Önleyici)
    try:
        # ARIMA
        model = pm.auto_arima(np.log(train['close']/train['close'].shift(1)).dropna(), seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', scoring='aic')
        arima_ret = float((train['close'].iloc[-1] * np.exp(model.predict(1)[0]) / train['close'].iloc[-1]) - 1.0)
    except: arima_ret = 0.0
    
    try:
        # GARCH
        am = arch_model(100 * train['log_ret'].dropna(), vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(disp='off')
        garch_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0])/100
    except: garch_vol = 0.0
    
    return arima_ret, garch_vol

def ga_optimize(df):
    return {'rf_depth': 5, 'rf_nest': 100, 'xgb_params': {'max_depth':5, 'n_estimators':100}}

def train_meta_learner(df, params):
    # Hızlı ve Güvenli Meta-Learner
    test_size=30
    if len(df)<100: return 0.0, None
    train=df.iloc[:-test_size]; test=df.iloc[-test_size:]
    
    # Basit Özellikler
    feats = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    X_tr=train[feats]; y_tr=train['target']
    X_test=test[feats]
    
    if X_tr.empty: return 0.0, None
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    etc = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=5).fit(X_tr, y_tr)
    
    # Meta-Model
    meta_X = pd.DataFrame({'RF': rf.predict_proba(X_tr)[:,1], 'ETC': etc.predict_proba(X_tr)[:,1], 'XGB': xgb_c.predict_proba(X_tr)[:,1]}, index=train.index)
    meta_model = LogisticRegression(C=1.0).fit(meta_X, y_tr)
    
    # Test
    mx_test = pd.DataFrame({'RF': rf.predict_proba(X_test)[:,1], 'ETC': etc.predict_proba(X_test)[:,1], 'XGB': xgb_c.predict_proba(X_test)[:,1]}, index=test.index)
    probs = meta_model.predict_proba(mx_test)[:,1]
    
    # ROI Hesaplama
    sim_eq = [100]; cash=100; coin=0
    for i in range(len(test)):
        p=test['close'].iloc[i]; s=(probs[i]-0.5)*2
        if s>0.1 and cash>0: coin=cash/p; cash=0
        elif s<-0.1 and coin>0: cash=coin*p; coin=0
        sim_eq.append(cash+coin*p)
        
    return (probs[-1]-0.5)*2, {'bot_roi': sim_eq[-1]-100}

# --- ANA MOTOR: ORTAK KASA MANTIĞI ---
def run_bot_logic():
    print(f"🚀 Bot Başlatılıyor... {datetime.now()}")
    pf_df, sheet = load_and_fix_portfolio()
    if pf_df.empty: return

    updated = pf_df.copy()
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    # 1. Tüm Nakit Bakiyelerini Topla (Ortak Havuz)
    total_cash_pool = updated['Nakit_Bakiye_USD'].sum()
    print(f"💰 Başlangıç Ortak Nakit Havuzu: ${total_cash_pool:.2f}")
    
    # 2. Analiz ve Sinyal Toplama
    signals = []
    
    for i, (idx, row) in enumerate(updated.iterrows()):
        ticker = row['Ticker']
        if len(str(ticker))<3: continue
        
        print(f"🔍 Analiz: {ticker}")
        raw_df = get_raw_data(ticker)
        if raw_df is None: continue
        
        current_price = float(raw_df['close'].iloc[-1])
        
        # Turnuva (En İyi Zaman Dilimi ve ROI Bulma)
        best_roi = -9999; final_sig = 0; winning_tf = "GÜNLÜK"
        
        for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
            df = process_data(raw_df, tf_code)
            if df is None: continue
            sig, info = train_meta_learner(df, ga_optimize(df))
            if info and info['bot_roi'] > best_roi:
                best_roi = info['bot_roi']
                final_sig = sig
                winning_tf = tf_name
        
        signals.append({
            'idx': idx,
            'ticker': ticker,
            'price': current_price,
            'signal': final_sig,
            'roi': best_roi,
            'tf': winning_tf,
            'status': row['Durum'],
            'amount': float(row['Miktar'])
        })

    # 3. Satış İşlemleri (Önce Nakit Yarat)
    for s in signals:
        if s['status'] == 'COIN' and s['signal'] < -0.1: # SAT Sinyali
            revenue = s['amount'] * s['price']
            total_cash_pool += revenue
            updated.at[s['idx'], 'Durum'] = 'CASH'
            updated.at[s['idx'], 'Miktar'] = 0.0
            updated.at[s['idx'], 'Nakit_Bakiye_USD'] = 0.0 # Para havuza gitti
            updated.at[s['idx'], 'Son_Islem_Fiyati'] = s['price']
            updated.at[s['idx'], 'Son_Islem_Log'] = f"SAT ({s['tf']}) Havuza Aktarıldı"
            updated.at[s['idx'], 'Son_Islem_Zamani'] = time_str
            print(f"🔻 SATILDI: {s['ticker']} -> +${revenue:.2f} Havuza Eklendi.")

    # 4. En İyi Alım Fırsatını Seç (Winner Takes All)
    # Sadece AL sinyali verenleri filtrele ve ROI'ye göre sırala
    buy_candidates = [s for s in signals if s['signal'] > 0.1]
    buy_candidates.sort(key=lambda x: x['roi'], reverse=True)
    
    print(f"💵 Dağıtılabilir Toplam Havuz: ${total_cash_pool:.2f}")

    if buy_candidates and total_cash_pool > 1.0:
        # Şimdilik EN İYİ TEK COINE tüm parayı bas (veya bölüştür)
        # Strateji: En yüksek ROI beklenen coine tüm parayı yatır.
        winner = buy_candidates[0]
        
        if updated.at[winner['idx'], 'Durum'] == 'CASH': # Eğer zaten coin değilse al
            amount_to_buy = total_cash_pool / winner['price']
            
            updated.at[winner['idx'], 'Durum'] = 'COIN'
            updated.at[winner['idx'], 'Miktar'] = amount_to_buy
            updated.at[winner['idx'], 'Nakit_Bakiye_USD'] = 0.0
            updated.at[winner['idx'], 'Son_Islem_Fiyati'] = winner['price']
            updated.at[winner['idx'], 'Son_Islem_Log'] = f"AL ({winner['tf']}) Lider"
            updated.at[winner['idx'], 'Son_Islem_Zamani'] = time_str
            
            # Diğer tüm satırların nakit bakiyesini sıfırla (Para kazanan coine gitti)
            # Not: Bu, tablodaki "Nakit_Bakiye_USD" sütununu sadece görsel takip için bırakır.
            # Gerçek bakiye matematiksel olarak winner'a aktarıldı.
            for idx in updated.index:
                if idx != winner['idx'] and updated.at[idx, 'Durum'] == 'CASH':
                    updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
            
            print(f"🚀 ALINDI: {winner['ticker']} (En İyi ROI: {winner['roi']:.1f}%) - Tutar: ${total_cash_pool:.2f}")
    else:
        # Hiçbir şey alınmazsa, para satılanlardan geldiyse "Nakit" olarak bir yere yazılmalı
        # Basitlik adına: Parayı USDT (veya ilk sıradaki coinin nakit hanesine) park et
        # Bu örnekte BTC-USD (ilk satır) nakit park yeri olsun
        first_idx = updated.index[0]
        current_parked = float(updated.at[first_idx, 'Nakit_Bakiye_USD'])
        updated.at[first_idx, 'Nakit_Bakiye_USD'] = current_parked + total_cash_pool
        # Diğerlerinin nakdini sıfırla (Havuza alındı çünkü)
        for idx in updated.index:
            if idx != first_idx and updated.at[idx, 'Durum'] == 'CASH':
                updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
        
        if total_cash_pool > 0:
            print(f"⏸️ İşlem Yok. ${total_cash_pool:.2f} Nakitte Bekliyor.")

    # 5. Değer Güncelleme (Raporlama İçin)
    for idx, row in updated.iterrows():
        current_p = get_raw_data(row['Ticker'])['close'].iloc[-1]
        if row['Durum'] == 'COIN':
            val = float(row['Miktar']) * current_p
        else:
            val = float(row['Nakit_Bakiye_USD'])
        updated.at[idx, 'Kaydedilen_Deger_USD'] = val

    save_portfolio(updated, sheet)
    print("🏁 Tur Tamamlandı.")

if __name__ == "__main__":
    run_bot_logic()
