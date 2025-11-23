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

# Sadece Gerekli ML Kütüphaneleri
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

warnings.filterwarnings("ignore")

# --- SABİTLER ---
SHEET_ID = os.environ.get("SHEET_ID") 
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "2y" # Daha odaklı veri

# --- BAĞLANTI ---
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    json_creds = os.environ.get("GCP_CREDENTIALS")
    creds = None
    if json_creds:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(json_creds), scope)
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
        df_export = df.copy().astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        print("✅ Google Sheets Güncellendi.")
    except Exception as e: print(f"Kayıt Hatası: {e}")

# --- GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ ---

def adaptive_kalman_filter(prices):
    """Piyasa volatilitesine göre kendini ayarlayan Kalman Filtresi."""
    prices = prices.values
    n_iter = len(prices)
    sz = (n_iter,)
    xhat = np.zeros(sz)      # Filtrelenmiş tahmin
    P = np.zeros(sz)         # Hata kovaryansı
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)         # Kalman Kazancı

    # Başlangıç parametreleri
    xhat[0] = prices[0]
    P[0] = 1.0
    
    # Dinamik Gürültü (Adaptive Noise)
    # Son 30 barın varyansına göre R'yi ayarla
    rolling_std = pd.Series(prices).rolling(30).std().fillna(method='bfill').values
    
    for k in range(1, n_iter):
        # Q: Process Noise (Trend değişimi)
        Q = 1e-5 
        # R: Measurement Noise (Piyasa gürültüsü - Adaptif)
        R = (rolling_std[k] * 0.1) ** 2 if rolling_std[k] > 0 else 0.01**2

        # Zaman Güncellemesi
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1] + Q

        # Ölçüm Güncellemesi
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return pd.Series(xhat, index=pd.Series(prices).index)

def calculate_volatility_regime(df):
    """GARCH yerine basit ve etkili ATR tabanlı volatilite rejimi."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    atr = true_range.rolling(14).mean()
    
    # Volatilite Rejimi: Şu anki ATR, son 50 günün ortalamasına göre nerede?
    # > 1: Yüksek Volatilite (Riskli/Trend), < 1: Düşük Volatilite (Yatay)
    vol_regime = atr / atr.rolling(50).mean()
    return vol_regime.fillna(1.0)

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df)<100: return None
    agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    if timeframe=='W': df_res=df.resample('W').agg(agg)
    elif timeframe=='M': df_res=df.resample('ME').agg(agg)
    else: df_res=df.copy()
    
    if len(df_res)<60: return None
    
    # Temel Özellikler
    df_res['kalman'] = adaptive_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['close']/df_res['close'].shift(1))
    
    # Trend Sinyalleri
    df_res['trend_kalman'] = np.where(df_res['close'] > df_res['kalman'], 1, -1)
    df_res['rsi_proxy'] = df_res['close'].pct_change().rolling(14).mean() / df_res['close'].pct_change().rolling(14).std()
    
    # Volatilite Rejimi
    df_res['vol_regime'] = calculate_volatility_regime(df_res)
    
    # Heuristic (Basitleştirilmiş Momentum)
    df_res['momentum'] = (np.sign(df_res['close'].diff(5)) + np.sign(df_res['close'].diff(20)))
    
    # Hedef (Gelecek Getiri Yönü)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    # Temizlik (Sonsuz ve NaN)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    
    return df_res

# --- MODELLEME (XGBoost Meta-Learner) ---

def train_smart_ensemble(df):
    """Gürültüden arındırılmış, XGBoost tabanlı Meta-Learner."""
    test_size = 30 # Son 30 bar gerçek test (Out-of-sample)
    
    # Özellik Seçimi (Noise feature'lar atıldı)
    features = ['log_ret', 'trend_kalman', 'vol_regime', 'momentum', 'rsi_proxy']
    
    # Train / Test Ayrımı
    train_data = df.iloc[:-test_size]
    test_data = df.iloc[-test_size:]
    
    X_train = train_data[features]
    y_train = train_data['target']
    X_test = test_data[features]
    
    # Imputation (Hızlı ve Güvenli: Mean)
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)
    
    # --- LEVEL 1 MODELLER (Base Learners) ---
    # Bu modellerin tahminleri Meta-Model için özellik olacak
    
    # 1. Random Forest (Genel Kalıplar)
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train_imp, y_train)
    
    # 2. Extra Trees (Gürültüye Dayanıklı)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42)
    et.fit(X_train_imp, y_train)
    
    # 3. XGBoost (Base)
    xgb_base = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss')
    xgb_base.fit(X_train_imp, y_train)
    
    # Meta-Data Oluşturma
    meta_X_train = pd.DataFrame({
        'RF': rf.predict_proba(X_train_imp)[:,1],
        'ET': et.predict_proba(X_train_imp)[:,1],
        'XGB': xgb_base.predict_proba(X_train_imp)[:,1],
        'Vol': train_data['vol_regime'].values # Volatilite bilgisini meta modele de veriyoruz
    })
    
    meta_X_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test_imp)[:,1],
        'ET': et.predict_proba(X_test_imp)[:,1],
        'XGB': xgb_base.predict_proba(X_test_imp)[:,1],
        'Vol': test_data['vol_regime'].values
    })
    
    # --- LEVEL 2 META-LEARNER (XGBoost) ---
    # Logistic Regression yerine XGBoost kullanıyoruz (Non-linear meta karar)
    meta_model = xgb.XGBClassifier(
        n_estimators=200, 
        max_depth=3, 
        learning_rate=0.05, 
        subsample=0.8, 
        colsample_bytree=0.8,
        eval_metric='logloss'
    )
    meta_model.fit(meta_X_train, y_train)
    
    # Tahminler
    probs = meta_model.predict_proba(meta_X_test)[:,1]
    
    # --- SİMÜLASYON (Position Sizing ile) ---
    sim_eq = [100]
    signals = []
    
    for i in range(len(test_data)):
        ret = test_data['ret'].iloc[i]
        prob = probs[i]
        
        # Position Sizing (Tanh ile yumuşatılmış)
        # Güven %50'den ne kadar uzaksa pozisyon o kadar büyür. Max 1.0 (Tam Pozisyon)
        raw_signal = (prob - 0.5) * 2 # -1 ile 1 arası
        pos_size = np.tanh(3 * raw_signal) # Sinyali biraz agresifleştir
        
        # Sinyal Yönü ve Büyüklüğü
        signals.append(pos_size)
        
        # Getiri Hesabı (Long/Short/Cash simülasyonu)
        # Sadece Long-Only (Spot piyasa varsayımı)
        if pos_size > 0.2: # Eşik değeri
            sim_eq.append(sim_eq[-1] * (1 + ret * abs(pos_size))) # Pozisyon büyüklüğüne göre getiri
        else:
            sim_eq.append(sim_eq[-1]) # Nakitte bekle
            
    roi = sim_eq[-1] - 100
    last_signal = signals[-1]
    
    info = {
        'roi': roi,
        'prob': probs[-1],
        'weights': dict(zip(meta_X_train.columns, meta_model.feature_importances_))
    }
    
    return last_signal, info

# --- ANA MOTOR ---
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
        
        current_p = float(raw_df['close'].iloc[-1])
        best_roi = -9999; final_sig = 0; winning_tf = "GÜNLÜK"
        
        # Sadece GÜNLÜK ve HAFTALIK bak (Aylık çok yavaş)
        for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
            df_proc = process_data(raw_df, tf_code)
            if df_proc is None: continue
            
            sig, info = train_smart_ensemble(df_proc)
            
            if info['roi'] > best_roi:
                best_roi = info['roi']
                final_sig = sig
                winning_tf = tf_name
        
        signals.append({'idx':idx, 'ticker':ticker, 'price':current_p, 'signal':final_sig, 'roi':best_roi, 'tf':winning_tf, 'status':row['Durum'], 'amount':float(row['Miktar'])})
        print(f"   > Sinyal Gücü: {final_sig:.2f} | ROI: {best_roi:.2f}")

    # Satış (Zayıf Sinyaller)
    for s in signals:
        # Eğer sinyal 0.2'nin altındaysa ve elimizde varsa SAT
        if s['status']=='COIN' and s['signal'] < 0.2:
            rev = s['amount']*s['price']; total_cash+=rev
            updated.at[s['idx'],'Durum']='CASH'; updated.at[s['idx'],'Miktar']=0.0
            updated.at[s['idx'],'Nakit_Bakiye_USD']=0.0
            updated.at[s['idx'],'Son_Islem_Log']=f"SAT ({s['tf']})"
            updated.at[s['idx'],'Son_Islem_Zamani']=time_str

    # Alım (Güçlü Sinyaller - Orantılı Dağıtım)
    buy_cands = [s for s in signals if s['signal'] > 0.2 and s['roi'] > 0]
    total_pos_signal = sum([s['signal'] for s in buy_cands]) # Sinyal gücüne göre ağırlık
    
    if buy_cands and total_cash > 1.0:
        for cand in buy_cands:
            weight = cand['signal'] / total_pos_signal
            allocation = total_cash * weight
            
            if updated.at[cand['idx'],'Durum'] == 'CASH':
                amt = allocation / cand['price']
                updated.at[cand['idx'],'Durum']='COIN'; updated.at[cand['idx'],'Miktar']=amt
                updated.at[cand['idx'],'Nakit_Bakiye_USD']=0.0
                updated.at[cand['idx'],'Son_Islem_Fiyati']=cand['price']
                updated.at[cand['idx'],'Son_Islem_Log']=f"AL (Güç: {cand['signal']:.2f})"
                updated.at[cand['idx'],'Son_Islem_Zamani']=time_str
    elif total_cash > 0:
        f_idx = updated.index[0]
        updated.at[f_idx,'Nakit_Bakiye_USD'] += total_cash
        for ix in updated.index:
            if ix!=f_idx and updated.at[ix,'Durum']=='CASH': updated.at[ix,'Nakit_Bakiye_USD']=0.0

    # Değerleme
    for idx, row in updated.iterrows():
        p = next((s['price'] for s in signals if s['idx']==idx), 0.0)
        if p>0: updated.at[idx,'Kaydedilen_Deger_USD'] = (float(updated.at[idx,'Miktar'])*p) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])

    save_portfolio(updated, sheet)
    print("🏁 Bitti.")

if __name__ == "__main__":
    run_bot_logic()
