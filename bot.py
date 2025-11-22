# Dosya adı: bot_engine.py
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
SHEET_ID = os.environ.get("SHEET_ID") # GitHub Secrets'tan gelecek
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

# --- GOOGLE SHEETS BAĞLANTISI ---
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    # GitHub Secrets üzerinden gelen JSON verisini kullanacağız
    json_creds = os.environ.get("GCP_CREDENTIALS")
    
    if json_creds:
        creds_dict = json.loads(json_creds)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    elif os.path.exists("service_account.json"):
        creds = ServiceAccountCredentials.from_json_keyfile_name("service_account.json", scope)
    else:
        print("HATA: Kimlik bilgileri bulunamadı!")
        return None

    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        print(f"Sheets Bağlantı Hatası: {e}")
        return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        
        # Sayısal dönüşümler
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        
        return df, sheet
    except Exception as e:
        print(f"Veri Okuma Hatası: {e}")
        return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        # DataFrame'i listeye çevirip güncelleme (Başlıklar dahil)
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        print("✅ Google Sheets Güncellendi.")
    except Exception as e:
        print(f"Kayıt Hatası: {e}")

# --- FONKSİYONLAR (Senin kodundaki matematiksel fonksiyonların aynısı) ---
# Not: Buraya senin kodundaki şu fonksiyonları birebir kopyala yapıştır:
# apply_kalman_filter, calculate_heuristic_score, get_raw_data, process_data
# select_best_garch_model, estimate_arch_garch_models, estimate_arima_models
# estimate_nnar_models, ga_optimize, train_meta_learner
# (Kod uzamasın diye burayı kısalttım, senin son kodundaki fonksiyonları buraya almalısın)

# ... [BURAYA SENİN FONKSİYONLARIN GELECEK] ...
# ... apply_kalman_filter ...
# ... train_meta_learner ... vb ...

# --- ANA ÇALIŞTIRMA MOTORU ---
def run_bot_logic():
    print(f"🚀 Bot Başlatılıyor... {datetime.now()}")
    
    pf_df, sheet = load_and_fix_portfolio()
    if pf_df.empty:
        print("Portföy okunamadı, işlem iptal.")
        return

    updated = pf_df.copy()
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")

    for i, (idx, row) in enumerate(updated.iterrows()):
        ticker = row['Ticker']
        if len(str(ticker)) < 3: continue
        
        print(f"🧠 Analiz ediliyor: {ticker}")
        
        # Turnuva Fonksiyonunun İçeriğini Buraya Entegre Ediyoruz (Basitleştirilmiş)
        raw_df = get_raw_data(ticker)
        if raw_df is None: 
            print(f"Veri yok: {ticker}")
            continue
            
        current_price = float(raw_df['close'].iloc[-1])
        timeframes = {'GÜNLÜK':'D', 'HAFTALIK':'W', 'AYLIK':'M'}
        best_roi = -9999
        final_decision = "BEKLE"
        winning_tf = "YOK"
        best_info = None
        
        for tf_name, tf_code in timeframes.items():
            df = process_data(raw_df, tf_code)
            if df is None: continue
            
            # Optimizasyon ve Eğitim
            params = ga_optimize(df)
            sig, info = train_meta_learner(df, params)
            
            if info is None: continue
            
            if info['bot_roi'] > best_roi:
                best_roi = info['bot_roi']
                winning_tf = tf_name
                best_info = info
                if sig > 0.10: final_decision = "AL"
                elif sig < -0.10: final_decision = "SAT"
                else: final_decision = "BEKLE"
        
        print(f"   > Karar: {final_decision} ({winning_tf}) | ROI: {best_roi:.2f}")

        # Karar Uygulama (Google Sheets Güncelleme Mantığı)
        if final_decision != "HATA" and best_info:
            stt = row['Durum']
            # AL Sinyali ve Nakitteysek
            if stt == 'CASH' and final_decision == 'AL':
                cash = float(row['Nakit_Bakiye_USD'])
                if cash > 1:
                    updated.at[idx, 'Durum'] = 'COIN'
                    updated.at[idx, 'Miktar'] = cash / current_price
                    updated.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                    updated.at[idx, 'Son_Islem_Fiyati'] = current_price
                    updated.at[idx, 'Son_Islem_Log'] = f"AL ({winning_tf}) R:{best_info['bot_roi']:.1f}"
                    updated.at[idx, 'Son_Islem_Zamani'] = time_str
            
            # SAT Sinyali ve Koindeysek
            elif stt == 'COIN' and final_decision == 'SAT':
                amt = float(row['Miktar'])
                if amt > 0:
                    updated.at[idx, 'Durum'] = 'CASH'
                    updated.at[idx, 'Nakit_Bakiye_USD'] = amt * current_price
                    updated.at[idx, 'Miktar'] = 0.0
                    updated.at[idx, 'Son_Islem_Fiyati'] = current_price
                    updated.at[idx, 'Son_Islem_Log'] = f"SAT ({winning_tf}) R:{best_info['bot_roi']:.1f}"
                    updated.at[idx, 'Son_Islem_Zamani'] = time_str
            
            # Değer Güncelleme
            if updated.at[idx, 'Durum'] == 'COIN':
                val = float(updated.at[idx, 'Miktar']) * current_price
            else:
                val = float(updated.at[idx, 'Nakit_Bakiye_USD'])
            updated.at[idx, 'Kaydedilen_Deger_USD'] = val

    # Tüm döngü bitince kaydet
    save_portfolio(updated, sheet)
    print("🏁 Tur Tamamlandı.")

if __name__ == "__main__":
    # Burada fonksiyonların tanımlandığından emin olmalısın.
    # Yukarıdaki boş bıraktığım fonksiyon alanlarını doldurduktan sonra çalışır.
    run_bot_logic()
