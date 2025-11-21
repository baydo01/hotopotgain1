import os
import json
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz
import warnings

# Uyarıları gizle
warnings.filterwarnings("ignore")

# --- AYARLAR ---
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
# Turnuva için en az 2018'den beri veri çekeceğiz
START_DATE = "2018-01-01" 

# --- GOOGLE SHEETS BAĞLANTISI ---
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds_json_str = os.environ.get("GCP_CREDS")
    if not creds_json_str:
        print("HATA: GCP_CREDS bulunamadı.")
        return None
    creds_dict = json.loads(creds_json_str)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    return client.open_by_key(SHEET_ID).sheet1

def load_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    required_cols = ["Ticker","Durum","Miktar","Son_Islem_Fiyati","Nakit_Bakiye_USD","Baslangic_USD","Kaydedilen_Deger_USD","Son_Islem_Log","Son_Islem_Zamani"]
    if df.empty: return pd.DataFrame(columns=required_cols), sheet

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if "USD" in col or "Miktar" in col or "Fiyat" in col else "-"

    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    df_export = df.copy()
    df_export = df_export.astype(str)
    sheet.clear()
    sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())

# --- GELİŞMİŞ PUANLAMA SİSTEMİ (SENİN KODUNUN AYNISI) ---
def calculate_custom_score(df):
    if len(df) < 366: return pd.Series(0, index=df.index)
    # 1. Kısa Vade (5 gün)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    # 2. Orta Vade (35 gün)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    # 3. Uzun Vade (150 gün)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    # 4. Makro Vade (365 gün)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    # 5. Volatilite Yönü
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    # 6. Hacim Trendi
    if 'volume' in df.columns:
        s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1)
    else: s6 = 0
    # 7. Mum Yapısı
    if 'open' in df.columns:
        s7 = np.where(df['close'] > df['open'], 1, -1)
    else: s7 = 0
    return s1 + s2 + s3 + s4 + s5 + s6 + s7

# --- TURNUVA MOTORU ---
def run_tournament_logic(ticker):
    """
    Bu fonksiyon verilen ticker için:
    1. Geçmiş veriyi çeker.
    2. Günlük, Haftalık, Aylık olarak dener.
    3. Farklı ağırlıkları dener.
    4. En iyi sonucu veren stratejinin ŞU ANKİ kararını (AL/SAT) döndürür.
    """
    try:
        # Veri Çekme (Uzun vade)
        df_raw = yf.download(ticker, start=START_DATE, progress=False)
        if df_raw.empty or len(df_raw) < 730: return "VERI_YOK", 0.0

        # MultiIndex düzeltme
        if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.columns = [c.lower() for c in df_raw.columns]
        if 'close' not in df_raw.columns and 'adj close' in df_raw.columns: df_raw['close'] = df_raw['adj close']
        
        # Parametreler
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        
        best_roi = -9999
        final_decision = "BEKLE"
        current_price = float(df_raw['close'].iloc[-1])
        
        # --- TURNUVA DÖNGÜSÜ ---
        for tf_name, tf_code in timeframes.items():
            # Resample (Zaman dilimi değiştirme)
            if tf_code == 'D':
                df = df_raw.copy()
            else:
                agg_dict = {'close': 'last', 'high': 'max', 'low': 'min'}
                if 'open' in df_raw.columns: agg_dict['open'] = 'first'
                if 'volume' in df_raw.columns: agg_dict['volume'] = 'sum'
                df = df_raw.resample(tf_code).agg(agg_dict).dropna()

            if len(df) < 200: continue

            # Feature Engineering
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
            df['range'] = (df['high'] - df['low']) / df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            if len(df) < 50: continue

            # HMM Eğitimi
            X = df[['log_ret', 'range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            
            try:
                model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_s)
                states = model.predict(X_s)
                df['state'] = states
            except: continue

            # Boğa/Ayı Tespiti
            state_stats = df.groupby('state')['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()
            
            # Ağırlık Testleri
            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm
                
                # Backtest Simülasyonu
                cash = 10000 # Sanal Kasa
                coin_amt = 0
                
                # Hızlı Backtest (Vektörize edilmemiş, döngüsel - güvenilir olması için)
                # Burada tüm geçmişi simüle edip ROI hesaplıyoruz
                portfolio_vals = []
                
                for idx, row in df.iterrows():
                    price = row['close']
                    state = row['state']
                    score = row['custom_score']
                    
                    # Sinyaller
                    hmm_sig = 1 if state == bull_state else (-1 if state == bear_state else 0)
                    score_sig = 1 if score >= 3 else (-1 if score <= -3 else 0)
                    
                    decision_val = (w_hmm * hmm_sig) + (w_score * score_sig)
                    
                    # İşlem Mantığı
                    if decision_val > 0.25: # AL
                        if cash > 0:
                            coin_amt = cash / price
                            cash = 0
                    elif decision_val < -0.25: # SAT
                        if coin_amt > 0:
                            cash = coin_amt * price
                            coin_amt = 0
                    
                    portfolio_vals.append(cash + (coin_amt * price))
                
                # Sonuç Değerlendirme
                final_balance = portfolio_vals[-1]
                roi = (final_balance - 10000) / 10000
                
                # Eğer bu strateji, şu ana kadarki en iyisi ise, SON kararını kaydet
                if roi > best_roi:
                    best_roi = roi
                    
                    # En son (güncel) durumun kararını bul
                    last_row = df.iloc[-1]
                    last_hmm = 1 if last_row['state'] == bull_state else (-1 if last_row['state'] == bear_state else 0)
                    last_score = last_row['custom_score']
                    last_score_sig = 1 if last_score >= 3 else (-1 if last_score <= -3 else 0)
                    
                    last_decision_val = (w_hmm * last_hmm) + (w_score * last_score_sig)
                    
                    if last_decision_val > 0.25: final_decision = "AL"
                    elif last_decision_val < -0.25: final_decision = "SAT"
                    else: final_decision = "BEKLE"
                    
                    print(f"  > Yeni Lider: {tf_name} | Ağırlık: %{int(w_hmm*100)} | ROI: %{int(roi*100)} | Karar: {final_decision}")

        return final_decision, current_price

    except Exception as e:
        print(f"Turnuva Hatası ({ticker}): {e}")
        return "HATA", 0.0

# --- ANA ÇALIŞMA MANTIĞI ---
def main():
    print("--- Turnuva Botu Başlatılıyor ---")
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    pf_df, sheet = load_portfolio()
    if pf_df.empty:
        print("Portföy boş.")
        return

    updated_portfolio = pf_df.copy()
    
    for idx, row in updated_portfolio.iterrows():
        ticker = row['Ticker']
        print(f"\nAnaliz ediliyor: {ticker}...")
        
        # Turnuvayı Çalıştır ve Kazananın Kararını Al
        decision, current_price = run_tournament_logic(ticker)
        print(f"🏆 {ticker} Kazanan Strateji Kararı: {decision} (Fiyat: {current_price})")
        
        if current_price <= 0 or decision == "HATA": continue
        
        # İŞLEM UYGULAMA (All-in / All-out)
        
        # SATIŞ
        if row['Durum']=='COIN' and decision=='SAT':
            cash = float(row['Miktar']) * current_price
            updated_portfolio.at[idx,'Durum'] = 'CASH'
            updated_portfolio.at[idx,'Nakit_Bakiye_USD'] = cash
            updated_portfolio.at[idx,'Miktar'] = 0.0
            updated_portfolio.at[idx,'Son_Islem_Fiyati'] = current_price
            updated_portfolio.at[idx,'Son_Islem_Log'] = "SATILDI"
            updated_portfolio.at[idx,'Son_Islem_Zamani'] = time_str
            print(f"🔴 İŞLEM: {ticker} SATILDI.")
            
        # ALIŞ
        elif row['Durum']=='CASH' and decision=='AL':
            cash = float(row['Nakit_Bakiye_USD'])
            if cash > 0:
                amount = cash / current_price
                updated_portfolio.at[idx,'Durum'] = 'COIN'
                updated_portfolio.at[idx,'Miktar'] = amount
                updated_portfolio.at[idx,'Nakit_Bakiye_USD'] = 0.0
                updated_portfolio.at[idx,'Son_Islem_Fiyati'] = current_price
                updated_portfolio.at[idx,'Son_Islem_Log'] = "ALINDI"
                updated_portfolio.at[idx,'Son_Islem_Zamani'] = time_str
                print(f"🟢 İŞLEM: {ticker} ALINDI.")
        
        # Değer Güncelleme
        if updated_portfolio.at[idx, 'Durum'] == 'COIN':
            val = float(updated_portfolio.at[idx, 'Miktar']) * current_price
        else:
            val = float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
        updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val

    save_portfolio(updated_portfolio, sheet)
    print("\n--- Tüm işlemler tamamlandı ve kaydedildi. ---")

if __name__ == "__main__":
    main()
