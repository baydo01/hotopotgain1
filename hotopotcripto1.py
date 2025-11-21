import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import time
import warnings
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

warnings.filterwarnings("ignore")

# --- AYARLAR ---
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
START_DATE = "2018-01-01" 

st.set_page_config(page_title="Hedge Fund Bot: Tournament Edition", layout="wide")

# --- BAĞLANTI ---
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    try:
        json_key = dict(st.secrets["gcp_service_account"])
        if "private_key" in json_key:
            json_key["private_key"] = json_key["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        st.error(f"Bağlantı Hatası: {e}")
        st.stop()

def save_to_google_sheets(df):
    try:
        sheet = connect_sheet()
        sheet.clear()
        df_export = df.copy().astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except Exception as e: st.error(f"Hata: {e}")

def load_from_google_sheets():
    try:
        sheet = connect_sheet()
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        required_cols = ["Ticker","Durum","Miktar","Son_Islem_Fiyati","Nakit_Bakiye_USD","Baslangic_USD","Kaydedilen_Deger_USD","Son_Islem_Log","Son_Islem_Zamani"]
        if df.empty: return pd.DataFrame(columns=required_cols)
        for col in required_cols:
            if col not in df.columns: df[col] = 0.0 if "USD" in col or "Miktar" in col or "Fiyat" in col else "-"
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df
    except: return pd.DataFrame(columns=["Ticker","Durum","Miktar","Son_Islem_Fiyati","Nakit_Bakiye_USD","Baslangic_USD","Kaydedilen_Deger_USD","Son_Islem_Log","Son_Islem_Zamani"])

def get_current_time_str():
    return datetime.now(pytz.timezone('Europe/Istanbul')).strftime("%d-%m %H:%M")

# --- SİMÜLASYON BAŞLATMA ---
def init_simulation(tickers, amount_per_coin=10):
    data = []
    progress = st.progress(0)
    for i, ticker in enumerate(tickers):
        # HATA DÜZELTMESİ BURADA YAPILDI (MultiIndex Handle)
        try:
            df = yf.download(ticker, period="1d", progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                
                col_name = 'Close' if 'Close' in df.columns else 'close'
                price = float(df[col_name].iloc[-1])
                
                data.append({
                    "Ticker": ticker, "Durum": "COIN", "Miktar": amount_per_coin/price,
                    "Son_Islem_Fiyati": price, "Nakit_Bakiye_USD": 0.0, "Baslangic_USD": float(amount_per_coin),
                    "Kaydedilen_Deger_USD": float(amount_per_coin), "Son_Islem_Log": "Başlangıç", "Son_Islem_Zamani": get_current_time_str()
                })
        except:
            pass
        progress.progress((i+1)/len(tickers))
    
    df = pd.DataFrame(data)
    save_to_google_sheets(df)
    return df

# --- YENİ PUANLAMA MOTORU (BOT İLE AYNI) ---
def calculate_custom_score(df):
    if len(df) < 366: return pd.Series(0, index=df.index)
    daily_steps = np.sign(df['close'].diff()).fillna(0)
    
    # 1. Kısa Vade (5 gün)
    s1 = np.where(daily_steps.rolling(5).sum() > 0, 1, -1)
    # 2. Orta Vade (35 gün)
    s2 = np.where(daily_steps.rolling(35).sum() > 0, 1, -1)
    # 3. Uzun Vade (Tersine / Mean Reversion 150 gün)
    s3 = np.where(daily_steps.rolling(150).sum() < 0, 1, -1)
    # 4. Makro Trend (Eğim)
    ma = df['close'].rolling(365).mean()
    s4 = np.where(ma > ma.shift(1), 1, -1)
    # 5. Volatilite
    vol = df['close'].pct_change().rolling(10).std()
    s5 = np.where(vol < vol.shift(1), 1, -1)
    # 6. Hacim
    s6 = np.where(df['volume'] > df['volume'].rolling(20).mean(), 1, 0) if 'volume' in df.columns else 0
    # 7. Mum
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    
    return s1 + s2 + s3 + s4 + s5 + s6 + s7

# --- TURNUVA MOTORU (BOT İLE AYNI) ---
@st.cache_data(ttl=600)
def run_tournament_logic_cached(ticker):
    try:
        df_raw = yf.download(ticker, start=START_DATE, progress=False)
        if df_raw.empty or len(df_raw) < 730: return "VERI_YOK", 0.0
        
        # MultiIndex Düzeltme
        if isinstance(df_raw.columns, pd.MultiIndex): df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.columns = [c.lower() for c in df_raw.columns]
        if 'close' not in df_raw.columns and 'adj close' in df_raw.columns: df_raw['close'] = df_raw['adj close']
        
        timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
        weight_scenarios = [0.50, 0.70, 0.85, 0.90, 0.95]
        best_roi = -9999
        final_decision = "BEKLE"
        current_price = float(df_raw['close'].iloc[-1])
        winning_info = ""

        for tf_name, tf_code in timeframes.items():
            if tf_code == 'D': df = df_raw.copy()
            else:
                agg = {'close': 'last', 'high': 'max', 'low': 'min'}
                if 'open' in df_raw.columns: agg['open'] = 'first'
                if 'volume' in df_raw.columns: agg['volume'] = 'sum'
                df = df_raw.resample(tf_code).agg(agg).dropna()
            
            if len(df) < 200: continue
            df['log_ret'] = np.log(df['close']/df['close'].shift(1))
            df['range'] = (df['high'] - df['low'])/df['close']
            df['custom_score'] = calculate_custom_score(df)
            df.dropna(inplace=True)
            if len(df) < 50: continue

            X = df[['log_ret', 'range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            try:
                model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
                model.fit(X_s)
                states = model.predict(X_s)
                df['state'] = states
            except: continue

            state_stats = df.groupby('state')['log_ret'].mean()
            bull_state = state_stats.idxmax()
            bear_state = state_stats.idxmin()

            for w_hmm in weight_scenarios:
                w_score = 1.0 - w_hmm
                cash = 10000; coin_amt = 0; history = []
                for _, row in df.iterrows():
                    p = row['close']
                    hm = 1 if row['state'] == bull_state else (-1 if row['state'] == bear_state else 0)
                    sc = 1 if row['custom_score'] >= 3 else (-1 if row['custom_score'] <= -3 else 0)
                    dv = (w_hmm * hm) + (w_score * sc)
                    if dv > 0.25 and cash > 0: coin_amt = cash/p; cash=0
                    elif dv < -0.25 and coin_amt > 0: cash = coin_amt*p; coin_amt=0
                    history.append(cash + coin_amt*p)
                
                roi = (history[-1] - 10000)/10000
                if roi > best_roi:
                    best_roi = roi
                    winning_info = f"{tf_name} (Ağırlık: %{int(w_hmm*100)})"
                    # Son Karar
                    lr = df.iloc[-1]
                    l_hm = 1 if lr['state'] == bull_state else (-1 if lr['state'] == bear_state else 0)
                    l_sc = 1 if lr['custom_score'] >= 3 else (-1 if lr['custom_score'] <= -3 else 0)
                    l_dv = (w_hmm * l_hm) + (w_score * l_sc)
                    if l_dv > 0.25: final_decision = "AL"
                    elif l_dv < -0.25: final_decision = "SAT"
                    else: final_decision = "BEKLE"

        return final_decision, current_price, winning_info
    except: return "HATA", 0.0, "-"

# --- ARAYÜZ ---
st.title("🏆 Hedge Fund Bot: Tournament Edition")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    selected_tickers = st.multiselect("Coinler", tickers, default=tickers)
    if st.button("⚠️ SİMÜLASYONU SIFIRLA"):
        init_simulation(selected_tickers, 10)
        st.success("Sıfırlandı!"); time.sleep(1); st.rerun()
    st.info("Bu ekran GitHub Botu ile %100 aynı 'Turnuva Mantığı'nı kullanır.")

pf_df = load_from_google_sheets()

if pf_df.empty: st.warning("Veri yok."); st.stop()

# Metrikler
cur_val = 0; saved_val = 0; invested = pf_df['Baslangic_USD'].sum()
if 'Kaydedilen_Deger_USD' in pf_df.columns: saved_val = pf_df['Kaydedilen_Deger_USD'].sum()

# Canlı Fiyat Çek (Hızlı Gösterim İçin) - DÜZELTİLEN KISIM
live_prices = {}
for t in pf_df['Ticker']:
    try:
        d = yf.download(t, period="1d", progress=False)
        if not d.empty: 
            # MultiIndex Kontrolü
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            
            # Column name case sensitivity handle
            col_name = 'Close' if 'Close' in d.columns else 'close'
            
            # Sadece değeri al (scalar)
            val = d[col_name].iloc[-1]
            live_prices[t] = float(val)
        else: 
            live_prices[t] = 0.0
    except:
        live_prices[t] = 0.0

display_data = []
for _, row in pf_df.iterrows():
    p = live_prices.get(row['Ticker'], 0)
    val = (row['Miktar']*p) if row['Durum']=='COIN' else row['Nakit_Bakiye_USD']
    cur_val += val
    display_data.append({
        "Coin": row['Ticker'], "Durum": row['Durum'], "Fiyat": p, "Değer": val,
        "Son İşlem": f"{row['Son_Islem_Log']} ({row['Son_Islem_Zamani']})"
    })

m1, m2, m3 = st.columns(3)
m1.metric("Toplam Portföy", f"${cur_val:.2f}")
m2.metric("Son Girişten Beri", f"${cur_val - saved_val:.2f}", delta_color="normal")
m3.metric("Net Kâr", f"${cur_val - invested:.2f}")

# Manuel Analiz Butonu
if st.button("🤖 MANUEL ANALİZ ET (Aynı Motoru Çalıştır)", type="primary"):
    with st.spinner("Turnuva Modeli Çalışıyor... (Bu işlem 30-60sn sürebilir)"):
        updated_pf = pf_df.copy()
        time_str = get_current_time_str()
        for idx, row in updated_pf.iterrows():
            t = row['Ticker']
            dec, price, info = run_tournament_logic_cached(t)
            if price <= 0: continue
            
            # İşlem Mantığı
            if row['Durum']=='COIN' and dec=='SAT':
                cash = row['Miktar'] * price
                updated_pf.at[idx,'Durum']='CASH'; updated_pf.at[idx,'Nakit_Bakiye_USD']=cash; updated_pf.at[idx,'Miktar']=0
                updated_pf.at[idx,'Son_Islem_Log']="SATILDI"; updated_pf.at[idx,'Son_Islem_Zamani']=time_str
                st.toast(f"🔴 {t} SATILDI (Strateji: {info})")
            elif row['Durum']=='CASH' and dec=='AL':
                cash = row['Nakit_Bakiye_USD']
                if cash > 0:
                    updated_pf.at[idx,'Durum']='COIN'; updated_pf.at[idx,'Miktar']=cash/price; updated_pf.at[idx,'Nakit_Bakiye_USD']=0
                    updated_pf.at[idx,'Son_Islem_Log']="ALINDI"; updated_pf.at[idx,'Son_Islem_Zamani']=time_str
                    st.toast(f"🟢 {t} ALINDI (Strateji: {info})")
            
            # Değer Güncelle
            val = (updated_pf.at[idx,'Miktar']*price) if updated_pf.at[idx,'Durum']=='COIN' else updated_pf.at[idx,'Nakit_Bakiye_USD']
            updated_pf.at[idx,'Kaydedilen_Deger_USD'] = val
            
        save_to_google_sheets(updated_pf)
        st.success("Güncellendi!")
        time.sleep(2); st.rerun()

st.dataframe(pd.DataFrame(display_data).style.format({"Fiyat": "${:.2f}", "Değer": "${:.2f}"}))
