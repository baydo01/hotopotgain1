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
import json

warnings.filterwarnings("ignore")

# -----------------------------
# Google Sheets bağlantısı
# -----------------------------
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"

def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    # Secrets'dan JSON key al
    json_key = json.loads(st.secrets["GOOGLE_SERVICE_ACCOUNT_KEY"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json_key, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key(SHEET_ID).sheet1
    return sheet

def save_to_google_sheets(df):
    sheet = connect_sheet()
    sheet.clear()
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

def load_from_google_sheets():
    sheet = connect_sheet()
    data = sheet.get_all_records()
    return pd.DataFrame(data)

# -----------------------------
# Streamlit hafıza/log
# -----------------------------
if 'logs' not in st.session_state:
    st.session_state.logs = []

def add_log(message):
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.logs.insert(0, f"[{timestamp}] {message}")

def load_portfolio():
    try:
        return load_from_google_sheets()
    except:
        return pd.DataFrame(columns=["Ticker","Durum","Miktar","Son_Islem_Fiyati","Nakit_Bakiye_USD","Baslangic_USD"])

def save_portfolio(df):
    save_to_google_sheets(df)

# -----------------------------
# Simülasyon başlat
# -----------------------------
def init_simulation(tickers, amount_per_coin=10):
    data = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        status_text.text(f"{ticker} için piyasa verisi alınıyor...")
        df_price = get_data_cached(ticker)
        if df_price is not None:
            current_price = df_price['close'].iloc[-1]
            coin_amount = amount_per_coin / current_price
            data.append({
                "Ticker": ticker,
                "Durum": "COIN", 
                "Miktar": coin_amount,
                "Son_Islem_Fiyati": current_price,
                "Nakit_Bakiye_USD": 0.0,
                "Baslangic_USD": amount_per_coin
            })
        progress_bar.progress((i + 1) / len(tickers))
        
    df = pd.DataFrame(data)
    save_portfolio(df)
    st.session_state.logs = []
    add_log("Simülasyon başlatıldı. Portföy Google Sheets’e kaydedildi.")
    
    status_text.empty()
    progress_bar.empty()
    return df

# -----------------------------
# Veri ve HMM analizi
# -----------------------------
def calculate_custom_score(df):
    if len(df) < 5: return pd.Series(0, index=df.index)
    s1 = np.where(df['close'] > df['close'].shift(10), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(30), 1, -1)
    vol = df['close'].pct_change().rolling(5).std().fillna(0)
    s3 = np.where(vol < vol.shift(5), 1, -1)
    return s1 + s2 + s3

def prepare_data(df):
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

@st.cache_data(ttl=1800)
def get_data_cached(ticker, start_date="2022-01-01"):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        df.dropna(inplace=True)
        return df
    except: return None

def get_bulk_signals(tickers):
    results = []
    progress = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        try:
            df = get_data_cached(ticker)
            if df is None or len(df) < 20:
                results.append({"Ticker": ticker, "Sinyal": "VERI_YOK", "Fiyat": 0, "Skor": 0})
                continue
                
            df = prepare_data(df)
            X = df[['log_ret','range']].values
            scaler = StandardScaler()
            X_s = scaler.fit_transform(X)
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=50, random_state=42)
            model.fit(X_s)
            states = model.predict(X_s)
            
            state_means = pd.DataFrame({'state': states, 'ret': df['log_ret']}).groupby('state')['ret'].mean()
            bull_state = state_means.idxmax()
            bear_state = state_means.idxmin()
            last_state = states[-1]
            
            hmm_signal = 1 if last_state == bull_state else (-1 if last_state == bear_state else 0)
            score = df['custom_score'].iloc[-1]
            score_signal = 1 if score > 0 else (-1 if score < 0 else 0)
            final_val = 0.6*hmm_signal + 0.4*score_signal
            decision = "AL" if final_val > 0.2 else ("SAT" if final_val < -0.2 else "BEKLE")
            
            results.append({"Ticker":ticker,"Sinyal":decision,"Fiyat":df['close'].iloc[-1],"Skor":int(score)})
        except:
            results.append({"Ticker":ticker,"Sinyal":"HATA","Fiyat":0,"Skor":0})
        progress.progress((i+1)/len(tickers))
    progress.empty()
    return pd.DataFrame(results)

# -----------------------------
# Portföy güncelleme mantığı
# -----------------------------
def run_bot_logic(portfolio_df, signals_df):
    updated_portfolio = portfolio_df.copy()
    for idx, row in updated_portfolio.iterrows():
        ticker = row['Ticker']
        signal_row = signals_df[signals_df['Ticker']==ticker]
        if signal_row.empty: continue
        current_price = signal_row.iloc[0]['Fiyat']
        signal = signal_row.iloc[0]['Sinyal']
        
        if row['Durum']=='COIN' and signal=='SAT':
            cash_obtained = row['Miktar']*current_price
            updated_portfolio.at[idx,'Durum']='CASH'
            updated_portfolio.at[idx,'Nakit_Bakiye_USD']=cash_obtained
            updated_portfolio.at[idx,'Miktar']=0
            updated_portfolio.at[idx,'Son_Islem_Fiyati']=current_price
            add_log(f"🔴 {ticker}: SATIŞ yapıldı. (${cash_obtained:.2f})")
        elif row['Durum']=='CASH' and signal=='AL':
            cash_available = row['Nakit_Bakiye_USD']
            if cash_available>0:
                new_amount = cash_available/current_price
                updated_portfolio.at[idx,'Durum']='COIN'
                updated_portfolio.at[idx,'Miktar']=new_amount
                updated_portfolio.at[idx,'Nakit_Bakiye_USD']=0
                updated_portfolio.at[idx,'Son_Islem_Fiyati']=current_price
                add_log(f"🟢 {ticker}: ALIŞ yapıldı. (${cash_available:.2f})")
    save_portfolio(updated_portfolio)
    return updated_portfolio

# -----------------------------
# Streamlit arayüz
# -----------------------------
st.set_page_config(page_title="Hedge Fund Bot: Sheets Edition", layout="wide")
st.title("🧠 Hedge Fund Bot: Google Sheets Edition")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    selected_tickers = st.multiselect("Coinler", default_tickers, default=default_tickers)
    
    if st.button("⚡ SİMÜLASYONU BAŞLAT/SIFIRLA"):
        init_simulation(selected_tickers, 10)
        st.success("Portföy Google Sheets’e yüklendi.")
        time.sleep(0.5)
        st.rerun()
        
    st.markdown("---")
    st.write("📜 **İşlem Logları**")
    for log in st.session_state.logs:
        st.text(log)

# Ana Ekran
pf_df = load_portfolio()

if pf_df.empty:
    st.info("👈 Soldaki butona basarak simülasyonu başlat.")
else:
    col_btn, col_info = st.columns([1,3])
    signals_df = None
    
    with col_btn:
        if st.button("🔄 BOTU ÇALIŞTIR (ANALİZ ET)"):
            with st.spinner("Portföy taranıyor..."):
                signals_df = get_bulk_signals(pf_df['Ticker'].tolist())
                pf_df = run_bot_logic(pf_df, signals_df)
                st.success("Analiz tamamlandı, portföy güncellendi.")

    # Tablo
    display_data = []
    total_balance = 0
    total_invested = pf_df['Baslangic_USD'].sum()
    
    current_prices = {}
    if signals_df is None:
        for t in pf_df['Ticker']:
            d = get_data_cached(t)
            current_prices[t] = d['close'].iloc[-1] if d is not None else 0
    else:
        for _, r in signals_df.iterrows():
            current_prices[r['Ticker']] = r['Fiyat']

    for idx,row in pf_df.iterrows():
        curr_price = current_prices.get(row['Ticker'],0)
        asset_val = row['Miktar']*curr_price if row['Durum']=='COIN' else row['Nakit_Bakiye_USD']
        pnl = asset_val-row['Baslangic_USD']
        pnl_pct = (pnl/row['Baslangic_USD'])*100
        current_sig = "-"
        if signals_df is not None:
            sig_row = signals_df[signals_df['Ticker']==row['Ticker']]
            if not sig_row.empty:
                current_sig = sig_row.iloc[0]['Sinyal']
        display_data.append({
            "Coin":row['Ticker'],
            "Durum":row['Durum'],
            "Sinyal":current_sig,
            "Fiyat":curr_price,
            "Değer ($)":asset_val,
            "Kâr/Zarar ($)":pnl,
            "Kâr/Zarar (%)":pnl_pct
        })
        total_balance += asset_val

    # Metrikler
    m1,m2,m3 = st.columns(3)
    net_pnl = total_balance - total_invested
    m1.metric("Toplam Portföy", f"${total_balance:.2f}")
    m2.metric("Yatırılan", f"${total_invested:.2f}")
    m3.metric("Net Kâr", f"${net_pnl:.2f}", f"%{(net_pnl/total_invested)*100:.2f}" if total_invested>0 else "0%")

    # Tablo
    st.dataframe(pd.DataFrame(display_data).style.format({
        "Fiyat":"${:.2f}", "Değer ($)":"${:.2f}",
        "Kâr/Zarar ($)":"{:+.2f}", "Kâr/Zarar (%)":"{:+.2f}%"
    }))
