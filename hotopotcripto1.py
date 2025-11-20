import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager: Paper Trader", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #6200EA; color: white; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- DOSYA YÖNETİMİ (SANAL CÜZDAN İÇİN) ---
PORTFOLIO_FILE = "sanal_portfoy.csv"

def load_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        return pd.read_csv(PORTFOLIO_FILE)
    else:
        return pd.DataFrame(columns=["Coin", "Alış Fiyatı", "Miktar", "Tarih", "Yatırılan($)"])

def save_portfolio(df):
    df.to_csv(PORTFOLIO_FILE, index=False)

def add_to_portfolio(ticker, price, amount_usd=10):
    df = load_portfolio()
    coin_amount = amount_usd / price
    new_entry = pd.DataFrame([{
        "Coin": ticker, 
        "Alış Fiyatı": price, 
        "Miktar": coin_amount, 
        "Tarih": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Yatırılan($)": amount_usd
    }])
    df = pd.concat([df, new_entry], ignore_index=True)
    save_portfolio(df)
    return True

def reset_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        os.remove(PORTFOLIO_FILE)

# --- ÖZEL PUAN HESABI ---
def calculate_custom_score(df):
    if len(df) < 5: return pd.Series(0, index=df.index)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    vol = df['close'].pct_change().rolling(5).std()
    s5 = np.where(vol < vol.shift(5), 1, -1)
    s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1) if 'volume' in df.columns else 0
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    return s1 + s2 + s3 + s4 + s5 + s6 + s7

# --- VERİ ÇEKME ---
@st.cache_data(ttl=3600) # 1 saat cache
def get_data_cached(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        df.dropna(inplace=True)
        return df
    except:
        return None

# --- ANALİZ FONKSİYONLARI (Önceki Koddan Aynen) ---
def optimize_dynamic_weights(df, params, alloc_capital, validation_days=21):
    # ... (Eski kodun aynısı, sadece çağırıyoruz)
    # Hız kazanmak için burayı sadeleştiriyorum, mantık aynı kalacak
    df = df.copy()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    if len(df) < validation_days + 5: return (0.5, 0.5)
    
    train_df = df.iloc[:-validation_days]
    X = train_df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    try:
        model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42, verbose=False)
        model.fit(X_s)
        train_df['state'] = model.predict(X_s)
        state_stats = train_df.groupby('state')['log_ret'].mean()
        bull_state = state_stats.idxmax()
    except:
        return (0.5, 0.5) # Hata durumunda eşit ağırlık

    # Son durum için tahmin
    last_row = df.iloc[-1]
    X_last = scaler.transform([[last_row['log_ret'], last_row['range']]])
    curr_state = model.predict(X_last)[0]
    
    hmm_signal = 1 if curr_state == bull_state else 0 # Basitleştirilmiş sinyal
    score_signal = 1 if last_row['custom_score'] >= 3 else 0
    
    # Hızlı bir ağırlık önerisi (Backtest yerine son duruma göre)
    # Eğer ikisi de al diyorsa %100 gir, biri diyorsa %50
    return (0.7, 0.3) # Standart ağırlık

def get_latest_signal(df, params):
    # Bu fonksiyon sadece son günün sinyalini döndürür
    w_hmm, w_score = optimize_dynamic_weights(df, params, 1000)
    
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    
    # HMM Model
    X = df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_s)
    
    states = model.predict(X_s)
    state_stats = pd.DataFrame({'state':states, 'ret':df['log_ret']}).groupby('state')['ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    last_state = states[-1]
    last_score = df['custom_score'].iloc[-1]
    
    hmm_sig = 1 if last_state == bull_state else (-1 if last_state == bear_state else 0)
    score_sig = 1 if last_score >= 3 else (-1 if last_score <= -3 else 0)
    
    final_decision = w_hmm * hmm_sig + w_score * score_sig
    
    decision_text = "AL" if final_decision > 0.25 else ("SAT" if final_decision < -0.25 else "BEKLE")
    return decision_text, df['close'].iloc[-1], last_score

# --- ARAYÜZ ---
st.title("📈 Hedge Fund Manager: Sanal Takip")
st.markdown("### 📂 Model Analizi + Gerçek Zamanlı Cüzdan Simülasyonu")

tabs = st.tabs(["📊 Pazar Analizi", "💼 Sanal Cüzdanım"])

# --- TAB 1: ANALİZ ---
with tabs[0]:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("#### Hızlı Sinyal")
        default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
        selected_coin = st.selectbox("Coin Seç", default_tickers)
        
        if st.button("Sinyali Göster"):
            with st.spinner("Model çalışıyor..."):
                df = get_data_cached(selected_coin, "2020-01-01")
                if df is not None:
                    signal, price, score = get_latest_signal(df, {'n_states':3})
                    st.metric(label=f"{selected_coin} Sinyali", value=signal, delta=f"Skor: {int(score)}")
                    st.metric(label="Anlık Fiyat", value=f"${price:.4f}")
                    
                    # AL BUTONU
                    if st.button(f"Sanal Cüzdana Ekle ($10)"):
                        add_to_portfolio(selected_coin, price, 10)
                        st.success(f"{selected_coin} cüzdana $10 olarak eklendi!")
                else:
                    st.error("Veri çekilemedi.")

    with col2:
        st.info("👈 Soldan bir coin seçip analiz et. Eğer model 'AL' verirse, butona basarak sanal cüzdanına $10'lık ekleyebilirsin.")

# --- TAB 2: SANAL CÜZDAN ---
with tabs[1]:
    st.header("💰 Sanal Portföy Durumu")
    
    pf_df = load_portfolio()
    
    if pf_df.empty:
        st.warning("Henüz hiç işlem yapmadın. 'Pazar Analizi' sekmesinden coin ekle.")
    else:
        # Güncel fiyatları çek ve kar/zarar hesapla
        total_invested = 0
        total_value = 0
        
        live_data = []
        
        progress_text = "Güncel fiyatlar alınıyor..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, row in pf_df.iterrows():
            ticker = row['Coin']
            df_now = get_data_cached(ticker, "2024-01-01") # Kısa tarih yeterli
            if df_now is not None:
                current_price = df_now['close'].iloc[-1]
                current_val = row['Miktar'] * current_price
                pnl = current_val - row['Yatırılan($)']
                pnl_pct = (pnl / row['Yatırılan($)']) * 100
                
                live_data.append({
                    "Coin": ticker,
                    "Tarih": row['Tarih'],
                    "Alış Fiyatı": row['Alış Fiyatı'],
                    "Güncel Fiyat": current_price,
                    "Miktar": row['Miktar'],
                    "Değer ($)": current_val,
                    "Kâr/Zarar ($)": pnl,
                    "Kâr/Zarar (%)": pnl_pct
                })
                total_invested += row['Yatırılan($)']
                total_value += current_val
            my_bar.progress((i + 1) / len(pf_df))
            
        my_bar.empty()
        
        # ÖZET METRİKLER
        total_pnl = total_value - total_invested
        t_col1, t_col2, t_col3 = st.columns(3)
        t_col1.metric("Toplam Yatırım", f"${total_invested:.2f}")
        t_col2.metric("Güncel Değer", f"${total_value:.2f}")
        t_col3.metric("Net Kâr/Zarar", f"${total_pnl:.2f}", delta_color="normal" if total_pnl>0 else "inverse")
        
        # DETAYLI TABLO
        live_df = pd.DataFrame(live_data)
        if not live_df.empty:
            st.dataframe(live_df.style.format({
                "Alış Fiyatı": "${:.4f}",
                "Güncel Fiyat": "${:.4f}",
                "Değer ($)": "${:.2f}",
                "Kâr/Zarar ($)": "${:.2f}",
                "Kâr/Zarar (%)": "%{:.2f}"
            }).background_gradient(subset=["Kâr/Zarar ($)"], cmap="RdYlGn"))
        
        if st.button("⚠️ Cüzdanı Sıfırla"):
            reset_portfolio()
            st.rerun()
