import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings

# Gereksiz uyarıları sustur
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

# --- HESAPLAMA VE VERİ TEMİZLİĞİ ---
def calculate_custom_score(df):
    if len(df) < 5: return pd.Series(0, index=df.index)
    s1 = np.where(df['close'] > df['close'].shift(5), 1, -1)
    s2 = np.where(df['close'] > df['close'].shift(35), 1, -1)
    s3 = np.where(df['close'] > df['close'].shift(150), 1, -1)
    s4 = np.where(df['close'] > df['close'].shift(365), 1, -1)
    
    # Volatilite kontrolü (Sıfıra bölme hatasını engellemek için fillna)
    pct_change = df['close'].pct_change().fillna(0)
    vol = pct_change.rolling(5).std().fillna(0)
    
    s5 = np.where(vol < vol.shift(5), 1, -1)
    s6 = np.where(df['volume'] > df['volume'].rolling(5).mean(), 1, -1) if 'volume' in df.columns else 0
    s7 = np.where(df['close'] > df['open'], 1, -1) if 'open' in df.columns else 0
    return s1 + s2 + s3 + s4 + s5 + s6 + s7

def prepare_data(df):
    """
    Veriyi hesaplamalar için hazırlar ve HMM'i çökertecek
    NaN veya Sonsuz değerleri temizler.
    """
    df = df.copy()
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['range'] = (df['high'] - df['low']) / df['close']
    df['custom_score'] = calculate_custom_score(df)
    
    # --- KRİTİK TEMİZLİK ---
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    # -----------------------
    return df

# --- VERİ ÇEKME ---
@st.cache_data(ttl=3600)
def get_data_cached(ticker, start_date):
    try:
        df = yf.download(ticker, start=start_date, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        
        # İlk ham verideki boşlukları atalım
        df.dropna(inplace=True)
        return df
    except:
        return None

# --- ANALİZ FONKSİYONLARI ---
def optimize_dynamic_weights(df, params, alloc_capital, validation_days=21):
    # Veriyi temizle ve hazırla
    df = prepare_data(df)
    
    if len(df) < validation_days + 10: 
        return (0.7, 0.3) # Yetersiz veri varsa varsayılan ağırlık
    
    train_df = df.iloc[:-validation_days]
    test_df = df.iloc[-validation_days:]
    
    X = train_df[['log_ret','range']].values
    scaler = StandardScaler()
    
    try:
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42, verbose=False)
        model.fit(X_s)
        
        train_df['state'] = model.predict(X_s)
        state_stats = train_df.groupby('state')['log_ret'].mean()
        bull_state = state_stats.idxmax()
        bear_state = state_stats.idxmin()
        
        # Basit bir optimizasyon döngüsü
        best_roi = -np.inf
        best_w = (0.5, 0.5)
        weight_candidates = [0.3, 0.5, 0.7, 0.9]

        for w_hmm in weight_candidates:
            w_score = 1 - w_hmm
            cash = alloc_capital
            coin_amt = 0
            
            # Test seti üzerinde simülasyon
            for idx, row in test_df.iterrows():
                # Tekil veri tahmini için array şekillendirme
                X_test = scaler.transform([[row['log_ret'], row['range']]])
                state_pred = model.predict(X_test)[0]
                
                hmm_signal = 1 if state_pred == bull_state else (-1 if state_pred == bear_state else 0)
                score_signal = 1 if row['custom_score'] >= 3 else (-1 if row['custom_score'] <= -3 else 0)
                
                decision = w_hmm * hmm_signal + w_score * score_signal
                
                price = row['close']
                # Basit al-sat mantığı
                if decision > 0.25 and cash > 0: 
                    coin_amt = cash / price
                    cash = 0
                elif decision < -0.25 and coin_amt > 0: 
                    cash = coin_amt * price
                    coin_amt = 0
            
            final_val = cash + coin_amt * test_df['close'].iloc[-1]
            roi = (final_val - alloc_capital) / alloc_capital
            
            if roi > best_roi:
                best_roi = roi
                best_w = (w_hmm, w_score)
                
        return best_w

    except Exception as e:
        # HMM çökse bile programı durdurma, güvenli moda geç
        return (0.5, 0.5)

def get_latest_signal(df, params):
    # 1. Veriyi temizle
    df = prepare_data(df)
    
    if len(df) < 10:
        return "YETERSİZ VERİ", 0, 0

    # 2. Ağırlıkları hesapla
    w_hmm, w_score = optimize_dynamic_weights(df, params, 1000)
    
    # 3. HMM Model Kurulumu (Son durum için)
    X = df[['log_ret','range']].values
    scaler = StandardScaler()
    
    try:
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
        price = df['close'].iloc[-1]
        
        return decision_text, price, last_score
        
    except Exception as e:
        return f"HATA: {str(e)}", 0, 0

# --- ARAYÜZ ---
st.title("📈 Hedge Fund Manager: Sanal Takip V8")
st.markdown("### ⚔️ HMM + Puan | Sanal İşlem Defteri | Hata Korumalı Mod")

tabs = st.tabs(["📊 Pazar Analizi", "💼 Sanal Cüzdanım"])

# --- TAB 1: ANALİZ ---
with tabs[0]:
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("#### Hızlı Sinyal")
        default_tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
        selected_coin = st.selectbox("Coin Seç", default_tickers)
        
        if st.button("Sinyali Göster"):
            with st.spinner("Model veriyi işliyor ve analiz ediyor..."):
                df = get_data_cached(selected_coin, "2020-01-01")
                
                if df is not None and not df.empty:
                    signal, price, score = get_latest_signal(df, {'n_states':3})
                    
                    if "HATA" in signal:
                        st.error(signal)
                    else:
                        # Renkli gösterim
                        color = "normal"
                        if signal == "AL": color = "normal" # Yeşilimsi (Streamlit default)
                        elif signal == "SAT": color = "inverse" # Kırmızımsı
                        
                        st.metric(label=f"{selected_coin} Kararı", value=signal, delta=f"Puan: {int(score)}", delta_color=color)
                        st.metric(label="Anlık Fiyat", value=f"${price:.4f}")
                        
                        # AL BUTONU (Sadece AL sinyali veya kullanıcı kararı için)
                        st.markdown("---")
                        if st.button(f"➕ Sanal Cüzdana Ekle ($10)"):
                            add_to_portfolio(selected_coin, price, 10)
                            st.success(f"✅ {selected_coin} cüzdana $10 değerinde eklendi!")
                else:
                    st.error("Veri çekilemedi veya çok eksik.")

    with col2:
        st.info("""
        **Nasıl Çalışır?**
        1. Soldan bir coin seç ve 'Sinyali Göster' butonuna bas.
        2. Model (HMM + Algoritma) sana **AL**, **SAT** veya **BEKLE** diyecek.
        3. Eğer 'AL' kararı çıkarsa (veya sen eklemek istersen), **Sanal Cüzdana Ekle** butonuna bas.
        4. 'Sanal Cüzdanım' sekmesinden kâr/zarar durumunu canlı izle.
        """)

# --- TAB 2: SANAL CÜZDAN ---
with tabs[1]:
    st.header("💰 Sanal Portföy Durumu")
    
    pf_df = load_portfolio()
    
    if pf_df.empty:
        st.warning("Henüz işlem kaydı yok. 'Pazar Analizi' sekmesinden işlem ekleyebilirsin.")
    else:
        # Canlı Fiyatları Çek ve Tabloyu Güncelle
        total_invested = 0
        total_value = 0
        live_data = []
        
        progress_text = "Portföy güncelleniyor..."
        my_bar = st.progress(0, text=progress_text)
        
        for i, row in pf_df.iterrows():
            ticker = row['Coin']
            # Sadece son 5 günü çeksek yeter, hız kazanalım
            df_now = get_data_cached(ticker, datetime.now().strftime("%Y-%m-01")) 
            
            current_price = row['Alış Fiyatı'] # Varsayılan olarak alış fiyatı kalsın (hata olursa)
            if df_now is not None and not df_now.empty:
                current_price = df_now['close'].iloc[-1]
            
            current_val = row['Miktar'] * current_price
            pnl = current_val - row['Yatırılan($)']
            pnl_pct = (pnl / row['Yatırılan($)']) * 100 if row['Yatırılan($)'] != 0 else 0
            
            live_data.append({
                "Coin": ticker,
                "Tarih": row['Tarih'],
                "Alış Fiyatı": row['Alış Fiyatı'],
                "Güncel Fiyat": current_price,
                "Miktar": row['Miktar'],
                "Yatırılan ($)": row['Yatırılan($)'], # Tabloda göstermek için
                "Değer ($)": current_val,
                "Kâr/Zarar ($)": pnl,
                "Kâr/Zarar (%)": pnl_pct
            })
            total_invested += row['Yatırılan($)']
            total_value += current_val
            
            my_bar.progress((i + 1) / len(pf_df))
            
        my_bar.empty()
        
        # ÖZET METRİKLERİ
        total_pnl = total_value - total_invested
        total_pnl_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Toplam Yatırım", f"${total_invested:.2f}")
        c2.metric("Güncel Değer", f"${total_value:.2f}")
        c3.metric("Genel Kâr/Zarar", f"${total_pnl:.2f}", f"%{total_pnl_pct:.2f}")
        
        st.markdown("---")
        
        # DETAYLI TABLO
        live_df = pd.DataFrame(live_data)
        if not live_df.empty:
            st.dataframe(live_df.style.format({
                "Alış Fiyatı": "${:.4f}",
                "Güncel Fiyat": "${:.4f}",
                "Yatırılan ($)": "${:.2f}",
                "Değer ($)": "${:.2f}",
                "Kâr/Zarar ($)": "${:.2f}",
                "Kâr/Zarar (%)": "%{:.2f}"
            }).background_gradient(subset=["Kâr/Zarar ($)"], cmap="RdYlGn", vmin=-5, vmax=5))
        
        # SIFIRLAMA BUTONU
        if st.button("⚠️ Tüm Cüzdanı Sıfırla (Dikkat!)"):
            reset_portfolio()
            st.rerun()
