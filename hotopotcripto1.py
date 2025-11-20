import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager: Dynamic V7", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #6200EA; color: white; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

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
@st.cache_data(ttl=21600)
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

# --- DİNAMİK AĞIRLIK OPTİMİZASYONU ---
def optimize_dynamic_weights(df, params, alloc_capital, validation_days=21):
    df = df.copy()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    if len(df) < validation_days + 5: 
        return (0.7, 0.3)
    
    train_df = df.iloc[:-validation_days]
    test_df = df.iloc[-validation_days:]
    
    X = train_df[['log_ret','range']].values
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    model = GaussianHMM(n_components=params['n_states'], covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_s)
    train_df['state'] = model.predict(X_s)
    
    state_stats = train_df.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    weight_candidates = np.linspace(0.1, 0.9, 9)
    best_roi = -np.inf
    best_w = (0.5, 0.5)
    
    for w_hmm in weight_candidates:
        w_score = 1 - w_hmm
        cash = alloc_capital
        coin_amt = 0
        for idx,row in test_df.iterrows():
            X_test = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_test)[0]==bull_state else (-1 if model.predict(X_test)[0]==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price = row['close']
            if decision>0.25: coin_amt = cash/price; cash=0
            elif decision<-0.25: cash = coin_amt*price; coin_amt=0
        final_val = cash + coin_amt*test_df['close'].iloc[-1]
        roi = (final_val - alloc_capital)/alloc_capital
        if roi > best_roi: best_roi = roi; best_w = (w_hmm, w_score)
    
    return best_w

# --- MULTI-TIMEFRAME STRATEJİ ---
def run_strategy(df, params, alloc_capital):
    n_states = params['n_states']
    commission = params['commission']
    timeframes = {'GÜNLÜK':'D','HAFTALIK':'W','AYLIK':'M'}
    best_roi = -np.inf
    best_portfolio = []
    best_config = {}
    
    w_hmm, w_score = optimize_dynamic_weights(df, params, alloc_capital)
    
    for tf_name, tf_code in timeframes.items():
        if tf_code=='D': df_tf = df.copy()
        else:
            agg = {'close':'last','high':'max','low':'min'}
            if 'open' in df.columns: agg['open']='first'
            if 'volume' in df.columns: agg['volume']='sum'
            df_tf = df.resample(tf_code).agg(agg).dropna()
        if len(df_tf)<5: continue
        
        df_tf['log_ret'] = np.log(df_tf['close']/df_tf['close'].shift(1))
        df_tf['range'] = (df_tf['high']-df_tf['low'])/df_tf['close']
        df_tf['custom_score'] = calculate_custom_score(df_tf)
        df_tf.dropna(inplace=True)
        if len(df_tf)<5: continue
        
        X = df_tf[['log_ret','range']].values
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
        model.fit(X_s)
        df_tf['state'] = model.predict(X_s)
        
        state_stats = df_tf.groupby('state')['log_ret'].mean()
        bull_state = state_stats.idxmax()
        bear_state = state_stats.idxmin()
        
        cash = alloc_capital
        coin_amt = 0
        portfolio_history = []
        last_day_info = {}
        
        for idx,row in df_tf.iterrows():
            price = row['close']
            hmm_signal = 1 if row['state']==bull_state else (-1 if row['state']==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            
            target_pct = 1.0 if decision>0.25 else (0.0 if decision<-0.25 else (coin_amt*price)/(cash+coin_amt*price))
            current_val = cash + coin_amt*price
            if current_val<=0: portfolio_history.append(0); continue
            current_pct = (coin_amt*price)/current_val
            if abs(target_pct-current_pct)>0.05:
                diff = (target_pct-current_pct)*current_val
                fee = abs(diff)*commission
                if diff>0 and cash>=diff: coin_amt += (diff-fee)/price; cash-=diff
                elif diff<0 and coin_amt*price>=abs(diff): coin_amt -= abs(diff)/price; cash+=abs(diff-fee)
            portfolio_history.append(cash+coin_amt*price)
            
            if idx==df_tf.index[-1]:
                regime_label = "BOĞA" if hmm_signal==1 else ("AYI" if hmm_signal==-1 else "YATAY")
                last_day_info = {"Fiyat":price, "HMM":regime_label, "Puan":int(row['custom_score']),
                                 "Öneri":"AL" if decision>0.25 else ("SAT" if decision<-0.25 else "BEKLE"),
                                 "Zaman":tf_name, "Ağırlık":f"%{int(w_hmm*100)} HMM / %{int(w_score*100)} Puan"}
        
        if len(portfolio_history)>0 and portfolio_history[-1]>best_roi:
            best_roi = portfolio_history[-1]
            best_portfolio = pd.Series(portfolio_history, index=df_tf.index)
            best_config = last_day_info
    
    return best_portfolio, best_config

# --- ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Dynamic V7")
st.markdown("### ⚔️ HMM+Puan | Dinamik Ağırlık Optimizasyonu | Güncel + Geçmiş Veriye Odaklı")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    tickers = st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital = st.number_input("Kasa ($)",10000)
    st.info("Sistem her coin için güncel ve geçmiş veriye bakarak en iyi HMM/Puan ağırlığını belirler ve son sinyali gösterir.")

if st.button("BÜYÜK TURNUVAYI BAŞLAT 🚀"):
    if not tickers: st.error("Coin seçmelisin.")
    else:
        capital_per_coin = initial_capital / len(tickers)
        results_list=[]
        total_balance=0
        total_hodl=0
        bar=st.progress(0)
        status=st.empty()
        
        params={'n_states':3,'commission':0.001}
        
        for i,ticker in enumerate(tickers):
            status.text(f"Turnuva Oynanıyor: {ticker}...")
            df=get_data_cached(ticker,"2018-01-01")
            if df is not None:
                res_series,best_conf = run_strategy(df, params, capital_per_coin)
                if res_series is not None:
                    final_val=res_series.iloc[-1]
                    total_balance+=final_val
                    start_price=df['close'].iloc[0]
                    end_price=df['close'].iloc[-1]
                    hodl_val=(capital_per_coin/start_price)*end_price
                    total_hodl+=hodl_val
                    if best_conf:
                        best_conf.update({"Coin":ticker,"Bakiye":final_val,"ROI":((final_val-capital_per_coin)/capital_per_coin)*100})
                        results_list.append(best_conf)
            bar.progress((i+1)/len(tickers))
        
        status.empty()
        
        if results_list:
            roi_total = ((total_balance-initial_capital)/initial_capital)*100
            alpha = total_balance - total_hodl
            c1,c2,c3 = st.columns(3)
            c1.metric("Toplam Bakiye", f"${total_balance:,.0f}", f"%{roi_total:.1f}")
            c2.metric("HODL Değeri", f"${total_hodl:,.0f}")
            c3.metric("Alpha (Fark)", f"${alpha:,.0f}", delta_color="normal" if alpha>0 else "inverse")
            
            df_res=pd.DataFrame(results_list)
            
            def highlight_decision(val):
                val_str=str(val)
                if val_str=='AL': return 'background-color:#00c853;color:white;font-weight:bold'
                if val_str=='SAT': return 'background-color:#d50000;color:white;font-weight:bold'
                return 'background-color:#ffd600;color:black'
            
            cols=['Coin','Fiyat','Öneri','Zaman','Ağırlık','HMM','Puan','ROI']
            st.dataframe(df_res[cols].style.applymap(highlight_decision, subset=['Öneri']).format({"Fiyat":"${:,.2f}","ROI":"%{:.1f}"}))
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")
