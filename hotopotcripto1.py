import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Hedge Fund Manager: Dynamic V7 (Alpha Odaklı)", layout="wide", initial_sidebar_state="expanded")

# --- CSS STİL ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #00796B; color: white; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

# --- CUSTOM SCORE ---
def calculate_custom_score(df):
    """5'li Puanlama Sistemi (-7 ile +7 arası)"""
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
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        df.dropna(inplace=True)
        return df
    except:
        return None

# --- DİNAMİK AĞIRLIK OPTİMİZASYONU ---
def optimize_dynamic_weights(df, n_states, commission, alloc_capital, validation_days=21):
    """
    HMM + Custom Score ağırlıklarını son 3 haftalık (21 gün) validation verisine göre optimize eder.
    En iyi ROI veren ağırlık kombinasyonu seçilir.
    """
    df = df.copy()
    df['log_ret'] = np.log(df['close']/df['close'].shift(1))
    df['range'] = (df['high']-df['low'])/df['close']
    df['custom_score'] = calculate_custom_score(df)
    df.dropna(inplace=True)
    
    if len(df) < validation_days + 10: return 0.7, 0.3

    train_data = df.iloc[:-validation_days]
    val_data = df.iloc[-validation_days:]
    
    # HMM eğitimi (Uzun Dönem Veri)
    X_train = train_data[['log_ret','range']].values
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=42)
    model.fit(X_train_s)
    
    train_data['state'] = model.predict(X_train_s)
    state_stats = train_data.groupby('state')['log_ret'].mean()
    bull_state = state_stats.idxmax()
    bear_state = state_stats.idxmin()
    
    best_roi = -np.inf
    best_w = (0.7,0.3)
    
    # Grid search HMM ağırlığı (0.1'den 0.95'e kadar 0.05'lik adımlarla)
    for w_hmm in np.arange(0.1, 1.0, 0.05):
        w_score = 1 - w_hmm
        cash = alloc_capital
        coin_amt = 0
        
        # Validation Simülasyonu
        for idx,row in val_data.iterrows():
            # HMM sinyali için val datası özelliklerini ölçeklendir
            X_point = scaler.transform([[row['log_ret'], row['range']]])
            hmm_signal = 1 if model.predict(X_point)[0]==bull_state else (-1 if model.predict(X_point)[0]==bear_state else 0)
            score_signal = 1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision = w_hmm*hmm_signal + w_score*score_signal
            price = row['close']
            
            # --- Karar Eşiği (Validation) ---
            if decision>0.2: # Sinyali 0.2'ye düşürerek daha az agresif sinyaller üretir
                coin_amt = cash/price
                cash = 0
            elif decision<-0.2:
                cash = coin_amt*price
                coin_amt = 0
            # ---------------------------------

        final_val = cash + coin_amt*val_data['close'].iloc[-1]
        roi = (final_val - alloc_capital)/alloc_capital
        if roi > best_roi:
            best_roi = roi
            best_w = (w_hmm,w_score)
            
    return best_w

# --- MULTI-TIMEFRAME BACKTEST ---
def run_multi_timeframe(df_raw, params, alloc_capital):
    n_states=params['n_states']
    commission=params['commission']
    timeframes={'GÜNLÜK':'D','HAFTALIK':'W','AYLIK':'M'}
    
    best_roi=-np.inf
    best_portfolio=[]
    best_config={}
    
    # Adım 1: En iyi HMM/Puan ağırlığını optimize et
    w_hmm,w_score = optimize_dynamic_weights(df_raw,n_states,commission,alloc_capital)
    
    # Adım 2: Tüm zaman dilimleri ve tüm data üzerinde backtest yap
    for tf_name,tf_code in timeframes.items():
        if tf_code=='D':
            df=df_raw.copy()
        else:
            agg={'close':'last','high':'max','low':'min'}
            if 'open' in df_raw.columns: agg['open']='first'
            if 'volume' in df_raw.columns: agg['volume']='sum'
            df=df_raw.resample(tf_code).agg(agg).dropna()
        if len(df)<50: continue
        
        df['log_ret']=np.log(df['close']/df['close'].shift(1))
        df['range']=(df['high']-df['low'])/df['close']
        df['custom_score']=calculate_custom_score(df)
        df.dropna(inplace=True)
        
        X=df[['log_ret','range']].values
        scaler=StandardScaler()
        X_s=scaler.fit_transform(X)
        try:
            model=GaussianHMM(n_components=n_states,covariance_type='full',n_iter=100,random_state=42)
            model.fit(X_s)
            df['state']=model.predict(X_s)
        except: continue
        
        state_stats=df.groupby('state')['log_ret'].mean()
        bull_state=state_stats.idxmax()
        bear_state=state_stats.idxmin()
        
        cash=alloc_capital
        coin_amt=0
        portfolio=[]
        history={}
        
        for idx,row in df.iterrows():
            price=row['close']
            hmm_signal=1 if row['state']==bull_state else (-1 if row['state']==bear_state else 0)
            score_signal=1 if row['custom_score']>=3 else (-1 if row['custom_score']<=-3 else 0)
            decision=w_hmm*hmm_signal + w_score*score_signal
            
            target_pct=None
            action_text="BEKLE"
            
            # --- Karar Eşiği (Backtest) ---
            if decision>0.2: 
                target_pct=1.0; action_text="AL"
            elif decision<-0.2:
                target_pct=0.0; action_text="SAT"
            # -------------------------------
            
            if target_pct is not None:
                curr_val=cash+coin_amt*price
                curr_pct=(coin_amt*price)/curr_val if curr_val>0 else 0
                
                # İşlem yap (Portföy Yeniden Dengeleme)
                diff=(target_pct-curr_pct)*curr_val
                fee=abs(diff)*commission
                if diff>0:
                    if cash>=diff: coin_amt+=(diff-fee)/price; cash-=diff
                else:
                    sell=abs(diff)
                    if coin_amt*price>=sell: coin_amt-=sell/price; cash+=sell-fee
            
            portfolio.append(cash+coin_amt*price)
            if idx==df.index[-1]:
                regime='BOĞA' if hmm_signal==1 else ('AYI' if hmm_signal==-1 else 'YATAY')
                history={"Fiyat":price,"HMM":regime,"Puan":int(row['custom_score']),"Öneri":action_text,
                         "Zaman":tf_name,"Ağırlık":f"%{int(w_hmm*100):.0f} HMM / %{int(w_score*100):.0f} Puan"}
        
        if portfolio[-1]>best_roi:
            best_roi=portfolio[-1]
            best_portfolio=pd.Series(portfolio,index=df.index)
            best_config=history
            
    return best_portfolio,best_config

# --- ARAYÜZ ---
st.title("🏆 Hedge Fund Manager: Dynamic HMM+Score V7")
st.markdown("### 📊 Alpha Odaklı, Dinamik Ağırlık Optimizasyonlu Backtest")

with st.sidebar:
    st.header("Ayarlar")
    default_tickers=["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","ADA-USD"]
    tickers=st.multiselect("Analiz Edilecek Coinler", default_tickers, default=default_tickers)
    initial_capital=st.number_input("Kasa ($)",10000)
    st.info("Sistem geçmiş veriyi dikkate alır ve **son 3 haftayı optimize ederek** HMM/Puan ağırlığını belirler. Alpha (HODL'dan fark) pozitifliği hedeflenmiştir.")

if st.button("BÜYÜK TURNUVAYI BAŞLAT 🚀"):
    if not tickers: st.error("Coin seçmelisin.")
    else:
        capital_per_coin=initial_capital/len(tickers)
        results=[]
        total_val=0
        total_hodl=0
        bar=st.progress(0)
        
        # Sadece 2018'den itibaren çek (Maksimum geçmiş)
        start_date_yf="2018-01-01" 
        
        for i,ticker in enumerate(tickers):
            bar.progress(i/len(tickers))
            
            df=get_data_cached(ticker,start_date_yf)
            
            if df is not None:
                port,conf=run_multi_timeframe(df,{'n_states':3,'commission':0.001},capital_per_coin)
                
                if port is not None:
                    final_val=port.iloc[-1]
                    total_val+=final_val
                    
                    # HODL Hesaplama (İlk günden itibaren)
                    hodl_val=(capital_per_coin/df['close'].iloc[0])*df['close'].iloc[-1]
                    total_hodl+=hodl_val
                    
                    if conf:
                        conf.update({"Coin":ticker,"Bakiye":final_val,"ROI":(final_val-capital_per_coin)/capital_per_coin*100})
                        results.append(conf)
                        
        bar.progress(1.0)
        st.success("Turnuva tamamlandı ✅")
        
        # Sonuç Metrikleri
        c1,c2,c3=st.columns(3)
        roi_total=(total_val-initial_capital)/initial_capital*100
        alpha=total_val-total_hodl
        
        c1.metric("Toplam Bakiye",f"${total_val:,.0f}",f"%{roi_total:.1f}")
        c2.metric("HODL Değeri",f"${total_hodl:,.0f}")
        c3.metric("Alpha (Fark)",f"${alpha:,.0f}",delta_color="normal" if alpha>0 else "inverse")
        
        if results:
            df_res=pd.DataFrame(results)
            def highlight(val):
                if val=='AL': return 'background-color: #00c853; color:white; font-weight:bold'
                if val=='SAT': return 'background-color: #d50000; color:white; font-weight:bold'
                return ''
                
            st.dataframe(df_res[['Coin','Fiyat','Öneri','Zaman','Ağırlık','HMM','Puan','ROI']].style.applymap(highlight,subset=['Öneri']).format({"Fiyat":"${:,.2f}","ROI":"%{:.1f}"}))
        else:
            st.error("Veri alınamadı veya hesaplanamadı.")
