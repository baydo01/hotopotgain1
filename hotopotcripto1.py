import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import plotly.graph_objects as go
import time
import threading
import warnings
import os
import gspread
import json
from google.oauth2.service_account import Credentials
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Kalman AI Trader Auto", layout="wide")
st.title("Kalman AI Trader — Full Enhanced (Otomatik & Oto-Kurulum)")

# -------------------- AYARLAR --------------------
# Eğer GitHub Secrets veya JSON varsa onu kullan, yoksa manuel path
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"

with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","LTC-USD","ADA-USD","MATIC-USD"]
    selected_tickers = st.multiselect("Sepet (Tüm coinleri seçebilirsiniz)", default_tickers, default=default_tickers)
    capital = st.number_input("Coin Başı Başlangıç ($)", value=10.0)
    window_size = st.slider("Öğrenme Penceresi (Bar Sayısı)", 20, 60, 30)
    use_ga = st.checkbox("Genetic Algoritma ile parametre optimizasyonu (Ağır)", value=False)
    ga_generations = st.number_input("GA generations", min_value=1, max_value=200, value=10)
    update_interval = st.number_input("Otomatik güncelleme aralığı (saniye)", min_value=60, max_value=3600, value=300)

# -------------------- GOOGLE SHEETS BAĞLANTISI & OTO-KURULUM --------------------
def init_gsheet(sheet_id=SHEET_ID):
    """Google Sheets bağlantısını kurar ve GEREKİRSE BAŞLIKLARI EKLER."""
    scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    
    creds = None
    # 1. Secrets Kontrolü (Streamlit Cloud için)
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        except: pass
    
    # 2. Yerel Dosya Kontrolü
    if not creds and os.path.exists(CREDENTIALS_FILE):
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
        
    if not creds:
        st.error("Google kimlik bilgisi bulunamadı!")
        return None

    try:
        client = gspread.authorize(creds)
        sheet = client.open_by_key(sheet_id).sheet1
        
        # --- OTO-KURULUM BAŞLANGIÇ ---
        try:
            existing_data = sheet.get_all_values()
            required_headers = ["timestamp", "ticker", "tf", "final_balance", "roi", "hodl_val"]
            
            # Eğer sayfa boşsa veya başlıklar yanlışsa
            if not existing_data or existing_data[0] != required_headers:
                print("⚠️ Sheet formatı düzeltiliyor...")
                sheet.clear()
                sheet.append_row(required_headers)
                print("✅ Başlıklar eklendi.")
        except Exception as e:
            print(f"Oto-kurulum hatası: {e}")
        # --- OTO-KURULUM BİTİŞ ---
        
        return sheet
    except Exception as e:
        st.error(f"Sheet Bağlantı Hatası: {e}")
        return None

def append_to_gsheet(sheet, data_dict):
    if sheet:
        try:
            row = [
                pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                data_dict['ticker'],
                data_dict['tf'],
                float(f"{data_dict['final']:.2f}"),
                f"{data_dict['roi']:.2%}",
                float(f"{data_dict['hodl']:.2f}")
            ]
            sheet.append_row(row)
        except Exception as e:
            print(f"Yazma hatası: {e}")

def save_to_csv(data_dict, csv_file="kalman_results.csv"):
    df = pd.DataFrame([data_dict])
    df['timestamp'] = pd.Timestamp.now()
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

# -------------------- KALMAN FİLTRESİ --------------------
def apply_kalman_filter(prices):
    n_iter = len(prices)
    sz = (n_iter,)
    Q = 1e-5
    R = 0.01 ** 2
    xhat = np.zeros(sz)
    P = np.zeros(sz)
    xhatminus = np.zeros(sz)
    Pminus = np.zeros(sz)
    K = np.zeros(sz)
    xhat[0] = prices.iloc[0]
    P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return pd.Series(xhat, index=prices.index)

# -------------------- VERİ --------------------
def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        return df
    except Exception:
        return None

def add_ma5_and_score(df, timeframe_code):
    df['ma5'] = df['close'].rolling(window=5).mean()
    long_w = {'D':252,'W':52,'M':36}.get(timeframe_code,'D')
    # timeframes dictindeki değerler string olduğu için default 252 (D) olarak kalsın
    if timeframe_code == 'D': win=252
    elif timeframe_code == 'W': win=52
    else: win=36
    
    df['ma5_long_mean'] = df['ma5'].rolling(window=win, min_periods=10).mean()
    df['ma5_long_std'] = df['ma5'].rolling(window=win, min_periods=10).std()
    df['ma5_score'] = (df['ma5'] - df['ma5_long_mean']) / (df['ma5_long_std'] + 1e-9)
    df['ma5_score'].fillna(0, inplace=True)
    return df

def process_data(df, timeframe):
    if df is None or len(df) < 100:
        return None
    agg_dict = {'open':'first','high':'max','low':'min','close':'last','volume':'sum'}
    if timeframe == 'W':
        df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M':
        df_res = df.resample('ME').agg(agg_dict).dropna()
    else:
        df_res = df.copy()
    if len(df_res) < 50: return None
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close']/df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low'])/df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    df_res = add_ma5_and_score(df_res, timeframe)
    df_res.replace([np.inf,-np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

# -------------------- WALK-FORWARD --------------------
def walk_forward_splits(df,n_splits=3,test_size_ratio=0.2):
    n=len(df)
    test_size = max(int(n*test_size_ratio),10)
    step = max(int((n-test_size)/(n_splits+1)),1)
    splits=[]
    for i in range(n_splits):
        train_end = step*(i+1)
        val_start = train_end
        val_end = val_start+step
        test_start = val_end
        test_end = min(test_start+test_size,n)
        if test_end-test_start<5: break
        splits.append((slice(0,train_end),slice(val_start,val_end),slice(test_start,test_end)))
    if not splits:
        train_end=int(n*0.6)
        val_end=int(n*0.8)
        splits=[(slice(0,train_end),slice(train_end,val_end),slice(val_end,n))]
    return splits

# -------------------- MODEL EĞİTİM --------------------
def train_models_for_window(train_df, rf_depth=5, xgb_params=None, n_hmm=3):
    features=['log_ret','range','trend_signal','ma5_score']
    X=train_df[features]; y=train_df['target']
    scaler=StandardScaler()
    X_s=scaler.fit_transform(X)
    clf_rf=RandomForestClassifier(n_estimators=30,max_depth=rf_depth,n_jobs=-1,random_state=42)
    clf_rf.fit(X,y)
    if xgb_params is None:
        xgb_params={'n_estimators':30,'max_depth':3,'learning_rate':0.1,'tree_method':'hist','n_jobs':-1}
    clf_xgb=xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
    clf_xgb.fit(X,y)
    try:
        meta_X=np.vstack([clf_rf.predict_proba(X)[:,1], clf_xgb.predict_proba(X)[:,1]]).T
        meta_clf=LogisticRegression(max_iter=200)
        meta_clf.fit(meta_X,y)
    except:
        meta_clf=None
    hmm_model=None
    try:
        Xh=train_df[['log_ret','range']].values
        Xh_s=StandardScaler().fit_transform(Xh)
        hmm_model=GaussianHMM(n_components=n_hmm,covariance_type='diag',n_iter=50,random_state=42)
        hmm_model.fit(Xh_s)
    except: pass
    return {'rf':clf_rf,'xgb':clf_xgb,'meta':meta_clf,'scaler':scaler,'hmm':hmm_model}

def predict_with_models(models,row):
    rf_prob=xgb_prob=0.5
    stack_sig=hmm_sig=0.0
    try:
        features=['log_ret','range','trend_signal','ma5_score']
        Xrow=row[features].values.reshape(1,-1)
        rf_prob=models['rf'].predict_proba(pd.DataFrame(Xrow,columns=features))[0][1]
        xgb_prob=models['xgb'].predict_proba(pd.DataFrame(Xrow,columns=features))[0][1]
    except: pass
    try:
        if models['meta'] is not None:
            stack_prob=models['meta'].predict_proba(np.array([[rf_prob,xgb_prob]]))[0][1]
            stack_sig=(stack_prob-0.5)*2
        else:
            stack_sig=((rf_prob+xgb_prob)/2-0.5)*2
    except:
        stack_sig=((rf_prob+xgb_prob)/2-0.5)*2
    try:
        if models['hmm'] is not None:
            Xh=row[['log_ret','range']].values.reshape(1,-1)
            probs=models['hmm'].predict_proba(StandardScaler().fit_transform(Xh))[0]
            bull=np.argmax(models['hmm'].means_[:,0])
            bear=np.argmin(models['hmm'].means_[:,0])
            hmm_sig=probs[bull]-probs[bear]
    except: hmm_sig=0.0
    k_trend=row['trend_signal']
    combined=hmm_sig*0.25+stack_sig*0.35+k_trend*0.4
    return combined

def simulate_walk_forward(df,start_cap,win_size,params=None):
    if params is None: params={}
    rf_depth=params.get('rf_depth',5)
    xgb_params=params.get('xgb_params',None)
    buy_t=params.get('buy_th',0.25)
    sell_t=params.get('sell_th',-0.25)
    splits=walk_forward_splits(df,n_splits=3)
    equity,dates=[],[]
    cash=start_cap; coin=0
    for tr,val,tst in splits:
        train_slice=slice(0,val.stop)
        test_slice=tst
        train_df=df.iloc[train_slice]
        test_df=df.iloc[test_slice]
        if len(test_df)==0: continue
        models=train_models_for_window(train_df,rf_depth=rf_depth,xgb_params=xgb_params)
        for idx in test_df.index:
            row=df.loc[idx]
            sig=predict_with_models(models,row)
            price=row['close']
            if sig>buy_t and cash>0:
                coin=cash/price
                cash=0
            elif sig<sell_t and coin>0:
                cash=coin*price
                coin=0
            equity.append(cash+coin*price)
            dates.append(idx)
    if len(equity)==0:
        for i in range(len(df)):
            sig=df['trend_signal'].iloc[i]
            price=df['close'].iloc[i]
            if sig>0 and cash>0:
                coin=cash/price; cash=0
            elif sig<0 and coin>0:
                cash=coin*price; coin=0
            equity.append(cash+coin*price)
            dates.append(df.index[i])
    final=equity[-1]
    roi=(final-start_cap)/start_cap
    return {'final':final,'roi':roi,'equity':equity,'dates':dates}

# -------------------- GA (hafif) --------------------
def ga_optimize_params_light(df, start_cap, win_size, n_gen=8,pop_size=10):
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,), overwrite=True)
        creator.create('Individual', list, fitness=creator.FitnessMax, overwrite=True)
        
    toolbox = base.Toolbox()
    toolbox.register('rf_depth', np.random.randint,3,13)
    toolbox.register('xgb_max_depth', np.random.randint,2,7)
    toolbox.register('xgb_eta', np.random.uniform,0.01,0.3)
    toolbox.register('buy_th', np.random.uniform,0.05,0.5)
    toolbox.register('sell_th', np.random.uniform,-0.5,-0.05)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.rf_depth,toolbox.xgb_max_depth,toolbox.xgb_eta,toolbox.buy_th,toolbox.sell_th), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    def eval_individual(ind):
        rf_depth, xgb_md, xgb_eta, buy_th, sell_th = ind
        xgb_params = {'max_depth':int(xgb_md),'learning_rate':float(xgb_eta),'n_estimators':30,'tree_method':'hist','n_jobs':-1}
        params={'rf_depth':int(rf_depth),'xgb_params':xgb_params,'buy_th':float(buy_th),'sell_th':float(sell_th)}
        splits = walk_forward_splits(df,n_splits=3)
        rois=[]
        for tr,val,tst in splits:
            train_df = df.iloc[0:val.stop]
            test_df = df.iloc[tst]
            res = simulate_walk_forward(pd.concat([train_df,test_df]),start_cap,win_size,params=params)
            rois.append(res['roi'])
        return (np.mean(rois),)
        
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg',np.mean)
    stats.register('max',np.max)
    
    try:
        algorithms.eaSimple(pop,toolbox,cxpb=0.5,mutpb=0.2,ngen=n_gen,stats=stats,halloffame=hof,verbose=False)
        best = hof[0]
        rf_depth, xgb_md, xgb_eta, buy_th, sell_th = best
        return {'rf_depth':int(rf_depth),'xgb_params':{'max_depth':int(xgb_md),'learning_rate':float(xgb_eta),'n_estimators':30,'tree_method':'hist','n_jobs':-1},'buy_th':float(buy_th),'sell_th':float(sell_th)}
    except:
        return None

# -------------------- STRATEGY RUN --------------------
def run_strategy_enhanced(ticker,start_cap,win_size,use_ga_flag=False,ga_gens=8):
    raw_df=get_raw_data(ticker)
    if raw_df is None: return None
    raw_df=raw_df.iloc[-1460:]
    best_roi=-9999
    best_res=None
    timeframes={'Günlük':'D','Haftalık':'W','Aylık':'M'}
    for tf_name,tf_code in timeframes.items():
        df=process_data(raw_df,tf_code)
        if df is None: continue
        params=None
        if use_ga_flag:
            try:
                params = ga_optimize_params_light(df,start_cap,win_size,n_gen=ga_gens)
            except Exception: params=None
            
        sim=simulate_walk_forward(df,start_cap,win_size,params=params)
        if sim['roi']>best_roi:
            start_p=df['close'].iloc[0]
            end_p=df['close'].iloc[-1]
            hodl_val=(start_cap/start_p)*end_p
            best_roi=sim['roi']
            best_res={'ticker':ticker,'tf':tf_name,'final':sim['final'],'roi':sim['roi'],'hodl':hodl_val,'equity':sim['equity'],'dates':sim['dates'],'kalman_data':df['kalman_close']}
    return best_res

# -------------------- ARKA PLAN THREAD --------------------
stop_flag=False
def background_loop():
    global stop_flag
    sheet=init_gsheet() # Bağlantı ve Oto-Kurulum burada
    
    while not stop_flag:
        results=[]
        for t in selected_tickers:
            if stop_flag: break
            res=run_strategy_enhanced(t,capital,window_size,use_ga_flag=use_ga,ga_gens=int(ga_generations))
            if res:
                results.append(res)
                data_dict={'ticker':res['ticker'],'tf':res['tf'],'final':res['final'],'roi':res['roi'],'hodl':res['hodl']}
                if sheet: append_to_gsheet(sheet,data_dict)
                save_to_csv(data_dict)
        
        for _ in range(int(update_interval)):
            if stop_flag: break
            time.sleep(1)

# -------------------- STREAMLIT ARAYÜZ --------------------
col1, col2 = st.columns(2)
with col1:
    if st.button("🛡️ Otomatik Arka Planı Başlat"):
        stop_flag=False
        thread=threading.Thread(target=background_loop,daemon=True)
        thread.start()
        st.success("Otomatik analiz başlatıldı!")

with col2:
    if st.button("🛑 Otomatik Arka Planı Durdur"):
        stop_flag=True
        st.warning("Otomatik analiz durduruldu.")
