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
from google.oauth2.service_account import Credentials
from deap import base, creator, tools, algorithms

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Kalman AI Trader - Auto", layout="wide")
st.title("Kalman AI Trader — Full Enhanced (Otonom & Sheets Bağlantılı)")

# -------------------- AYARLAR --------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    default_tickers = ["BTC-USD","ETH-USD","SOL-USD","BNB-USD","XRP-USD","AVAX-USD","DOGE-USD","LTC-USD","ADA-USD","MATIC-USD"]
    selected_tickers = st.multiselect("Sepet", default_tickers, default=default_tickers)
    capital = st.number_input("Coin Başı Başlangıç ($)", value=10.0)
    window_size = st.slider("Öğrenme Penceresi (Bar Sayısı)", 20, 60, 30)
    use_ga = st.checkbox("Genetic Algoritma (GA) Aktif", value=False)
    ga_generations = st.number_input("GA Jenerasyon Sayısı", min_value=1, max_value=200, value=8)
    
    st.divider()
    st.header("🤖 Otomasyon")
    update_interval = st.number_input("Döngü Hızı (Saniye)", min_value=60, max_value=3600, value=300, help="Otomatik modda kaç saniyede bir analiz yapsın?")

# -------------------- VERİ KAYIT FONKSİYONLARI --------------------
def init_gsheet(sheet_name="KalmanAI_Results", creds_file="service_account.json"):
    """Google Sheets bağlantısını kurar, yoksa oluşturur."""
    try:
        scopes = ['https://www.googleapis.com/auth/spreadsheets']
        creds = Credentials.from_service_account_file(creds_file, scopes=scopes)
        client = gspread.authorize(creds)
        try:
            sheet = client.open(sheet_name).sheet1
        except gspread.SpreadsheetNotFound:
            sheet = client.create(sheet_name).sheet1
            sheet.append_row(["timestamp", "ticker", "tf", "final_balance", "roi", "hodl_val", "action_signal"])
        return sheet
    except Exception as e:
        st.error(f"Google Sheets Hatası: {e}")
        return None

def append_to_gsheet(sheet, data_dict):
    """Veriyi sheete ekler."""
    if sheet:
        try:
            # Sıralama: Timestamp, Ticker, TF, Final, ROI, Hodl
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
            print(f"Sheet yazma hatası: {e}")

def save_to_csv(data_dict, csv_file="kalman_results.csv"):
    """Yedek olarak CSV'ye kaydeder."""
    df = pd.DataFrame([data_dict])
    df['timestamp'] = pd.Timestamp.now()
    if os.path.exists(csv_file):
        df.to_csv(csv_file, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_file, mode='w', header=True, index=False)

# -------------------- CORE MANTIK (KALMAN & DATA) --------------------
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
    long_w = {'D':252, 'W':52, 'M':36}.get(timeframe_code, 36)
    df['ma5_long_mean'] = df['ma5'].rolling(window=long_w, min_periods=10).mean()
    df['ma5_long_std'] = df['ma5'].rolling(window=long_w, min_periods=10).std()
    df['ma5_score'] = (df['ma5'] - df['ma5_long_mean']) / (df['ma5_long_std'] + 1e-9)
    df['ma5_score'].fillna(0, inplace=True)
    return df

def process_data(df, timeframe):
    if df is None or len(df) < 100: return None
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
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
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    return df_res

# -------------------- AI MODELLERİ & STRATEJİ --------------------
def walk_forward_splits(df, n_splits=3, test_size_ratio=0.2):
    n = len(df)
    test_size = max(int(n * test_size_ratio), 10)
    step = max(int((n - test_size) / (n_splits + 1)), 1)
    splits = []
    for i in range(n_splits):
        train_end = step * (i + 1)
        val_start = train_end
        val_end = val_start + step
        test_start = val_end
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 5: break
        splits.append((slice(0, train_end), slice(val_start, val_end), slice(test_start, test_end)))
    if not splits:
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        splits = [(slice(0, train_end), slice(train_end, val_end), slice(val_end, n))]
    return splits

def train_models_for_window(train_df, rf_depth=5, xgb_params=None, n_hmm=3):
    features = ['log_ret','range','trend_signal','ma5_score']
    X = train_df[features]
    y = train_df['target']
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    if xgb_params is None:
        xgb_params = {'n_estimators':30, 'max_depth':3, 'learning_rate':0.1,'tree_method':'hist','n_jobs':-1}
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
    clf_xgb.fit(X, y)
    
    meta_clf = None
    try:
        meta_X = np.vstack([clf_rf.predict_proba(X)[:,1], clf_xgb.predict_proba(X)[:,1]]).T
        meta_clf = LogisticRegression(max_iter=200)
        meta_clf.fit(meta_X, y)
    except: pass
    
    hmm_model = None
    try:
        X_hmm = train_df[['log_ret','range']].values
        Xh_s = StandardScaler().fit_transform(X_hmm)
        hmm_model = GaussianHMM(n_components=n_hmm, covariance_type='diag', n_iter=50, random_state=42)
        hmm_model.fit(Xh_s)
    except: pass
    
    return {'rf': clf_rf, 'xgb': clf_xgb, 'meta': meta_clf, 'scaler': scaler, 'hmm': hmm_model}

def predict_with_models(models, row):
    rf_prob = xgb_prob = 0.5
    stack_sig = hmm_sig = 0.0
    try:
        features = ['log_ret','range','trend_signal','ma5_score']
        Xrow = row[features].values.reshape(1,-1)
        rf_prob = models['rf'].predict_proba(pd.DataFrame(Xrow, columns=features))[0][1]
        xgb_prob = models['xgb'].predict_proba(pd.DataFrame(Xrow, columns=features))[0][1]
    except: pass
    
    try:
        if models['meta']:
            stack_prob = models['meta'].predict_proba(np.array([[rf_prob,xgb_prob]]))[0][1]
            stack_sig = (stack_prob-0.5)*2
        else:
            stack_sig = ((rf_prob+xgb_prob)/2-0.5)*2
    except: stack_sig = ((rf_prob+xgb_prob)/2-0.5)*2
    
    try:
        if models['hmm']:
            Xh = row[['log_ret','range']].values.reshape(1,-1)
            probs = models['hmm'].predict_proba(StandardScaler().fit_transform(Xh))[0]
            bull = np.argmax(models['hmm'].means_[:,0])
            bear = np.argmin(models['hmm'].means_[:,0])
            hmm_sig = probs[bull]-probs[bear]
    except: hmm_sig = 0.0
    
    k_trend = row['trend_signal']
    combined = hmm_sig*0.25 + stack_sig*0.35 + k_trend*0.4
    return combined

def simulate_walk_forward(df, start_cap, win_size, params=None):
    if params is None: params={}
    rf_depth = params.get('rf_depth',5)
    xgb_params = params.get('xgb_params',None)
    buy_t = params.get('buy_th',0.25)
    sell_t = params.get('sell_th',-0.25)
    
    splits = walk_forward_splits(df, n_splits=3)
    equity, dates = [], []
    cash = start_cap
    coin = 0
    
    for tr,val,tst in splits:
        train_df = df.iloc[0:val.stop] # Kümülatif öğrenme
        test_df = df.iloc[tst]
        if len(test_df)==0: continue
        
        models = train_models_for_window(train_df, rf_depth=rf_depth, xgb_params=xgb_params)
        
        for idx in test_df.index:
            row = df.loc[idx]
            sig = predict_with_models(models, row)
            price = row['close']
            if sig > buy_t and cash > 0:
                coin = cash/price; cash = 0
            elif sig < sell_t and coin > 0:
                cash = coin*price; coin = 0
            equity.append(cash + coin*price)
            dates.append(idx)
            
    if not equity: # Fallback: Trend Signal
        for i in range(len(df)):
            sig = df['trend_signal'].iloc[i]
            price = df['close'].iloc[i]
            if sig>0 and cash>0: coin=cash/price; cash=0
            elif sig<0 and coin>0: cash=coin*price; coin=0
            equity.append(cash+coin*price)
            dates.append(df.index[i])
            
    final = equity[-1]
    roi = (final-start_cap)/start_cap
    return {'final':final, 'roi':roi, 'equity':equity, 'dates':dates}

# -------------------- GA & MAIN RUNNER --------------------
def ga_optimize_params_light(df, start_cap, win_size, n_gen=8, pop_size=10):
    creator.create('FitnessMax', base.Fitness, weights=(1.0,), overwrite=True)
    creator.create('Individual', list, fitness=creator.FitnessMax, overwrite=True)
    toolbox = base.Toolbox()
    toolbox.register('rf_depth', np.random.randint, 3, 13)
    toolbox.register('xgb_max_depth', np.random.randint, 2, 7)
    toolbox.register('xgb_eta', np.random.uniform, 0.01, 0.3)
    toolbox.register('buy_th', np.random.uniform, 0.05, 0.5)
    toolbox.register('sell_th', np.random.uniform, -0.5, -0.05)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.rf_depth, toolbox.xgb_max_depth, toolbox.xgb_eta, toolbox.buy_th, toolbox.sell_th), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    def eval_individual(ind):
        rf_depth, xgb_md, xgb_eta, buy_th, sell_th = ind
        xgb_params = {'max_depth':int(xgb_md), 'learning_rate':float(xgb_eta), 'n_estimators':30, 'tree_method':'hist', 'n_jobs':-1}
        params = {'rf_depth':int(rf_depth), 'xgb_params':xgb_params, 'buy_th':float(buy_th), 'sell_th':float(sell_th)}
        
        splits = walk_forward_splits(df, n_splits=3)
        rois = []
        for tr,val,tst in splits:
            tr_df = df.iloc[0:val.stop]
            tst_df = df.iloc[tst]
            res = simulate_walk_forward(pd.concat([tr_df,tst_df]), start_cap, win_size, params=params)
            rois.append(res['roi'])
        return (np.mean(rois),)
        
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, stats=stats, halloffame=hof, verbose=False)
    
    best = hof[0]
    return {
        'rf_depth': int(best[0]),
        'xgb_params': {'max_depth':int(best[1]), 'learning_rate':float(best[2]), 'n_estimators':30, 'tree_method':'hist', 'n_jobs':-1},
        'buy_th': float(best[3]), 'sell_th': float(best[4])
    }

def run_strategy_enhanced(ticker, start_cap, win_size, use_ga_flag=False, ga_gens=8):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    raw_df = raw_df.iloc[-1460:] # Son 4 yıl
    
    best_roi = -9999
    best_res = None
    timeframes = {'Günlük':'D', 'Haftalık':'W', 'Aylık':'M'}
    
    for tf_name, tf_code in timeframes.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        params = None
        if use_ga_flag:
            try:
                params = ga_optimize_params_light(df, start_cap, win_size, n_gen=ga_gens)
            except: params = None
            
        sim = simulate_walk_forward(df, start_cap, win_size, params=params)
        
        if sim['roi'] > best_roi:
            start_p = df['close'].iloc[0]
            end_p = df['close'].iloc[-1]
            hodl_val = (start_cap/start_p)*end_p
            best_roi = sim['roi']
            best_res = {
                'ticker': ticker, 'tf': tf_name, 'final': sim['final'],
                'roi': sim['roi'], 'hodl': hodl_val, 'equity': sim['equity'],
                'dates': sim['dates']
            }
    return best_res

# -------------------- ARKA PLAN OTOMASYONU --------------------
stop_flag = False

def background_loop():
    """Arka planda sürekli çalışacak döngü."""
    global stop_flag
    # Sheet başlat (Varsa bağlan, yoksa oluştur)
    sheet = init_gsheet()
    
    while not stop_flag:
        # Analizi çalıştır
        for t in selected_tickers:
            if stop_flag: break # Döngü ortasında durdurma kontrolü
            
            res = run_strategy_enhanced(t, capital, window_size, use_ga_flag=use_ga, ga_gens=int(ga_generations))
            if res:
                data_dict = {
                    'ticker': res['ticker'],
                    'tf': res['tf'],
                    'final': res['final'],
                    'roi': res['roi'],
                    'hodl': res['hodl']
                }
                # Sheet'e yaz
                if sheet: append_to_gsheet(sheet, data_dict)
                # CSV'ye yedekle
                save_to_csv(data_dict)
        
        # Bekleme süresi (Saniye)
        for _ in range(int(update_interval)):
            if stop_flag: break
            time.sleep(1)

# -------------------- ARAYÜZ KONTROLLERİ --------------------
col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    if st.button("🛡️ TEK SEFERLİK ANALİZ BAŞLAT"):
        cols = st.columns(2)
        prog = st.progress(0)
        results = []
        start_time = time.time()
        
        for i, t in enumerate(selected_tickers):
            with cols[i % 2]:
                with st.spinner(f"{t} analiz ediliyor..."):
                    res = run_strategy_enhanced(t, capital, window_size, use_ga_flag=use_ga, ga_gens=int(ga_generations))
                
                if res:
                    results.append(res)
                    is_profit = res['roi'] > 0
                    color = "#00ff00" if is_profit else "#ff4444"
                    st.markdown(f"**{t} — {res['tf']}** — ROI: {res['roi']:.2%}")
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=res['dates'], y=res['equity'], name="Bot"))
                    fig.add_trace(go.Scatter(x=[res['dates'][0], res['dates'][-1]], y=[capital, res['hodl']], name="HODL", line=dict(dash='dot')))
                    fig.update_layout(height=200, margin=dict(t=0,b=0,l=0,r=0), template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)
            
            prog.progress((i + 1) / len(selected_tickers))
        
        prog.empty()
        if results:
            total_final = sum([r['final'] for r in results])
            total_hodl = sum([r['hodl'] for r in results])
            total_start = capital * len(results)
            
            st.markdown('---')
            c1, c2, c3 = st.columns(3)
            c1.metric('Başlangıç', f"${total_start:.2f}")
            c2.metric('Bot Bitiş', f"${total_final:.2f}", f"%{((total_final-total_start)/total_start)*100:.2f}")
            c3.metric('HODL Bitiş', f"${total_hodl:.2f}", delta=f"${total_final-total_hodl:.2f}")
        
        st.success(f"Analiz Tamamlandı ({time.time()-start_time:.1f}s) ✅")

with col_btn2:
    # Otomasyon Kontrolleri
    if st.button("🔄 OTOMATİK MODU BAŞLAT (Sheets)"):
        stop_flag = False
        # Thread başlat
        t = threading.Thread(target=background_loop, daemon=True)
        t.start()
        st.success(f"Otomatik mod başlatıldı! Her {update_interval} saniyede bir Google Sheets güncellenecek.")

    if st.button("🛑 OTOMATİK MODU DURDUR"):
        stop_flag = True
        st.warning("Otomatik mod durduruluyor...")
