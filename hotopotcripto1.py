import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import gspread
import os
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- ML & AUTOML ---
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from hmmlearn.hmm import GaussianHMM
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# UI CONFIG
st.set_page_config(page_title="Hedge Fund AI: V8 AutoML", layout="wide", page_icon="🧠")
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {background: linear-gradient(135deg, #141E30 0%, #243B55 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #00d2ff; margin-bottom: 25px;}
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #b0b0b0; margin-top: 5px;}
</style>
<div class="header-box">
    <div class="header-title">🧠 Hedge Fund AI: V8 (AutoML & Tournament)</div>
    <div class="header-sub">Auto-Imputation Selection • XGBoost Auto-Tuning • Transparency Report</div>
</div>
""", unsafe_allow_html=True)

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
DATA_PERIOD = "730d"

# --- CONNECT ---
def connect_sheet_services():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try: creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        except: pass
    elif os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    if not creds: return None, None
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(SHEET_ID)
        try: hist = spreadsheet.worksheet("Gecmis")
        except: hist = spreadsheet.add_worksheet("Gecmis", 1000, 6)
        return spreadsheet.sheet1, hist
    except: return None, None

def load_portfolio():
    pf_sheet, _ = connect_sheet_services()
    if pf_sheet is None: return pd.DataFrame(), None
    try:
        data = pf_sheet.get_all_records()
        df = pd.DataFrame(data)
        cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, pf_sheet
    except: return pd.DataFrame(), None

def log_transaction(ticker, action, amount, price, model, sheet):
    if sheet:
        now = datetime.now(pytz.timezone('Turkey')).strftime('%Y-%m-%d %H:%M')
        sheet.append_row([now, ticker, action, float(amount), float(price), model])

def save_portfolio(df, sheet):
    if sheet:
        df_exp = df.copy().astype(str)
        sheet.clear()
        sheet.update([df_exp.columns.values.tolist()] + df_exp.values.tolist())

# --- AUTOML LOGIC ---
def get_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

class DataArchitect:
    def find_best_imputer(self, df, features):
        """
        Verinin %10'unu saklayıp KNN, MICE ve Linear Interpolation yarıştırır.
        En düşük hatayı (RMSE) vereni seçer.
        MICE ve KNN için OHLC verilerini de kullanarak korelasyonu artırır.
        """
        # MICE'ın daha iyi çalışması için features'a OHLCV verilerini de ekliyoruz (Geçici olarak)
        aux_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
        # Hedeflenen sütunlar (Features)
        cols_to_test = features
        
        # Test seti oluştur
        test_df = df[features].copy() # Sadece features üzerinde test yapacağız
        mask = np.random.choice([True, False], size=test_df.shape, p=[0.1, 0.9])
        ground_truth = test_df.values.copy()
        
        # Simülasyon Verisi (Bozuk)
        sim_df = df.copy()
        for col in features:
            sim_df.loc[mask[:, features.index(col)], col] = np.nan
            
        results = {}
        
        # 1. MICE (OHLC destekli)
        try:
            # Tüm sütunları vererek MICE'ın aradaki ilişkiyi çözmesini sağlıyoruz
            mice_imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
            df_mice = sim_df.copy()
            # Sadece sayısal sütunları seç
            numeric_cols = df_mice.select_dtypes(include=[np.number]).columns
            filled_matrix = mice_imp.fit_transform(df_mice[numeric_cols])
            df_mice[numeric_cols] = filled_matrix
            
            # Sadece features kısmındaki hatayı ölç
            mice_filled_vals = df_mice[features].values
            results['MICE'] = np.sqrt(mean_squared_error(ground_truth[mask], mice_filled_vals[mask]))
        except: results['MICE'] = 999.0

        # 2. KNN
        try:
            knn_imp = KNNImputer(n_neighbors=5)
            df_knn = sim_df.copy()
            numeric_cols = df_knn.select_dtypes(include=[np.number]).columns
            filled_matrix = knn_imp.fit_transform(df_knn[numeric_cols])
            df_knn[numeric_cols] = filled_matrix
            
            knn_filled_vals = df_knn[features].values
            results['KNN'] = np.sqrt(mean_squared_error(ground_truth[mask], knn_filled_vals[mask]))
        except: results['KNN'] = 999.0
        
        # 3. Linear Interpolation (Basit)
        try:
            df_lin = sim_df[features].copy() # Interpolation sadece kendi sütununa bakar
            lin_filled_vals = df_lin.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
            results['Linear'] = np.sqrt(mean_squared_error(ground_truth[mask], lin_filled_vals[mask]))
        except: results['Linear'] = 999.0
        
        # Kazananı Belirle
        winner = min(results, key=results.get)
        final_df = df.copy()
        
        # Gerçek Veriyi Kazananla Doldur
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        
        if winner == 'MICE':
            final_df[numeric_cols] = IterativeImputer(estimator=BayesianRidge(), max_iter=10).fit_transform(final_df[numeric_cols])
        elif winner == 'KNN':
            final_df[numeric_cols] = KNNImputer(n_neighbors=5).fit_transform(final_df[numeric_cols])
        else:
            final_df[features] = final_df[features].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            
        return final_df, winner, results

def process_data_automl(df):
    if df is None or len(df)<150: return None, None, None
    df = df.copy()
    
    # Feature Engineering (Önce featureları üretip sonra temizlemek daha mantıklı olabilir ama NaN riski var)
    # Strateji: Önce basit temizlik (ffill), sonra feature üretimi, sonra AutoML temizlik
    df.fillna(method='ffill', inplace=True)
    
    df['kalman'] = df['close'].rolling(3).mean()
    df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
    
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
    df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # AutoML Imputation'a gidecek featurelar
    features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
    
    # AutoML'i çalıştır (Featurelardaki olası NaN'lar için)
    # Burada NaN üretmemeye çalıştık ama log_ret vb. baştaki satırlarda NaN üretir.
    # Architect bunları en iyi yöntemle dolduracak.
    architect = DataArchitect()
    df_clean, winner_imp, imp_scores = architect.find_best_imputer(df, features)
    
    df_clean.dropna(subset=['target'], inplace=True)
    return df_clean, winner_imp, imp_scores

class Brain:
    def __init__(self): self.meta = LogisticRegression(C=1.0)
    
    def optimize_xgboost(self, X_tr, y_tr):
        depths, lrs = [3, 5, 7], [0.01, 0.1, 0.2]
        best_score, best_model, best_desc = -1, None, ""
        v = int(len(X_tr)*0.8)
        X_t, X_v, y_t, y_v = X_tr.iloc[:v], X_tr.iloc[v:], y_tr.iloc[:v], y_tr.iloc[v:]
        
        for d in depths:
            for lr in lrs:
                m = xgb.XGBClassifier(n_estimators=100, max_depth=d, learning_rate=lr, random_state=42)
                m.fit(X_t, y_t)
                s = m.score(X_v, y_v)
                if s > best_score: best_score=s; best_model=m; best_desc=f"d={d},lr={lr}"
        best_model.fit(X_tr, y_tr)
        return best_model, best_desc

    def run_tournament(self, df):
        features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
        test_size = 60
        train, test = df.iloc[:-test_size], df.iloc[-test_size:]
        X_tr, y_tr = train[features], train['target']
        X_test = test[features]
        
        rf = RandomForestClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=100, max_depth=5).fit(X_tr, y_tr)
        opt_xgb, xgb_params = self.optimize_xgboost(X_tr, y_tr)
        
        try:
            scaler = StandardScaler()
            X_hmm = scaler.fit_transform(train[['log_ret', 'range_vol_delta']].fillna(0))
            hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50).fit(X_hmm)
            hmm_probs = hmm.predict_proba(X_hmm)[:,0]
            hmm_probs_t = hmm.predict_proba(scaler.transform(test[['log_ret', 'range_vol_delta']].fillna(0)))[:,0]
        except: hmm_probs=np.zeros(len(train)); hmm_probs_t=np.zeros(len(test))
        
        meta_X = pd.DataFrame({'RF': rf.predict_proba(X_tr)[:,1], 'ETC': etc.predict_proba(X_tr)[:,1], 
                               'XGB': opt_xgb.predict_proba(X_tr)[:,1], 'HMM': hmm_probs}, index=train.index).fillna(0)
        self.meta.fit(meta_X, y_tr)
        
        meta_X_test = pd.DataFrame({'RF': rf.predict_proba(X_test)[:,1], 'ETC': etc.predict_proba(X_test)[:,1], 
                                    'XGB': opt_xgb.predict_proba(X_test)[:,1], 'HMM': hmm_probs_t}, index=test.index).fillna(0)
        
        p_ens = self.meta.predict_proba(meta_X_test)[:,1]
        p_solo = opt_xgb.predict_proba(X_test)[:,1]
        
        sim_ens, sim_solo = 100.0, 100.0
        rets = test['close'].pct_change().fillna(0).values
        eq_ens, eq_solo = [100.0], [100.0]
        for i in range(len(test)):
            if p_ens[i]>0.55: sim_ens*=(1+rets[i])
            if p_solo[i]>0.55: sim_solo*=(1+rets[i])
            eq_ens.append(sim_ens); eq_solo.append(sim_solo)
            
        winner = "Solo Optimized XGB" if sim_solo > sim_ens else "Ensemble"
        return {'prob': p_solo[-1] if winner=="Solo Optimized XGB" else p_ens[-1], 
                'winner': winner, 'roi': (sim_solo if winner=="Solo Optimized XGB" else sim_ens)-100, 
                'xgb_params': xgb_params, 'eq_ens': eq_ens, 'eq_solo': eq_solo, 'dates': test.index}

# --- MAIN UI ---
pf_df, sheet_pf = load_portfolio()
_, sheet_hist = connect_sheet_services()
tab1, tab2, tab3, tab4 = st.tabs(["📊 Bot Kontrol & AutoML", "📑 Canlı Veri", "📜 Geçmiş Kayıtları", "🔬 MICE Lab"])

if not pf_df.empty:
    with tab1:
        st.metric("Toplam Varlık", f"${pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum():.2f}")
        if st.button("🚀 AUTOML BAŞLAT (Impute & Tune)", type="primary"):
            updated_pf = pf_df.copy()
            pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
            updated_pf['Nakit_Bakiye_USD'] = 0.0
            brain = Brain()
            buy_orders = []
            
            prog = st.progress(0)
            for i, (idx, row) in enumerate(updated_pf.iterrows()):
                ticker = row['Ticker']
                df = get_data(ticker)
                if df is not None:
                    # AutoML İşlemi
                    df_clean, imp_winner, imp_scores = process_data_automl(df)
                    
                    if df_clean is not None:
                        res = brain.run_tournament(df_clean)
                        decision = "HOLD"
                        if res['prob'] > 0.55: decision = "BUY"
                        elif res['prob'] < 0.45: decision = "SELL"
                        
                        with st.expander(f"{ticker} | {decision} | {res['winner']} (ROI: %{res['roi']:.1f})"):
                            c1, c2, c3 = st.columns(3)
                            c1.markdown(f"**Temizlik:** `{imp_winner}`")
                            c1.caption(f"Hata (RMSE): {imp_scores[imp_winner]:.5f}")
                            c2.markdown(f"**XGB Tuning:** `{res['xgb_params']}`")
                            c3.markdown("**Ensemble:** RF+ETC+HMM+XGB")
                            st.line_chart(pd.DataFrame({'Ensemble': res['eq_ens'], 'Solo': res['eq_solo']}))
                        
                        current_p = df['close'].iloc[-1]
                        model_desc = f"{res['winner']}|{imp_winner}|{res['xgb_params']}"
                        if row['Durum']=='COIN' and decision=="SELL":
                            pool_cash += float(row['Miktar']) * current_p
                            updated_pf.at[idx,'Durum']='CASH'; updated_pf.at[idx,'Miktar']=0.0
                            log_transaction(ticker, "SAT", row['Miktar'], current_p, model_desc, sheet_hist)
                        elif row['Durum']=='CASH' and decision=="BUY":
                            buy_orders.append({'idx':idx, 'ticker':ticker, 'p':current_p, 'w':res['prob'], 'm':model_desc})
                prog.progress((i+1)/len(updated_pf))
                
            if buy_orders and pool_cash > 5:
                total_w = sum([b['w'] for b in buy_orders])
                for b in buy_orders:
                    share = (b['w']/total_w)*pool_cash; amt = share/b['p']
                    updated_pf.at[b['idx'],'Durum']='COIN'; updated_pf.at[b['idx'],'Miktar']=amt
                    updated_pf.at[b['idx'],'Nakit_Bakiye_USD']=0.0
                    log_transaction(b['ticker'], "AL", amt, b['p'], b['m'], sheet_hist)
            elif pool_cash > 0: updated_pf.at[updated_pf.index[0], 'Nakit_Bakiye_USD'] += pool_cash
            save_portfolio(updated_pf, sheet_pf)
            st.success("AutoML Analizi Tamamlandı.")

    with tab2: st.dataframe(pf_df)
    with tab3: 
        if sheet_hist: st.dataframe(pd.DataFrame(sheet_hist.get_all_records()).iloc[::-1])
    with tab4:
        st.header("🔬 MICE Imputation Denetçisi")
        test_ticker = st.selectbox("Test Coin:", ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD"])
        if st.button("🧪 Veriyi Test Et"):
            df = get_data(test_ticker)
            if df is not None:
                # process_data_automl içindeki mantığı burada kullanmak yerine
                # sadece featureları hazırlayıp Architect'i çağırıyoruz
                df['kalman'] = df['close'].rolling(3).mean()
                df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
                df['ret'] = df['close'].pct_change()
                df['range'] = (df['high']-df['low'])/df['close']
                df['range_vol_delta'] = df['range'].pct_change(5)
                df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
                df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
                df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
                avg_feats = df[['avg_ret_5m','avg_ret_3y']].fillna(0)
                df['historical_avg_score'] = StandardScaler().fit_transform(avg_feats).mean(axis=1)
                
                features = ['log_ret', 'range', 'heuristic', 'historical_avg_score', 'range_vol_delta']
                
                architect = DataArchitect()
                _, winner, scores = architect.find_best_imputer(df, features)
                
                st.success(f"**Kazanan Yöntem:** {winner}")
                st.write("RMSE Hata Skorları (Düşük olan iyi):")
                st.json(scores)
