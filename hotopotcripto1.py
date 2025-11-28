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
from sklearn.metrics import accuracy_score
from hmmlearn.hmm import GaussianHMM
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# UI CONFIG
st.set_page_config(page_title="Hedge Fund AI: V9 Baybebek", layout="wide", page_icon="👶")
st.markdown("""
<style>
    .main {background-color: #0E1117;}
    .header-box {background: linear-gradient(135deg, #1A2980 0%, #26D0CE 100%); padding: 25px; border-radius: 12px; border-left: 5px solid #FFD700; margin-bottom: 25px;}
    .header-title {font-size: 32px; font-weight: 700; color: #fff; margin:0;}
    .header-sub {font-size: 14px; color: #e0e0e0; margin-top: 5px;}
    .metric-card {background-color: #1e2126; padding: 10px; border-radius: 8px; border: 1px solid #333;}
</style>
<div class="header-box">
    <div class="header-title">👶 Hedge Fund AI: V9 (Grand League)</div>
    <div class="header-sub">Baybebek Imputation • Cartesian Product Optimization • Train/Val/Test Split</div>
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

# --- DATA & FEATURES ---
def get_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def prepare_raw_features(df):
    """
    Ham featureları üretir ama NaN'ları doldurmaz. 
    Imputation daha sonra Train/Val split üzerinde yapılacak.
    """
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Teknik İndikatörler
    df['kalman'] = df['close'].rolling(3).mean()
    df['log_ret'] = np.log(df['kalman']/df['kalman'].shift(1))
    df['ret'] = df['close'].pct_change()
    df['range'] = (df['high']-df['low'])/df['close']
    df['range_vol_delta'] = df['range'].pct_change(5)
    df['heuristic'] = (np.sign(df['close'].pct_change(5)) + np.sign(df['close'].pct_change(30)))/2.0
    
    # Historical
    df['avg_ret_5m'] = df['ret'].rolling(100).mean()*100
    df['avg_ret_3y'] = df['ret'].rolling(750).mean()*100
    
    # Target (Shifted Close)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Son satırı (gelecek) tut, ama eğitimden çıkar
    future_row = df.iloc[[-1]].copy()
    df_historic = df.iloc[:-1].copy()
    
    return df_historic, future_row

# --- IMPUTATION METHODS ---
class ImputationLab:
    def baybebek_impute(self, df):
        """
        Baybebek Tekniği: n. nokta boşsa, (n-2, n-1, n+1, n+2) ortalamasıyla doldur.
        Rolling window=5, center=True kullanır.
        """
        filled = df.copy()
        numeric_cols = filled.select_dtypes(include=[np.number]).columns
        # min_periods=1: Komşulardan en az 1 tanesi doluysa hesapla
        rolling_means = filled[numeric_cols].rolling(window=5, center=True, min_periods=1).mean()
        filled[numeric_cols] = filled[numeric_cols].fillna(rolling_means)
        # Hala boş kalan varsa (en uçlar) linear yap
        filled = filled.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
        return filled

    def apply_imputation(self, df_train, df_val, method):
        """Train verisine fit et, Val verisini transform et (Data Leakage Önleme)"""
        # Sütunlar
        features = ['log_ret', 'range', 'heuristic', 'range_vol_delta', 'avg_ret_5m', 'avg_ret_3y']
        
        # Sadece feature sütunlarını al, diğerlerini koru
        X_train = df_train[features].copy()
        X_val = df_val[features].copy()
        
        # Method Seçimi
        if method == 'Baybebek':
            X_train_filled = self.baybebek_impute(X_train)
            X_val_filled = self.baybebek_impute(X_val)
            
        elif method == 'MICE':
            try:
                imp = IterativeImputer(estimator=BayesianRidge(), max_iter=5, random_state=42)
                X_train_filled = pd.DataFrame(imp.fit_transform(X_train), columns=features, index=X_train.index)
                X_val_filled = pd.DataFrame(imp.transform(X_val), columns=features, index=X_val.index)
            except: # Hata durumunda Baybebek'e düş
                X_train_filled = self.baybebek_impute(X_train)
                X_val_filled = self.baybebek_impute(X_val)

        elif method == 'KNN':
            try:
                imp = KNNImputer(n_neighbors=5)
                X_train_filled = pd.DataFrame(imp.fit_transform(X_train), columns=features, index=X_train.index)
                X_val_filled = pd.DataFrame(imp.transform(X_val), columns=features, index=X_val.index)
            except:
                X_train_filled = self.baybebek_impute(X_train)
                X_val_filled = self.baybebek_impute(X_val)
                
        else: # Linear (Fallback)
            X_train_filled = X_train.interpolate(method='linear').fillna(0)
            X_val_filled = X_val.interpolate(method='linear').fillna(0)
            
        # Doldurulmuş dataları geri birleştir
        df_train_out = df_train.copy(); df_train_out[features] = X_train_filled
        df_val_out = df_val.copy(); df_val_out[features] = X_val_filled
        
        return df_train_out, df_val_out

# --- MODEL TRAINING & LEAGUE ---
class GrandLeagueBrain:
    def __init__(self):
        self.meta_model = LogisticRegression(C=1.0)
        self.lab = ImputationLab()
        
    def train_xgboost(self, X_tr, y_tr, X_val, y_val):
        # Hızlı Grid Search
        best_score = -1; best_model = None
        for d in [3, 5]:
            for lr in [0.05, 0.1]:
                m = xgb.XGBClassifier(n_estimators=100, max_depth=d, learning_rate=lr, random_state=42, n_jobs=1)
                m.fit(X_tr, y_tr)
                preds = m.predict(X_val)
                acc = accuracy_score(y_val, preds)
                if acc > best_score:
                    best_score = acc
                    best_model = m
        return best_model, best_score

    def train_ensemble(self, X_tr, y_tr, X_val, y_val):
        # Base Models
        rf = RandomForestClassifier(n_estimators=50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
        etc = ExtraTreesClassifier(n_estimators=50, max_depth=5, n_jobs=1).fit(X_tr, y_tr)
        xgb_m = xgb.XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, n_jobs=1).fit(X_tr, y_tr)
        
        # Meta Features
        meta_tr = pd.DataFrame({'RF': rf.predict_proba(X_tr)[:,1], 'ETC': etc.predict_proba(X_tr)[:,1], 'XGB': xgb_m.predict_proba(X_tr)[:,1]})
        meta_val = pd.DataFrame({'RF': rf.predict_proba(X_val)[:,1], 'ETC': etc.predict_proba(X_val)[:,1], 'XGB': xgb_m.predict_proba(X_val)[:,1]})
        
        lr = LogisticRegression().fit(meta_tr, y_tr)
        preds = lr.predict(meta_val)
        acc = accuracy_score(y_val, preds)
        
        return (rf, etc, xgb_m, lr), acc

    def run_grand_league(self, df):
        """
        Kombinasyon Ligi:
        (MICE, KNN, Linear, Baybebek) x (XGBoost, Ensemble)
        Train/Val split üzerinde en iyiyi seçer.
        """
        # 1. SPLIT (Zaman Serisi Olduğu için Shuffle=False)
        train_size = int(len(df) * 0.70)
        val_size = int(len(df) * 0.15)
        
        df_train = df.iloc[:train_size].copy()
        df_val = df.iloc[train_size : train_size+val_size].copy()
        df_test = df.iloc[train_size+val_size:].copy() # Test seti final sınavı
        
        features = ['log_ret', 'range', 'heuristic', 'range_vol_delta', 'avg_ret_5m', 'avg_ret_3y']
        impute_methods = ['Baybebek', 'MICE', 'KNN', 'Linear']
        
        league_table = []
        
        # 2. TOURNAMENT LOOP
        for imp_name in impute_methods:
            # Impute (Train fit, Val transform)
            d_tr, d_val = self.lab.apply_imputation(df_train, df_val, imp_name)
            
            # Veri Hazırlığı
            X_tr, y_tr = d_tr[features], d_tr['target']
            X_v, y_v = d_val[features], d_val['target']
            
            # A. XGBoost
            model_xgb, acc_xgb = self.train_xgboost(X_tr, y_tr, X_v, y_v)
            league_table.append({
                'combo': f"{imp_name} + XGBoost",
                'imputer': imp_name, 'model_type': 'XGB',
                'model': model_xgb, 'acc': acc_xgb
            })
            
            # B. Ensemble
            model_ens, acc_ens = self.train_ensemble(X_tr, y_tr, X_v, y_v)
            league_table.append({
                'combo': f"{imp_name} + Ensemble",
                'imputer': imp_name, 'model_type': 'ENS',
                'model': model_ens, 'acc': acc_ens
            })
            
        # 3. WINNER SELECTION
        # Accuracy'ye göre sırala
        league_table.sort(key=lambda x: x['acc'], reverse=True)
        winner = league_table[0]
        
        # 4. FINAL PREDICTION (TEST SET & FUTURE)
        # Kazanan strateji ile Test ve Future verisini hazırla
        # Not: Final prediction için tüm veriyi (Train+Val) impute etmek daha doğru olur ama
        # tutarlılık için kazanan imputer mantığını test setine uygulayacağız.
        
        # Test seti imputasyonu (Sadece transform gibi düşünelim ama Baybebek stateless olduğu için direkt uygulanır)
        df_combined = pd.concat([df_train, df_val])
        d_combined_filled, d_test_filled = self.lab.apply_imputation(df_combined, df_test, winner['imputer'])
        
        # Kazanan modeli (XGB veya Ens) kullanarak tahmin üret
        X_test = d_test_filled[features]
        
        if winner['model_type'] == 'XGB':
            probs = winner['model'].predict_proba(X_test)[:, 1]
        else: # ENS
            rf, etc, xg, lr = winner['model']
            meta_test = pd.DataFrame({
                'RF': rf.predict_proba(X_test)[:,1], 
                'ETC': etc.predict_proba(X_test)[:,1], 
                'XGB': xg.predict_proba(X_test)[:,1]
            })
            probs = lr.predict_proba(meta_test)[:, 1]
            
        # Simülasyon (ROI Hesabı)
        sim = 100.0
        eq_curve = [100.0]
        rets = d_test_filled['close'].pct_change().fillna(0).values
        for i in range(len(probs)):
            if probs[i] > 0.55: sim *= (1+rets[i])
            eq_curve.append(sim)
            
        final_prob = probs[-1]
        
        return {
            'winner_combo': winner['combo'],
            'acc': winner['acc'],
            'prob': final_prob,
            'roi': sim - 100,
            'equity': eq_curve,
            'dates': df_test.index,
            'league_table': league_table
        }

# --- MAIN UI ---
pf_df, sheet_pf = load_portfolio()
_, sheet_hist = connect_sheet_services()

tab1, tab2, tab3 = st.tabs(["🏆 Grand League (Bot Kontrol)", "📑 Canlı Veri", "📜 Geçmiş Kayıtları"])

if not pf_df.empty:
    with tab1:
        st.metric("Toplam Varlık", f"${pf_df['Nakit_Bakiye_USD'].sum() + pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum():.2f}")
        
        if st.button("⚽ LİGİ BAŞLAT (Train/Val/Test + Impute)", type="primary"):
            updated_pf = pf_df.copy()
            pool_cash = updated_pf['Nakit_Bakiye_USD'].sum()
            updated_pf['Nakit_Bakiye_USD'] = 0.0
            
            brain = GrandLeagueBrain()
            buy_orders = []
            
            prog = st.progress(0)
            for i, (idx, row) in enumerate(updated_pf.iterrows()):
                ticker = row['Ticker']
                raw_data = get_data(ticker)
                
                if raw_data is not None and len(raw_data) > 200:
                    # 1. Ham Veriyi Hazırla (Boşluklarla)
                    df_hist, df_future = prepare_raw_features(raw_data)
                    
                    # 2. Ligi Başlat
                    res = brain.run_grand_league(df_hist)
                    
                    decision = "HOLD"
                    if res['prob'] > 0.55: decision = "BUY"
                    elif res['prob'] < 0.45: decision = "SELL"
                    
                    # --- UI GÖSTERİMİ ---
                    with st.expander(f"{ticker} | {decision} | {res['winner_combo']} (Val Acc: %{res['acc']*100:.1f})"):
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.markdown("### 🏅 Kazanan")
                            st.info(f"**{res['winner_combo']}**")
                            st.metric("Validation Başarısı", f"%{res['acc']*100:.1f}")
                            st.metric("Tahmini ROI (Test)", f"%{res['roi']:.1f}")
                            
                            st.markdown("#### 📊 Lig Sıralaması (Top 3)")
                            for rank, team in enumerate(res['league_table'][:3]):
                                st.caption(f"{rank+1}. {team['combo']} (Acc: %{team['acc']*100:.1f})")
                                
                        with c2:
                            st.markdown("### 📈 Test Performansı")
                            fig = go.Figure()
                            # Daha okunabilir grafik
                            fig.add_trace(go.Scatter(
                                x=res['dates'], y=res['equity'], 
                                mode='lines', name='Model Bakiye',
                                line=dict(color='#00ff88', width=3)
                            ))
                            fig.update_layout(
                                template="plotly_dark", 
                                margin=dict(l=0,r=0,t=0,b=0),
                                height=250,
                                yaxis_title="Bakiye ($)",
                                showlegend=True
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    current_p = raw_data['close'].iloc[-1]
                    model_desc = f"{res['winner_combo']} (Acc:{res['acc']:.2f})"
                    
                    # İŞLEM MANTIĞI
                    if row['Durum']=='COIN' and decision=="SELL":
                        pool_cash += float(row['Miktar']) * current_p
                        updated_pf.at[idx,'Durum']='CASH'; updated_pf.at[idx,'Miktar']=0.0
                        log_transaction(ticker, "SAT", row['Miktar'], current_p, model_desc, sheet_hist)
                        
                    elif row['Durum']=='CASH' and decision=="BUY":
                        buy_orders.append({'idx':idx, 'ticker':ticker, 'p':current_p, 'w':res['prob'], 'm':model_desc})
                        
                prog.progress((i+1)/len(updated_pf))
                
            # ALIMLAR
            if buy_orders and pool_cash > 5:
                total_w = sum([b['w'] for b in buy_orders])
                for b in buy_orders:
                    share = (b['w']/total_w)*pool_cash; amt = share/b['p']
                    updated_pf.at[b['idx'],'Durum']='COIN'; updated_pf.at[b['idx'],'Miktar']=amt
                    updated_pf.at[b['idx'],'Nakit_Bakiye_USD']=0.0
                    log_transaction(b['ticker'], "AL", amt, b['p'], b['m'], sheet_hist)
            elif pool_cash > 0: updated_pf.at[updated_pf.index[0], 'Nakit_Bakiye_USD'] += pool_cash
            
            save_portfolio(updated_pf, sheet_pf)
            st.success("Büyük Lig Tamamlandı! Baybebek ve Rakipleri Analiz Edildi.")

    with tab2: st.dataframe(pf_df)
    with tab3: 
        if sheet_hist: st.dataframe(pd.DataFrame(sheet_hist.get_all_records()).iloc[::-1])
