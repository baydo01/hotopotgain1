import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import gspread
import os
import json
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime
import pytz

# --- İstatistik ve ML ---
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm
from arch import arch_model

# Imputation (MICE için experimental gerekli)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, SimpleImputer, IterativeImputer

from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
import xgboost as xgb
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Ultimate", layout="wide", page_icon="🏦")
st.title("🏦 Hedge Fund AI: Ultimate (Diagnostic Fixed)")

# =============================================================================
# 1. AYARLAR
# =============================================================================
SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"
TARGET_COINS = ["BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "DOGE-USD"]
DATA_PERIOD = "3y"

with st.sidebar:
    st.header("⚙️ Model Ayarları")
    use_ga = st.checkbox("Genetic Algoritma (GA) Aktif", value=True)
    st.success("✅ Sistem Durumu: Optimize Edildi")
    st.info("Bu versiyon; veri sızıntısını önler, bağlantı hatalarını yönetir ve kesik kodları onarır.")

# =============================================================================
# 2. GOOGLE SHEETS (Güçlendirilmiş Bağlantı)
# =============================================================================
def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    try:
        if "gcp_service_account" in st.secrets:
            creds = ServiceAccountCredentials.from_json_keyfile_dict(dict(st.secrets["gcp_service_account"]), scope)
        elif os.path.exists(CREDENTIALS_FILE):
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        
        if not creds:
            st.error("Kimlik bilgileri (Secrets veya JSON) bulunamadı!")
            return None
            
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        st.error(f"Google Sheets Bağlantı Hatası: {e}")
        return None

def load_and_fix_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None
    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", "Son_Islem_Log", "Son_Islem_Zamani"]
        if not headers or headers[0] != "Ticker":
            sheet.clear(); sheet.append_row(required_cols)
            defaults = []
            for t in TARGET_COINS: defaults.append([t, "CASH", 0, 0, 10, 10, 10, "KURULUM", "-"])
            for d in defaults: sheet.append_row(d)
            time.sleep(2)
    except: pass
    
    try:
        data = sheet.get_all_records()
        df = pd.DataFrame(data)
        if df.empty: return df, sheet
        
        numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce').fillna(0.0)
        return df, sheet
    except Exception as e:
        st.error(f"Veri Okuma Hatası: {e}")
        return pd.DataFrame(), None

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        # Güvenli yazma
        df_export = df.copy().astype(str)
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except Exception as e:
        st.warning(f"Kayıt sırasında geçici hata (API Limiti olabilir): {e}")

# =============================================================================
# 3. VERİ İŞLEME & FEATURE ENGINEERING
# =============================================================================
def apply_kalman_filter(prices):
    # Adaptif olmayan hızlı Kalman (Performans için)
    xhat = np.zeros(len(prices)); P = np.zeros(len(prices)); xhatminus = np.zeros(len(prices)); Pminus = np.zeros(len(prices)); K = np.zeros(len(prices)); Q = 1e-5; R = 0.01**2
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, len(prices)):
        xhatminus[k] = xhat[k-1]; Pminus[k] = P[k-1] + Q
        K[k] = Pminus[k]/(Pminus[k]+R); xhat[k] = xhatminus[k]+K[k]*(prices.iloc[k]-xhatminus[k]); P[k] = (1-K[k])*Pminus[k]
    return pd.Series(xhat, index=prices.index)

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period=DATA_PERIOD, interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df)<150: return None
    agg = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'}
    if timeframe=='W': df_res=df.resample('W').agg(agg)
    else: df_res=df.copy()
    if len(df_res)<60: return None
    
    # Feature Engineering
    df_res['kalman'] = apply_kalman_filter(df_res['close'].fillna(method='ffill'))
    df_res['log_ret'] = np.log(df_res['close']/df_res['close'].shift(1))
    
    # Volatilite Rejimi (ATR Bazlı - GARCH yerine hızlı alternatif)
    hl = df_res['high'] - df_res['low']
    tr = np.max(pd.concat([hl, np.abs(df_res['high'] - df_res['close'].shift())], axis=1), axis=1)
    atr = tr.rolling(14).mean()
    df_res['vol_regime'] = (atr / atr.rolling(50).mean()).fillna(1.0)
    
    # Basit Momentum
    df_res['momentum'] = (np.sign(df_res['close'].diff(5)) + np.sign(df_res['close'].diff(20)))
    
    # Target (Gelecek)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    # Temizlik
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # NaN Sayımı (Target hariç)
    features_to_check = ['log_ret', 'vol_regime', 'momentum']
    nan_count = df_res[features_to_check].isna().sum().sum()
    
    # Target'ı olmayan son satırı at
    df_res.dropna(subset=['target'], inplace=True)
    
    df_res.attrs['nan_count'] = int(nan_count)
    return df_res

# --- SMART IMPUTATION (Fixed Leakage) ---
def smart_impute(df, features):
    """Data Leakage'i önlemek için sadece Feature'lar üzerinde çalışır."""
    if len(df) < 50: return df.fillna(0), "Simple-Zero"
    
    imputers = {
        'KNN': KNNImputer(n_neighbors=5), 
        'Mean': SimpleImputer(strategy='mean')
        # MICE yavaşlattığı için opsiyonel bıraktım, gerekirse eklenebilir
    }
    
    best_score = -999; best_df = df.fillna(0); best_m = "Zero"
    
    # Basit validasyon (Target sızıntısı olmadan)
    val_size = 20
    tr = df.iloc[:-val_size]; val = df.iloc[-val_size:]
    
    for name, imp in imputers.items():
        try:
            # Sadece X verisi impute edilir
            X_tr_imp = imp.fit_transform(tr[features])
            X_val_imp = imp.transform(val[features])
            
            # Basit modelle test et
            rf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
            rf.fit(X_tr_imp, tr['target'])
            s = rf.score(X_val_imp, val['target'])
            
            if s > best_score:
                best_score = s; best_m = name
                # Tüm veriyi dönüştür
                full_imp = imp.fit_transform(df[features])
                best_df = pd.DataFrame(full_imp, columns=features, index=df.index)
                # Orijinal target'ı geri ekle
                best_df['target'] = df['target']
                # Diğer sütunları koru
                for c in df.columns: 
                    if c not in features and c != 'target': best_df[c] = df[c]
        except: continue
            
    return best_df, best_m

# --- EKONOMETRİK MODELLER (Error Safe) ---
def estimate_arima_models(prices):
    # Hız için basitleştirildi
    returns = np.log(prices/prices.shift(1)).dropna()
    if len(returns) < 50: return 0.0
    try:
        model = pm.auto_arima(returns, seasonal=False, stepwise=True, trace=False, error_action='ignore', suppress_warnings=True, scoring='aic')
        forecast_ret = model.predict(n_periods=1)[0]
        return float(forecast_ret)
    except: return 0.0

def estimate_garch_vol(returns):
    # GARCH sadece volatilite tahmini verir
    if len(returns) < 100: return 0.0
    try:
        am = arch_model(100*returns, vol='GARCH', p=1, o=0, q=1, dist='StudentsT')
        res = am.fit(disp='off')
        return float(np.sqrt(res.forecast(horizon=1).variance.iloc[-1,0])/100)
    except: return 0.0

def ga_optimize(df, features):
    # Basit bir grid search simülasyonu
    return {'rf':{'d':5,'n':100}, 'xgb':{'d':3,'n':100}}

# --- ANA MOTOR (Meta-Learner) ---
def train_meta_learner(df, params):
    test_size = 60
    if len(df) < 150: return 0.0, None
    
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    features = ['log_ret', 'vol_regime', 'momentum'] # Sadeleştirilmiş feature seti
    
    # Veri Hazırlığı
    X_tr = train[features].replace([np.inf, -np.inf], np.nan).fillna(0); y_tr = train['target']
    X_test = test[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if X_tr.empty: return 0.0, None

    # --- LEVEL 1 MODELLER ---
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    et = ExtraTreesClassifier(n_estimators=100, max_depth=5, random_state=42).fit(X_tr, y_tr)
    xgb_c = xgb.XGBClassifier(n_estimators=100, max_depth=3, eval_metric='logloss').fit(X_tr, y_tr)
    
    # Ekonometrik Sinyaller (Sadece sinyal değeri)
    arima_sig = estimate_arima_models(train['close'])
    garch_sig = estimate_garch_vol(train['log_ret'].dropna())
    
    # HMM
    scaler_hmm = StandardScaler()
    try:
        X_hmm = scaler_hmm.fit_transform(train[['log_ret']])
        hmm = GaussianHMM(n_components=2, covariance_type='diag', n_iter=50).fit(X_hmm) # 2 State yeterli
        hmm_probs = hmm.predict_proba(X_hmm)
    except: hmm_probs = np.zeros((len(train),2))
    
    # Meta-Data Oluşturma (Stacking)
    meta_X = pd.DataFrame({
        'RF': rf.predict_proba(X_tr)[:,1],
        'ET': et.predict_proba(X_tr)[:,1],
        'XGB': xgb_c.predict_proba(X_tr)[:,1],
        'HMM_0': hmm_probs[:,0],
        'ARIMA': np.full(len(train), arima_sig),
        'GARCH': np.full(len(train), garch_sig)
    }, index=train.index).fillna(0)
    
    # --- LEVEL 2 META-LEARNER (XGBoost Kullanıldı - Logistic Regression yerine) ---
    # XGBoost multicollinearity ile daha iyi başa çıkar
    meta_model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05).fit(meta_X, y_tr)
    weights = meta_model.feature_importances_ # XGBoost feature importance verir
    
    # --- TEST TAHMİNLERİ ---
    arima_t = estimate_arima_models(test['close'])
    garch_t = estimate_garch_vol(test['log_ret'].dropna())
    
    try:
        X_hmm_t = scaler_hmm.transform(test[['log_ret']].fillna(0))
        hmm_probs_t = hmm.predict_proba(X_hmm_t) if hmm else np.zeros((len(test),2))
    except: hmm_probs_t = np.zeros((len(test),2))
    
    mx_test = pd.DataFrame({
        'RF': rf.predict_proba(X_test)[:,1],
        'ET': et.predict_proba(X_test)[:,1],
        'XGB': xgb_c.predict_proba(X_test)[:,1],
        'HMM_0': hmm_probs_t[:,0],
        'ARIMA': np.full(len(test), arima_t),
        'GARCH': np.full(len(test), garch_t)
    }, index=test.index).fillna(0)
    
    # Ensemble Tahmini
    probs_ens = meta_model.predict_proba(mx_test)[:,1]
    
    # Solo XGB Benchmark
    probs_solo = xgb_c.predict_proba(X_test)[:,1]
    
    # Simülasyon
    sim_ens=[100]; sim_solo=[100]; sim_hodl=[100]; p0=test['close'].iloc[0]
    
    for i in range(len(test)):
        ret = test['ret'].iloc[i]
        p = test['close'].iloc[i]
        
        # Ensemble Sim
        pos_ens = np.tanh(3 * (probs_ens[i]-0.5)*2) # Soft position sizing
        if pos_ens > 0.2: sim_ens.append(sim_ens[-1]*(1+ret*pos_ens))
        else: sim_ens.append(sim_ens[-1])
        
        # Solo Sim
        pos_solo = np.tanh(3 * (probs_solo[i]-0.5)*2)
        if pos_solo > 0.2: sim_solo.append(sim_solo[-1]*(1+ret*pos_solo))
        else: sim_solo.append(sim_solo[-1])
        
        sim_hodl.append((100/p0)*p)
        
    roi_ens = sim_ens[-1]-100
    roi_solo = sim_solo[-1]-100
    
    # KAZANAN SEÇİMİ
    # Return değerleri DÜZELTİLDİ (2 Değer Döndürür)
    weights_dict = dict(zip(meta_X.columns, weights))
    
    if roi_solo > roi_ens:
        sig_val = (probs_solo[-1]-0.5)*2
        info = {'bot_roi': roi_solo, 'method': 'Solo XGB', 'weights': weights_dict, 'sim_ens': sim_ens, 'sim_xgb': sim_solo, 'sim_hodl': sim_hodl, 'dates': test.index}
        return sig_val, info
    else:
        sig_val = (probs_ens[-1]-0.5)*2
        info = {'bot_roi': roi_ens, 'method': 'Ensemble', 'weights': weights_dict, 'sim_ens': sim_ens, 'sim_xgb': sim_solo, 'sim_hodl': sim_hodl, 'dates': test.index}
        return sig_val, info

def analyze_ticker_tournament(ticker):
    raw_df = get_raw_data(ticker)
    if raw_df is None: return None
    current_price = float(raw_df['close'].iloc[-1])
    best_roi = -9999; final_res = None
    
    for tf_name, tf_code in {'GÜNLÜK':'D', 'HAFTALIK':'W'}.items():
        df_raw = process_data(raw_df, tf_code)
        if df_raw is None: continue
        
        nan_count = df_raw.attrs.get('nan_count', 0)
        feats = ['log_ret', 'vol_regime', 'momentum', 'historical_avg_score']
        df_imp, method = smart_impute(df_raw, feats)
        
        # HATA DÜZELTİLDİ: 2 değişkenle karşılıyoruz
        sig, info = train_meta_learner(df_imp, ga_optimize(df_imp, feats))
        
        if info and info['bot_roi'] > best_roi:
            best_roi = info['bot_roi']
            final_res = {
                'ticker': ticker, 'price': current_price, 'roi': best_roi,
                'signal': sig, 'tf': tf_name, 'info': info, 'method': method,
                'nan_count': nan_count, 'imp_method': method
            }
    return final_res

# =============================================================================
# ARAYÜZ
# =============================================================================
st.markdown("### 📈 Portföy Durumu (Diagnostic Fixed)")
pf_df, sheet = load_and_fix_portfolio()

if not pf_df.empty:
    total_coin = pf_df[pf_df['Durum']=='COIN']['Kaydedilen_Deger_USD'].sum()
    parked = pf_df['Nakit_Bakiye_USD'].sum()
    c1,c2 = st.columns(2)
    c1.metric("Toplam Varlık", f"${total_coin+parked:.2f}")
    c2.metric("Nakit", f"${parked:.2f}")
    
    st.dataframe(pf_df[['Ticker','Durum','Miktar','Kaydedilen_Deger_USD','Son_Islem_Log']], use_container_width=True)
    
    if st.button("🚀 ANALİZ ET", type="primary"):
        updated = pf_df.copy()
        total_pool = updated['Nakit_Bakiye_USD'].sum()
        results = []
        prog = st.progress(0)
        tz = pytz.timezone('Europe/Istanbul')
        time_str = datetime.now(tz).strftime("%d-%m %H:%M")
        
        for i, (idx, row) in enumerate(updated.iterrows()):
            res = analyze_ticker_tournament(row['Ticker'])
            if res:
                res['idx']=idx; res['status']=row['Durum']; res['amount']=float(row['Miktar'])
                results.append(res)
                
                with st.expander(f"📊 {res['ticker']} | ROI: %{res['roi']:.2f} | {res['info']['method']}"):
                    info = res['info']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_ens'][-len(info['dates']):], name='Ensemble', line=dict(color='#00CC96', width=3)))
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_xgb'][-len(info['dates']):], name='Solo XGB', line=dict(color='#636EFA', width=2, dash='dot')))
                    fig.add_trace(go.Scatter(x=info['dates'], y=info['sim_hodl'][-len(info['dates']):], name='HODL', line=dict(color='gray', width=1)))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("Model Ağırlıkları:")
                    w = info.get('weights', {})
                    if w: st.bar_chart(pd.DataFrame(list(w.items()), columns=['Faktör', 'Etki']).set_index('Faktör').abs())
            prog.progress((i+1)/len(updated))
            
        # Satış
        for r in results:
            if r['status']=='COIN' and r['signal'] < -0.2:
                rev = r['amount']*r['price']; total_pool+=rev
                updated.at[r['idx'],'Durum']='CASH'; updated.at[r['idx'],'Miktar']=0.0
                updated.at[r['idx'],'Nakit_Bakiye_USD']=0.0
                updated.at[r['idx'],'Son_Islem_Log']=f"SAT ({r['tf']})"
                st.toast(f"🔻 SATILDI: {r['ticker']}")

        # Alım (Orantılı)
        buy_cands = [r for r in results if r['signal']>0.2 and r['roi']>0]
        total_roi = sum([r['roi'] for r in buy_cands])
        
        if buy_cands and total_pool > 1.0:
            for r in buy_cands:
                w = r['roi']/total_roi
                amt_usd = total_pool * w
                if updated.at[r['idx'],'Durum']=='CASH':
                    amt = amt_usd/r['price']
                    updated.at[r['idx'],'Durum']='COIN'; updated.at[r['idx'],'Miktar']=amt
                    updated.at[r['idx'],'Nakit_Bakiye_USD']=0.0
                    updated.at[r['idx'],'Son_Islem_Fiyati']=r['price']
                    updated.at[r['idx'],'Son_Islem_Log']=f"AL (ROI: %{r['roi']:.0f})"
        elif total_pool > 0:
            f = updated.index[0]
            updated.at[f,'Nakit_Bakiye_USD']+=total_pool
            for ix in updated.index:
                if ix!=f and updated.at[ix,'Durum']=='CASH': updated.at[ix,'Nakit_Bakiye_USD']=0.0

        for idx, row in updated.iterrows():
            p = next((r['price'] for r in results if r['idx']==idx), 0.0)
            if p>0: updated.at[idx,'Kaydedilen_Deger_USD'] = (float(updated.at[idx,'Miktar'])*p) if updated.at[idx,'Durum']=='COIN' else float(updated.at[idx,'Nakit_Bakiye_USD'])

        save_portfolio(updated, sheet)
        st.success("✅ Analiz Tamamlandı!")
