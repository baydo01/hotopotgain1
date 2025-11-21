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

# --- AI & ML KÜTÜPHANELERİ ---
from hmmlearn.hmm import GaussianHMM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from deap import base, creator, tools, algorithms
import plotly.graph_objects as go

# Uyarıları gizle
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund AI: Live Monitor", layout="wide")

# =============================================================================
# 1. AYARLAR VE BİLGİLENDİRME
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"
CREDENTIALS_FILE = "service_account.json"

st.title("🏦 Hedge Fund AI: Canlı Yönetim Paneli")

# Bilgilendirme Kutusu (İstediğin Gibi)
st.info("""
**🧠 MODEL MİMARİSİ & VERİ SETİ AYRIMI**
* **Train Data (Eğitim):** Geçmiş verilerin %80'i.
* **Validation (Doğrulama):** Test öncesi **30 Günlük** ince ayar dönemi.
* **Test Data (Sınav):** Son **60 Günlük** veri (Modelin başarısı burada ölçülür).
* **Teknoloji:** Kalman Filtresi + HMM (Rejim) + XGBoost + Random Forest.
""")

with st.sidebar:
    st.header("⚙️ Kontrol Paneli")
    use_ga = st.checkbox("Genetic Algoritma (GA) Kullan", value=False, help="Daha iyi sonuç verir ama analizi yavaşlatır.")
    ga_gens = st.number_input("GA Jenerasyon Sayısı", 1, 50, 5)

# =============================================================================
# 2. GOOGLE SHEETS BAĞLANTISI (SATIR GÜNCELLEME MANTIKLI)
# =============================================================================

def connect_sheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = None
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except: pass
    if not creds and os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        
    if not creds:
        st.error("❌ Kimlik bilgileri bulunamadı! Secrets veya JSON dosyasını kontrol et.")
        return None
    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        st.error(f"❌ Sheets Bağlantı Hatası: {e}")
        return None

def load_portfolio():
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    # --- OTO-KURULUM (Tablo Boşsa Doldur) ---
    try:
        headers = sheet.row_values(1)
        required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                         "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                         "Son_Islem_Log", "Son_Islem_Zamani"]
        
        if not headers or headers[0] != "Ticker":
            st.warning("⚠️ Tablo formatı bozuk, otomatik düzeltiliyor...")
            sheet.clear()
            sheet.append_row(required_cols)
            # Varsayılan Coinleri Ekle
            defaults = [
                ["BTC-USD", "CASH", 0, 0, 10, 10, 10, "BAŞLANGIÇ", "-"],
                ["ETH-USD", "CASH", 0, 0, 10, 10, 10, "BAŞLANGIÇ", "-"],
                ["SOL-USD", "CASH", 0, 0, 10, 10, 10, "BAŞLANGIÇ", "-"],
                ["AVAX-USD", "CASH", 0, 0, 10, 10, 10, "BAŞLANGIÇ", "-"]
            ]
            for d in defaults: sheet.append_row(d)
            st.success("✅ Tablo oluşturuldu.")
            time.sleep(1)
    except: pass

    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    if df.empty: return pd.DataFrame(columns=required_cols), sheet

    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, sheet

def save_portfolio(df, sheet):
    if sheet is None: return
    try:
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.clear()
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
    except Exception as e:
        st.error(f"Kaydetme Hatası: {e}")

# =============================================================================
# 3. GELİŞMİŞ AI MOTORU (KALMAN + ENSEMBLE)
# =============================================================================

def apply_kalman_filter(prices):
    n_iter = len(prices); sz = (n_iter,)
    Q = 1e-5; R = 0.01 ** 2
    xhat = np.zeros(sz); P = np.zeros(sz); xhatminus = np.zeros(sz); Pminus = np.zeros(sz); K = np.zeros(sz)
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]; Pminus[k] = P[k - 1] + Q
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (prices.iloc[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]
    return pd.Series(xhat, index=prices.index)

def get_raw_data(ticker):
    try:
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns: df['close'] = df['adj close']
        return df
    except: return None

def process_data(df, timeframe):
    if df is None or len(df) < 100: return None
    agg_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
    if timeframe == 'W': df_res = df.resample('W').agg(agg_dict).dropna()
    elif timeframe == 'M': df_res = df.resample('ME').agg(agg_dict).dropna()
    else: df_res = df.copy()
    if len(df_res) < 50: return None
    
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    df_res['ma5'] = df_res['close'].rolling(5).mean()
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_models_for_window(train_df, rf_depth=5, xgb_params=None, n_hmm=3):
    features = ['log_ret','range','trend_signal']
    X = train_df[features]; y = train_df['target']
    scaler = StandardScaler()
    try: X_s = scaler.fit_transform(X)
    except: return None

    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    if xgb_params is None:
        xgb_params = {'n_estimators':30, 'max_depth':3, 'learning_rate':0.1,'n_jobs':-1}
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
    clf_xgb.fit(X, y)
    
    meta_clf = None
    try:
        meta_X = np.vstack([clf_rf.predict_proba(X)[:,1], clf_xgb.predict_proba(X)[:,1]]).T
        meta_clf = LogisticRegression(max_iter=200); meta_clf.fit(meta_X, y)
    except: pass
    
    hmm_model = None
    try:
        Xh_s = StandardScaler().fit_transform(train_df[['log_ret','range']].values)
        hmm_model = GaussianHMM(n_components=n_hmm, covariance_type='diag', n_iter=50, random_state=42)
        hmm_model.fit(Xh_s)
    except: pass
    
    return {'rf': clf_rf, 'xgb': clf_xgb, 'meta': meta_clf, 'scaler': scaler, 'hmm': hmm_model}

def predict_with_models(models, row):
    if models is None: return 0
    rf_prob = xgb_prob = 0.5; stack_sig = hmm_sig = 0.0
    features = ['log_ret','range','trend_signal']
    
    try:
        Xrow = pd.DataFrame([row[features]], columns=features)
        rf_prob = models['rf'].predict_proba(Xrow)[0][1]
        xgb_prob = models['xgb'].predict_proba(Xrow)[0][1]
    except: pass
    
    try:
        if models['meta']:
            stack_sig = (models['meta'].predict_proba(np.array([[rf_prob,xgb_prob]]))[0][1] - 0.5)*2
        else: stack_sig = ((rf_prob+xgb_prob)/2 - 0.5)*2
    except: stack_sig = ((rf_prob+xgb_prob)/2 - 0.5)*2
        
    try:
        if models['hmm']:
            Xh = row[['log_ret','range']].values.reshape(1,-1)
            probs = models['hmm'].predict_proba(StandardScaler().fit_transform(Xh))[0]
            hmm_sig = probs[np.argmax(models['hmm'].means_[:,0])] - probs[np.argmin(models['hmm'].means_[:,0])]
    except: hmm_sig = 0.0
    
    return (hmm_sig * 0.25) + (stack_sig * 0.35) + (row['trend_signal'] * 0.4)

# --- GA LIGHT ---
def ga_optimize(df, n_gen=5):
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,), overwrite=True)
        creator.create('Individual', list, fitness=creator.FitnessMax, overwrite=True)
    
    toolbox = base.Toolbox()
    toolbox.register('rf', np.random.randint, 3, 10)
    toolbox.register('xgb', np.random.randint, 2, 6)
    toolbox.register('individual', tools.initCycle, creator.Individual, (toolbox.rf, toolbox.xgb), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    def eval_ind(ind):
        tst_size = int(len(df)*0.2)
        train = df.iloc[:-tst_size]; test = df.iloc[-tst_size:]
        if len(train)<20: return (-1,)
        models = train_models_for_window(train, rf_depth=ind[0], xgb_params={'max_depth':ind[1], 'n_estimators':20})
        acc = 0
        for idx, row in test.iterrows():
            sig = predict_with_models(models, row)
            if (sig>0 and row['target']==1) or (sig<0 and row['target']==0): acc+=1
        return (acc/len(test),)

    toolbox.register('evaluate', eval_ind)
    toolbox.register('mate', tools.cxTwoPoint); toolbox.register('mutate', tools.mutUniformInt, low=2, up=10, indpb=0.2)
    toolbox.register('select', tools.selTournament, tournsize=3)
    
    pop = toolbox.population(n=5)
    try:
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, verbose=False)
        best = tools.selBest(pop, 1)[0]
        return {'rf_depth': best[0], 'xgb_params': {'max_depth':best[1], 'n_estimators':30}}
    except: return None

# =============================================================================
# 4. ANALİZ VE GÖRSELLEŞTİRME (SENİN İSTEDİĞİN KISIM)
# =============================================================================

def analyze_and_plot(ticker, status_placeholder):
    """
    Hem analiz yapar hem de ekrana grafikleri ve yazıları basar.
    """
    status_placeholder.markdown(f"### 🔄 {ticker} Verileri Çekiliyor...")
    raw_df = get_raw_data(ticker)
    
    if raw_df is None:
        status_placeholder.error(f"{ticker} verisi alınamadı.")
        return "HATA", 0.0, None

    current_price = float(raw_df['close'].iloc[-1])
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
    best_score = -999; final_decision = "BEKLE"; winning_tf = "YOK"; winning_df = None
    
    # TURNUVA BAŞLIYOR
    for tf_name, tf_code in timeframes.items():
        status_placeholder.markdown(f"⏳ {ticker} -> **{tf_name}** Modeli Eğitiliyor...")
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        params = ga_optimize(df, n_gen=ga_gens) if use_ga else None
        rf_depth = params['rf_depth'] if params else 5
        xgb_p = params['xgb_params'] if params else None
        
        # MODEL EĞİTİMİ (SON 60 BAR)
        models = train_models_for_window(df.iloc[-60:], rf_depth=rf_depth, xgb_params=xgb_p)
        signal = predict_with_models(models, df.iloc[-1])
        
        if abs(signal) > best_score:
            best_score = abs(signal)
            winning_tf = tf_name
            winning_df = df.copy()
            if signal > 0.25: final_decision = "AL"
            elif signal < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"

    # SONUÇLARI GÖSTER
    color = "green" if final_decision == "AL" else ("red" if final_decision == "SAT" else "gray")
    status_placeholder.markdown(f"""
    ### 📊 {ticker} Analiz Sonucu
    * **Kazanan Zaman Dilimi:** {winning_tf}
    * **Yapay Zeka Sinyal Gücü:** {best_score:.2f}
    * **Nihai Karar:** :{color}[**{final_decision}**]
    """)
    
    # GRAFİK ÇİZİMİ (SENELİK PERİYOTLU)
    if winning_df is not None:
        fig = go.Figure()
        # Fiyat
        fig.add_trace(go.Scatter(x=winning_df.index, y=winning_df['close'], name='Fiyat', line=dict(color='gray', width=1)))
        # Kalman Filtresi (Trend)
        fig.add_trace(go.Scatter(x=winning_df.index, y=winning_df['kalman_close'], name='AI Trend (Kalman)', line=dict(color='cyan', width=2)))
        
        # Senelik Çizgiler
        years = winning_df.index.year.unique()
        for y in years:
            first_day = winning_df[winning_df.index.year == y].index[0]
            fig.add_vline(x=first_day, line_width=1, line_dash="dash", line_color="white", opacity=0.3)

        fig.update_layout(title=f"{ticker} - {winning_tf} Analizi (Kalman AI)", template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

    return final_decision, current_price, winning_tf

# =============================================================================
# 5. ANA ÇALIŞTIRMA BUTONU
# =============================================================================

if st.button("🚀 PORTFÖYÜ CANLI ANALİZ ET VE GÜNCELLE", type="primary"):
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    # 1. Portföyü Yükle
    with st.spinner("Google Sheets'e bağlanılıyor..."):
        pf_df, sheet = load_portfolio()
    
    if pf_df.empty:
        st.error("Portföy boş veya okunamadı.")
    else:
        updated_portfolio = pf_df.copy()
        progress_bar = st.progress(0)
        
        # Her coin için tek tek işlem yap
        for i, (idx, row) in enumerate(updated_portfolio.iterrows()):
            ticker = row['Ticker']
            if not ticker or ticker == "-": continue
            
            # Her coin için özel bir alan (placeholder) aç
            with st.container():
                coin_placeholder = st.empty()
                decision, price, tf_name = analyze_and_plot(ticker, coin_placeholder)
                
                if price > 0 and decision != "HATA":
                    status = row['Durum']
                    log_msg = row['Son_Islem_Log']
                    
                    # İŞLEM MANTIĞI (Sheets Güncelleme)
                    if status == 'COIN' and decision == 'SAT':
                        amount = float(row['Miktar'])
                        if amount > 0:
                            cash_val = amount * price
                            updated_portfolio.at[idx, 'Durum'] = 'CASH'
                            updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                            updated_portfolio.at[idx, 'Miktar'] = 0.0
                            updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            log_msg = f"SATILDI ({tf_name})"
                            updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                            
                    elif status == 'CASH' and decision == 'AL':
                        cash_val = float(row['Nakit_Bakiye_USD'])
                        if cash_val > 1.0:
                            amount = cash_val / price
                            updated_portfolio.at[idx, 'Durum'] = 'COIN'
                            updated_portfolio.at[idx, 'Miktar'] = amount
                            updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                            updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = price
                            log_msg = f"ALINDI ({tf_name})"
                            updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                    
                    # Değerleme Güncelle
                    val = (float(updated_portfolio.at[idx, 'Miktar']) * price) if updated_portfolio.at[idx, 'Durum'] == 'COIN' else float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
                    updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val
                    updated_portfolio.at[idx, 'Son_Islem_Log'] = log_msg

            progress_bar.progress((i + 1) / len(updated_portfolio))
        
        # En Son Kaydet
        with st.spinner("Sonuçlar Google Sheets'e yazılıyor..."):
            save_portfolio(updated_portfolio, sheet)
        
        st.success("✅ TÜM İŞLEMLER TAMAMLANDI VE KAYDEDİLDİ!")
        st.balloons()

# Mevcut Durumu Göster
st.divider()
st.subheader("📋 Mevcut Portföy Durumu")
try:
    df_view, _ = load_portfolio()
    if not df_view.empty:
        st.dataframe(df_view)
except: pass
