
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time
import threading
import warnings
import gspread
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

# Uyarıları gizle
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Hedge Fund Manager: Kalman AI", layout="wide")
st.title("🏦 Hedge Fund Manager: Portföy Yönetimi (Sheets Entegre)")

# =============================================================================
# 1. AYARLAR (SENİN ESKİ KODUN VE YENİ PARAMETRELER)
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE" 
CREDENTIALS_FILE = "service_account.json"

with st.sidebar:
    st.header("⚙️ Bot Ayarları")
    update_interval = st.number_input("Döngü Hızı (Saniye)", 60, 3600, 300)
    use_ga = st.checkbox("Genetic Algoritma Kullan (Yavaşlatır)", False)
    ga_gens = st.number_input("GA Jenerasyon", 1, 50, 5)
    
    st.info("Bot, Google Sheets'teki portföy tablosunu okur, Kalman AI ile karar verir ve bakiyeleri günceller.")

# =============================================================================
# 2. GOOGLE SHEETS BAĞLANTISI (ESKİ KODUN YAPISI KORUNDU)
# =============================================================================

def connect_sheet():
    """Google Sheets bağlantısını kurar (Secrets veya Local File)."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    creds = None
    # Önce Streamlit Secrets kontrol et
    if "gcp_service_account" in st.secrets:
        try:
            creds_dict = dict(st.secrets["gcp_service_account"])
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        except: pass
    
    # Yoksa yerel dosyaya bak
    if not creds and os.path.exists(CREDENTIALS_FILE):
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        
    if not creds:
        st.error("Kimlik bilgileri (Secrets veya JSON) bulunamadı!")
        return None

    try:
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        st.error(f"Sheets Bağlantı Hatası: {e}")
        return None

def load_portfolio():
    """Mevcut portföy durumunu çeker."""
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                     "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                     "Son_Islem_Log", "Son_Islem_Zamani"]
    
    if df.empty: return pd.DataFrame(columns=required_cols), sheet

    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if "USD" in col or "Miktar" in col or "Fiyat" in col else "-"

    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            # Virgül/Nokta dönüşümü
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, sheet

def save_portfolio(df, sheet):
    """Güncellenmiş tabloyu yazar."""
    if sheet is None: return
    try:
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.clear()
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        # Streamlit loguna yazma, print kullan (terminalde görünür)
        print(f"[{datetime.now().strftime('%H:%M')}] Portföy Sheets'e kaydedildi.")
    except Exception as e:
        print(f"Kaydetme Hatası: {e}")

# =============================================================================
# 3. YENİ BEYİN: KALMAN + AI MOTORU
# =============================================================================

def apply_kalman_filter(prices):
    n_iter = len(prices)
    sz = (n_iter,)
    Q = 1e-5; R = 0.01 ** 2
    xhat = np.zeros(sz); P = np.zeros(sz)
    xhatminus = np.zeros(sz); Pminus = np.zeros(sz); K = np.zeros(sz)
    xhat[0] = prices.iloc[0]; P[0] = 1.0
    
    for k in range(1, n_iter):
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q
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
    
    # Hareketli Ortalama Skoru
    df_res['ma5'] = df_res['close'].rolling(5).mean()
    long_w = {'D':252, 'W':52, 'M':36}.get(timeframe, 36)
    df_res['ma5_score'] = (df_res['ma5'] - df_res['ma5'].rolling(long_w).mean()) / (df_res['ma5'].rolling(long_w).std() + 1e-9)
    
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    df_res.dropna(inplace=True)
    return df_res

def train_models_for_window(train_df, rf_depth=5, xgb_params=None, n_hmm=3):
    features = ['log_ret','range','trend_signal','ma5_score']
    X = train_df[features]; y = train_df['target']
    scaler = StandardScaler()
    
    try: X_s = scaler.fit_transform(X)
    except: return None

    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    if xgb_params is None:
        xgb_params = {'n_estimators':30, 'max_depth':3, 'learning_rate':0.1,'tree_method':'hist','n_jobs':-1}
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
    features = ['log_ret','range','trend_signal','ma5_score']
    
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

# --- GA LIGHT (Hafif Versiyon) ---
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
        # Basit backtest (Hız için son %20 veri)
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
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, verbose=False)
    best = tools.selBest(pop, 1)[0]
    return {'rf_depth': best[0], 'xgb_params': {'max_depth':best[1], 'n_estimators':30}}

# =============================================================================
# 4. KARAR MEKANİZMASI VE UYGULAMA
# =============================================================================

def get_ai_decision(ticker):
    """Yeni Beyin: Ticker alır, 'AL/SAT/BEKLE' ve Fiyat döner."""
    raw_df = get_raw_data(ticker)
    if raw_df is None: return "HATA", 0.0
    
    current_price = float(raw_df['close'].iloc[-1])
    
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W'}
    best_score = -999; final_decision = "BEKLE"; winning_tf = "YOK"
    
    for tf_name, tf_code in timeframes.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        params = ga_optimize(df, n_gen=ga_gens) if use_ga else None
        rf_depth = params['rf_depth'] if params else 5
        xgb_p = params['xgb_params'] if params else None
        
        # Son 60 bar ile eğit, ŞİMDİKİ zamana (son satıra) karar ver
        models = train_models_for_window(df.iloc[-60:], rf_depth=rf_depth, xgb_params=xgb_p)
        signal = predict_with_models(models, df.iloc[-1])
        
        if abs(signal) > best_score:
            best_score = abs(signal)
            winning_tf = tf_name
            if signal > 0.25: final_decision = "AL"
            elif signal < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"
            
    print(f"   > {ticker}: Kazanan TF: {winning_tf} | Sinyal: {best_score:.2f} | Karar: {final_decision}")
    return final_decision, current_price

# =============================================================================
# 5. OTOMASYON DÖNGÜSÜ
# =============================================================================

stop_flag = False

def background_bot_loop():
    global stop_flag
    while not stop_flag:
        try:
            print("\n--- Analiz Döngüsü Başlıyor ---")
            tz = pytz.timezone('Europe/Istanbul')
            time_str = datetime.now(tz).strftime("%d-%m %H:%M")
            
            pf_df, sheet = load_portfolio()
            if pf_df.empty:
                print("Portföy boş, bekleniyor...")
                time.sleep(60); continue
                
            updated = False
            
            for idx, row in pf_df.iterrows():
                if stop_flag: break
                ticker = row['Ticker']
                if not ticker or ticker == "-": continue
                
                # --- YENİ BEYNE SOR ---
                decision, price = get_ai_decision(ticker)
                
                if price <= 0 or decision == "HATA": continue
                
                status = row['Durum']
                
                # --- İŞLEM MANTIĞI (SENİN ŞABLONUNA UYGUN) ---
                if status == 'COIN' and decision == 'SAT':
                    # Satış: Coin Miktarını Nakite Çevir
                    amount = float(row['Miktar'])
                    if amount > 0:
                        cash_val = amount * price
                        pf_df.at[idx, 'Durum'] = 'CASH'
                        pf_df.at[idx, 'Nakit_Bakiye_USD'] = cash_val
                        pf_df.at[idx, 'Miktar'] = 0.0
                        pf_df.at[idx, 'Son_Islem_Fiyati'] = price
                        pf_df.at[idx, 'Son_Islem_Log'] = f"SATILDI (AI)"
                        pf_df.at[idx, 'Son_Islem_Zamani'] = time_str
                        updated = True
                        
                elif status == 'CASH' and decision == 'AL':
                    # Alış: Nakit Bakiyeyi Coine Çevir
                    cash_val = float(row['Nakit_Bakiye_USD'])
                    if cash_val > 1.0: # Min 1$ işlem
                        amount = cash_val / price
                        pf_df.at[idx, 'Durum'] = 'COIN'
                        pf_df.at[idx, 'Miktar'] = amount
                        pf_df.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                        pf_df.at[idx, 'Son_Islem_Fiyati'] = price
                        pf_df.at[idx, 'Son_Islem_Log'] = f"ALINDI (AI)"
                        pf_df.at[idx, 'Son_Islem_Zamani'] = time_str
                        updated = True
                
                # Değer Güncelleme (Her döngüde güncel değer yazılsın)
                if pf_df.at[idx, 'Durum'] == 'COIN':
                    val = float(pf_df.at[idx, 'Miktar']) * price
                else:
                    val = float(pf_df.at[idx, 'Nakit_Bakiye_USD'])
                
                # Sadece değer değişse bile kaydedelim ki canlı takip edilsin
                if abs(float(row['Kaydedilen_Deger_USD']) - val) > 0.1:
                    pf_df.at[idx, 'Kaydedilen_Deger_USD'] = val
                    updated = True

            if updated:
                save_portfolio(pf_df, sheet)
            
            print(f"Döngü bitti. {update_interval} saniye bekleniyor...")
            time.sleep(update_interval)
            
        except Exception as e:
            print(f"Döngü Hatası: {e}")
            time.sleep(60)

# =============================================================================
# 6. STREAMLIT ARAYÜZÜ
# =============================================================================

col1, col2 = st.columns(2)

with col1:
    if st.button("▶️ OTOMATİK BOTU BAŞLAT"):
        if not stop_flag:
            stop_flag = False
            t = threading.Thread(target=background_bot_loop, daemon=True)
            t.start()
            st.success("Bot Arka Planda Çalışıyor! Google Sheets'i takip edin.")
        else:
            st.warning("Bot zaten çalışıyor olabilir veya durdurma sinyali bekliyor.")

with col2:
    if st.button("⏹️ BOTU DURDUR"):
        stop_flag = True
        st.error("Durdurma sinyali gönderildi. Mevcut işlem bitince duracak.")

# Canlı Durum Göstergesi
st.divider()
st.subheader("📋 Canlı Portföy Özeti")

try:
    df_view, _ = load_portfolio()
    if not df_view.empty:
        st.dataframe(df_view[["Ticker", "Durum", "Kaydedilen_Deger_USD", "Son_Islem_Log"]])
        
        total_val = df_view["Kaydedilen_Deger_USD"].sum()
        st.metric("Toplam Portföy Değeri", f"${total_val:,.2f}")
except:
    st.write("Veri yüklenemedi veya Sheets bağlantısı yok.")
