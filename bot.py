import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import gspread
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

# =============================================================================
# 1. AYARLAR VE PORTFÖY YÖNETİMİ (GÖVDE)
# =============================================================================

SHEET_ID = "16zjLeps0t1P26OF3o7XQ-djEKKZtZX6t5lFxLmnsvpE"  # Senin Sheet ID'n
CREDENTIALS_FILE = "service_account.json"  # JSON dosyanın adı

# Hız ve Performans Ayarları
USE_GA = False       # Genetik Algoritma her seferinde çalışırsa çok yavaşlar. Arada bir True yapabilirsin.
GA_GENERATIONS = 5   # GA çalışırsa kaç nesil baksın?
WINDOW_SIZE = 30     # Öğrenme penceresi

def connect_sheet():
    """Google Sheets bağlantısını kurar."""
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
        client = gspread.authorize(creds)
        return client.open_by_key(SHEET_ID).sheet1
    except Exception as e:
        print(f"Google Sheets Bağlantı Hatası: {e}")
        return None

def load_portfolio():
    """Portföy verilerini çeker ve DataFrame'e çevirir."""
    sheet = connect_sheet()
    if sheet is None: return pd.DataFrame(), None

    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    
    required_cols = ["Ticker", "Durum", "Miktar", "Son_Islem_Fiyati", 
                     "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD", 
                     "Son_Islem_Log", "Son_Islem_Zamani"]
    
    if df.empty: 
        # Eğer boşsa başlıkları oluştur
        return pd.DataFrame(columns=required_cols), sheet

    # Eksik kolon varsa tamamla
    for col in required_cols:
        if col not in df.columns:
            df[col] = 0.0 if "USD" in col or "Miktar" in col or "Fiyat" in col else "-"

    # Sayısal verileri temizle
    numeric_cols = ["Miktar", "Son_Islem_Fiyati", "Nakit_Bakiye_USD", "Baslangic_USD", "Kaydedilen_Deger_USD"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
    return df, sheet

def save_portfolio(df, sheet):
    """Güncellenmiş portföyü kaydeder."""
    if sheet is None: return
    try:
        df_export = df.copy()
        df_export = df_export.astype(str)
        sheet.clear()
        sheet.update([df_export.columns.values.tolist()] + df_export.values.tolist())
        print("✅ Portföy Google Sheets'e kaydedildi.")
    except Exception as e:
        print(f"Kaydetme Hatası: {e}")

# =============================================================================
# 2. KALMAN AI MOTORU (BEYİN)
# =============================================================================

def apply_kalman_filter(prices):
    """Fiyat serisindeki gürültüyü temizler."""
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
        df = yf.download(ticker, period="2y", interval="1d", progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        if 'close' not in df.columns and 'adj close' in df.columns:
            df['close'] = df['adj close']
        return df
    except:
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
    
    # Kalman Filtresi
    df_res['kalman_close'] = apply_kalman_filter(df_res['close'])
    
    # Feature Engineering
    df_res['log_ret'] = np.log(df_res['kalman_close'] / df_res['kalman_close'].shift(1))
    df_res['range'] = (df_res['high'] - df_res['low']) / df_res['close']
    df_res['trend_signal'] = np.where(df_res['close'] > df_res['kalman_close'], 1, -1)
    
    # Target (Gelecek Yönü)
    df_res['target'] = (df_res['close'].shift(-1) > df_res['close']).astype(int)
    
    df_res.dropna(inplace=True)
    df_res = add_ma5_and_score(df_res, timeframe)
    df_res.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_res.dropna(inplace=True)
    
    return df_res

def train_models_for_window(train_df, rf_depth=5, xgb_params=None, n_hmm=3):
    features = ['log_ret','range','trend_signal','ma5_score']
    X = train_df[features]
    y = train_df['target']
    
    scaler = StandardScaler()
    try:
        X_s = scaler.fit_transform(X)
    except: return None # Veri hatası durumunda

    # 1. Random Forest
    clf_rf = RandomForestClassifier(n_estimators=30, max_depth=rf_depth, n_jobs=-1, random_state=42)
    clf_rf.fit(X, y)
    
    # 2. XGBoost
    if xgb_params is None:
        xgb_params = {'n_estimators':30, 'max_depth':3, 'learning_rate':0.1,'tree_method':'hist','n_jobs':-1}
    clf_xgb = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
    clf_xgb.fit(X, y)
    
    # 3. Stacking (Meta Model)
    meta_clf = None
    try:
        meta_X = np.vstack([clf_rf.predict_proba(X)[:,1], clf_xgb.predict_proba(X)[:,1]]).T
        meta_clf = LogisticRegression(max_iter=200)
        meta_clf.fit(meta_X, y)
    except: pass
    
    # 4. HMM
    hmm_model = None
    try:
        X_hmm = train_df[['log_ret','range']].values
        Xh_s = StandardScaler().fit_transform(X_hmm)
        hmm_model = GaussianHMM(n_components=n_hmm, covariance_type='diag', n_iter=50, random_state=42)
        hmm_model.fit(Xh_s)
    except: pass
    
    return {'rf': clf_rf, 'xgb': clf_xgb, 'meta': meta_clf, 'scaler': scaler, 'hmm': hmm_model}

def predict_with_models(models, row):
    if models is None: return 0
    rf_prob = xgb_prob = 0.5
    stack_sig = hmm_sig = 0.0
    
    features = ['log_ret','range','trend_signal','ma5_score']
    
    try:
        Xrow = row[features].values.reshape(1,-1)
        rf_prob = models['rf'].predict_proba(pd.DataFrame(Xrow, columns=features))[0][1]
        xgb_prob = models['xgb'].predict_proba(pd.DataFrame(Xrow, columns=features))[0][1]
    except: pass
    
    # Stacking Sinyali
    try:
        if models['meta']:
            stack_prob = models['meta'].predict_proba(np.array([[rf_prob,xgb_prob]]))[0][1]
            stack_sig = (stack_prob-0.5)*2
        else:
            stack_sig = ((rf_prob+xgb_prob)/2-0.5)*2
    except: 
        stack_sig = ((rf_prob+xgb_prob)/2-0.5)*2
        
    # HMM Sinyali
    try:
        if models['hmm']:
            Xh = row[['log_ret','range']].values.reshape(1,-1)
            probs = models['hmm'].predict_proba(StandardScaler().fit_transform(Xh))[0]
            bull = np.argmax(models['hmm'].means_[:,0])
            bear = np.argmin(models['hmm'].means_[:,0])
            hmm_sig = probs[bull]-probs[bear]
    except: hmm_sig = 0.0
    
    k_trend = row['trend_signal']
    
    # Ağırlıklı Karar
    combined = (hmm_sig * 0.25) + (stack_sig * 0.35) + (k_trend * 0.4)
    return combined

# -------------------- GA OPTİMİZASYONU (OPSİYONEL) --------------------
def walk_forward_splits(df, n_splits=3, test_size_ratio=0.2):
    # ... GA fonksiyonu için gerekli split logic ...
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

def ga_optimize_params_light(df, n_gen=5, pop_size=8):
    # GA tanımları
    if not hasattr(creator, 'FitnessMax'):
        creator.create('FitnessMax', base.Fitness, weights=(1.0,), overwrite=True)
        creator.create('Individual', list, fitness=creator.FitnessMax, overwrite=True)
        
    toolbox = base.Toolbox()
    toolbox.register('rf_depth', np.random.randint, 3, 13)
    toolbox.register('xgb_max_depth', np.random.randint, 2, 7)
    toolbox.register('xgb_eta', np.random.uniform, 0.01, 0.3)
    toolbox.register('individual', tools.initCycle, creator.Individual,
                     (toolbox.rf_depth, toolbox.xgb_max_depth, toolbox.xgb_eta), n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    def eval_individual(ind):
        rf_depth, xgb_md, xgb_eta = ind
        xgb_params = {'max_depth':int(xgb_md), 'learning_rate':float(xgb_eta), 'n_estimators':30, 'tree_method':'hist', 'n_jobs':-1}
        
        splits = walk_forward_splits(df, n_splits=2) # Hızlı olsun diye 2 split
        rois = []
        for tr, val, tst in splits:
            tr_df = df.iloc[0:val.stop]
            tst_df = df.iloc[tst]
            if len(tst_df) == 0: continue
            
            models = train_models_for_window(tr_df, rf_depth=int(rf_depth), xgb_params=xgb_params)
            cash = 100; coin = 0
            
            for idx in tst_df.index:
                row = df.loc[idx]
                sig = predict_with_models(models, row)
                p = row['close']
                if sig > 0.25 and cash>0: coin=cash/p; cash=0
                elif sig < -0.25 and coin>0: cash=coin*p; coin=0
            
            final = cash + (coin * tst_df.iloc[-1]['close'])
            rois.append((final-100)/100)
            
        return (np.mean(rois) if rois else -1,)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register('avg', np.mean)
    
    try:
        algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=n_gen, stats=stats, halloffame=hof, verbose=False)
        best = hof[0]
        return {'rf_depth': int(best[0]), 'xgb_params': {'max_depth':int(best[1]), 'learning_rate':float(best[2]), 'n_estimators':30, 'tree_method':'hist', 'n_jobs':-1}}
    except:
        return None

# =============================================================================
# 3. KÖPRÜ (BOT VE MODELİN BULUŞMASI)
# =============================================================================

def get_live_decision(ticker):
    """
    Yeni modelin 'Turnuva Mantığı' yerine geçen 'Canlı Karar' fonksiyonu.
    En iyi Timeframe'i bulur, modelleri eğitir ve SON kararı verir.
    """
    print(f" > Veri indiriliyor: {ticker}...")
    raw_df = get_raw_data(ticker)
    if raw_df is None: return "HATA", 0.0
    
    current_price = float(raw_df['close'].iloc[-1])
    
    # Timeframe Turnuvası
    timeframes = {'GÜNLÜK': 'D', 'HAFTALIK': 'W', 'AYLIK': 'M'}
    best_score = -9999
    final_decision = "BEKLE"
    winning_tf = "YOK"
    
    for tf_name, tf_code in timeframes.items():
        df = process_data(raw_df, tf_code)
        if df is None: continue
        
        # GA Optimizasyonu (Eğer açıksa)
        params = None
        if USE_GA:
            print(f"   > {tf_name} için GA çalışıyor...")
            params = ga_optimize_params_light(df, n_gen=GA_GENERATIONS)
        
        # Parametreleri ayarla
        rf_depth = params['rf_depth'] if params else 5
        xgb_p = params['xgb_params'] if params else None
        
        # SON PENCERE EĞİTİMİ (Son 60 barı kullanarak modelleri eğit)
        train_df = df.iloc[-60:] # Son 60 mumda eğit
        if len(train_df) < 20: continue
        
        # Modelleri yarat
        models = train_models_for_window(train_df, rf_depth=rf_depth, xgb_params=xgb_p)
        
        # CANLI TAHMİN (Listenin en sonundaki eleman = ŞİMDİ)
        # Burada son satırı tahmin etmeye çalışıyoruz
        last_row = df.iloc[-1]
        signal = predict_with_models(models, last_row)
        
        # Basit bir 'Recent Performance' puanı (Son 5 bardaki başarı gibi düşünebiliriz)
        # Turnuva kazananını belirlemek için sinyal gücüne ve trende bakıyoruz
        score = abs(signal) 
        
        if score > best_score:
            best_score = score
            winning_tf = tf_name
            
            # Karar Eşikleri
            if signal > 0.25: final_decision = "AL"
            elif signal < -0.25: final_decision = "SAT"
            else: final_decision = "BEKLE"

    print(f"   > Kazanan: {winning_tf} | Sinyal Gücü: {best_score:.2f} | Karar: {final_decision}")
    return final_decision, current_price

# =============================================================================
# 4. ANA DÖNGÜ (BOT ÇALIŞTIRMA)
# =============================================================================

def main():
    print("\n=== KALMAN AI TRADER (OTONOM) BAŞLATILIYOR ===")
    tz = pytz.timezone('Europe/Istanbul')
    time_str = datetime.now(tz).strftime("%d-%m %H:%M")
    
    # Portföyü Oku
    pf_df, sheet = load_portfolio()
    if pf_df.empty: 
        print("Portföy okunamadı veya boş.")
        return

    updated_portfolio = pf_df.copy()
    
    # Her satır (coin) için işlem yap
    for idx, row in updated_portfolio.iterrows():
        ticker = row['Ticker']
        if not ticker or ticker == "-": continue
        
        print(f"\nAnaliz ediliyor: {ticker}")
        
        # --- YENİ BEYİN DEVREYE GİRİYOR ---
        decision, current_price = get_live_decision(ticker)
        
        if current_price <= 0 or decision == "HATA":
            print(f"❌ Veri hatası: {ticker}")
            continue
        
        # --- İŞLEM MANTIĞI (MEVCUT BOT YAPISI) ---
        current_status = row['Durum']
        
        # 1. SATIŞ SİNYALİ
        if current_status == 'COIN' and decision == 'SAT':
            amount = float(row['Miktar'])
            cash_val = amount * current_price
            
            updated_portfolio.at[idx, 'Durum'] = 'CASH'
            updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = cash_val
            updated_portfolio.at[idx, 'Miktar'] = 0.0
            updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = current_price
            updated_portfolio.at[idx, 'Son_Islem_Log'] = f"SATILDI ({winning_tf if 'winning_tf' in locals() else 'AI'})"
            updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
            print(f"🔴 SATIŞ YAPILDI: {ticker} @ ${current_price:.2f} -> ${cash_val:.2f}")
            
        # 2. ALIŞ SİNYALİ
        elif current_status == 'CASH' and decision == 'AL':
            cash_val = float(row['Nakit_Bakiye_USD'])
            if cash_val > 1.0: # 1 Dolar altı işlem yapma
                amount = cash_val / current_price
                
                updated_portfolio.at[idx, 'Durum'] = 'COIN'
                updated_portfolio.at[idx, 'Miktar'] = amount
                updated_portfolio.at[idx, 'Nakit_Bakiye_USD'] = 0.0
                updated_portfolio.at[idx, 'Son_Islem_Fiyati'] = current_price
                updated_portfolio.at[idx, 'Son_Islem_Log'] = f"ALINDI ({winning_tf if 'winning_tf' in locals() else 'AI'})"
                updated_portfolio.at[idx, 'Son_Islem_Zamani'] = time_str
                print(f"🟢 ALIŞ YAPILDI: {ticker} @ ${current_price:.2f} -> {amount:.4f} Adet")
        
        else:
            print(f"⚪ İşlem Yok: {ticker} ({current_status}) -> Karar: {decision}")

        # 3. DEĞER GÜNCELLEME (Her durumda güncelle)
        if updated_portfolio.at[idx, 'Durum'] == 'COIN':
            val = float(updated_portfolio.at[idx, 'Miktar']) * current_price
        else:
            val = float(updated_portfolio.at[idx, 'Nakit_Bakiye_USD'])
        
        updated_portfolio.at[idx, 'Kaydedilen_Deger_USD'] = val

    # Google Sheets'e Kaydet
    save_portfolio(updated_portfolio, sheet)
    print("\n✅ Tüm işlemler tamamlandı ve Sheets güncellendi.")

if __name__ == "__main__":
    main()
