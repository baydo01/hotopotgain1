from sklearn.decomposition import PCA
from arch import arch_model
from pyearth import Earth
from statsmodels.tsa.arima.model import ARIMA

def train_meta_learner(df, params=None):
    rf_d = params['rf_depth'] if params else 5
    rf_n = params['rf_nest'] if params else 50
    test_size = 60
    if len(df) < test_size + 50: return 0.0, None
    
    train = df.iloc[:-test_size]; test = df.iloc[-test_size:]
    X_tr_base = train[['log_ret', 'range', 'heuristic', 'your_rule']]
    y_tr = train['target']

    # --- Alt Modeller ---
    # RandomForest
    rf = RandomForestClassifier(n_estimators=rf_n, max_depth=rf_d, random_state=42).fit(X_tr_base, y_tr)
    rf_pred = rf.predict_proba(X_tr_base)[:,1]

    # XGBoost
    xgb_c = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=50, max_depth=3)
    xgb_c.fit(X_tr_base, y_tr)
    xgb_pred = xgb_c.predict_proba(X_tr_base)[:,1]

    # HMM
    scaler = StandardScaler()
    X_hmm = scaler.fit_transform(train[['log_ret','range']])
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=50, random_state=42)
    try: hmm.fit(X_hmm)
    except: hmm=None
    hmm_pred = np.zeros(len(train))
    if hmm:
        pr = hmm.predict_proba(X_hmm)
        bull = np.argmax(hmm.means_[:,0]); bear = np.argmin(hmm.means_[:,0])
        hmm_pred = pr[:, bull] - pr[:, bear]

    # ARIMA (tek feature örnek)
    arima_pred = []
    for i in range(len(train)):
        try:
            model = ARIMA(train['close'].iloc[max(0,i-20):i+1], order=(1,1,0))
            res = model.fit()
            arima_pred.append(res.forecast()[0])
        except: arima_pred.append(0)
    arima_pred = np.array(arima_pred)

    # ARCH/GARCH (volatility)
    arch_pred = []
    for i in range(len(train)):
        try:
            am = arch_model(train['log_ret'].iloc[max(0,i-50):i+1]*100)
            res = am.fit(disp="off")
            arch_pred.append(res.conditional_volatility[-1]/100)
        except: arch_pred.append(0)
    arch_pred = np.array(arch_pred)

    # GPBoost / Earth (MARS)
    earth_model = Earth(max_terms=5).fit(X_tr_base, y_tr)
    earth_pred = earth_model.predict(X_tr_base)

    # Heuristic + YourRule
    heuristic_pred = train['heuristic'].values
    your_rule_pred = train['your_rule'].values

    # --- PCA Pipeline ---
    meta_X = np.vstack([rf_pred, xgb_pred, hmm_pred, arima_pred, arch_pred, earth_pred, heuristic_pred, your_rule_pred]).T
    meta_X = StandardScaler().fit_transform(meta_X)
    pca = PCA()
    meta_X_pca = pca.fit_transform(meta_X)
    var_cum = np.cumsum(pca.explained_variance_ratio_)
    n_comp = np.searchsorted(var_cum, 0.9) + 1  # %90 varyansı kapsayan ilk bileşen sayısı
    meta_X_pca_final = meta_X_pca[:,:n_comp]

    # --- Meta Model ---
    meta_model = LogisticRegression(max_iter=500).fit(meta_X_pca_final, y_tr)

    # --- Test Set Tahmini ---
    X_te_base = test[['log_ret', 'range', 'heuristic', 'your_rule']]
    rf_te = rf.predict_proba(X_te_base)[:,1]
    xgb_te = xgb_c.predict_proba(X_te_base)[:,1]
    X_hmm_te = scaler.transform(test[['log_ret','range']])
    hmm_te = hmm.predict_proba(X_hmm_te)[:, bull] - hmm.predict_proba(X_hmm_te)[:, bear] if hmm else np.zeros(len(test))
    # ARIMA
    arima_te = []
    for i in range(len(test)):
        try:
            model = ARIMA(test['close'].iloc[max(0,i-20):i+1], order=(1,1,0))
            res = model.fit()
            arima_te.append(res.forecast()[0])
        except: arima_te.append(0)
    arima_te = np.array(arima_te)
    # ARCH
    arch_te = []
    for i in range(len(test)):
        try:
            am = arch_model(test['log_ret'].iloc[max(0,i-50):i+1]*100)
            res = am.fit(disp="off")
            arch_te.append(res.conditional_volatility[-1]/100)
        except: arch_te.append(0)
    arch_te = np.array(arch_te)
    earth_te = earth_model.predict(X_te_base)
    heuristic_te = test['heuristic'].values
    your_rule_te = test['your_rule'].values

    meta_X_test = np.vstack([rf_te, xgb_te, hmm_te, arima_te, arch_te, earth_te, heuristic_te, your_rule_te]).T
    meta_X_test = StandardScaler().fit_transform(meta_X_test)
    meta_X_test_pca = PCA(n_components=n_comp).fit_transform(meta_X_test)
    probs = meta_model.predict_proba(meta_X_test_pca)[:,1]

    # --- Simülasyon ---
    sim_eq=[100]; hodl_eq=[100]; cash=100; coin=0; p0=test['close'].iloc[0]
    for i in range(len(test)):
        p = test['close'].iloc[i]; s=(probs[i]-0.5)*2
        if s>0.25 and cash>0: coin=cash/p; cash=0
        elif s<-0.25 and coin>0: cash=coin*p; coin=0
        sim_eq.append(cash+coin*p); hodl_eq.append((100/p0)*p)
    final_signal=(probs[-1]-0.5)*2
    info={'weights': meta_model.coef_[0],
          'bot_eq': sim_eq[1:],'hodl_eq': hodl_eq[1:],
          'dates': test.index,'alpha': (sim_eq[-1]-hodl_eq[-1]),
          'bot_roi': (sim_eq[-1]-100),'hodl_roi': (hodl_eq[-1]-100),
          'conf': probs[-1],'my_score': test['heuristic'].iloc[-1]}
    return final_signal, info
