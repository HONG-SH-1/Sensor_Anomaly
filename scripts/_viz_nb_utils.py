# Utils: load_deep_model은 .pt state_dict shape으로 hidden/latent/layers 추정

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, layers=2, dropout=0.1, **kw):
        super().__init__()
        self.model_name = 'LSTM-AE'
        drop = dropout if layers > 1 else 0
        self.enc = nn.LSTM(n_feat, hidden, layers, batch_first=True, dropout=drop)
        self.fc_e = nn.Linear(hidden, latent)
        self.fc_d = nn.Linear(latent, hidden)
        self.dec = nn.LSTM(hidden, hidden, layers, batch_first=True, dropout=drop)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.fc_e(h[-1])
        di = self.fc_d(z).unsqueeze(1).repeat(1, x.size(1), 1)
        dec, _ = self.dec(di)
        return self.out(dec)


class CNN1DAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, dropout=0.1, **kw):
        super().__init__()
        self.model_name = 'CNN1D-AE'
        self.enc = nn.Sequential(
            nn.Conv1d(n_feat, hidden, 7, padding=3), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden * 2, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc_e = nn.Linear(hidden * 2, latent)
        self.fc_d = nn.Linear(latent, hidden * 2)
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(hidden * 2, hidden, 5, padding=2), nn.ReLU(), nn.Dropout(dropout),
            nn.ConvTranspose1d(hidden, n_feat, 7, padding=3),
        )

    def forward(self, x):
        W = x.size(1)
        z = self.fc_e(self.enc(x.permute(0, 2, 1)).squeeze(-1))
        di = self.fc_d(z).unsqueeze(-1).repeat(1, 1, W)
        return self.dec(di).permute(0, 2, 1)


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=32, nhead=4, layers=2, dropout=0.1, **kw):
        super().__init__()
        self.model_name = 'Transformer-AE'
        nhead = max(h for h in [1, 2, 4, 8] if hidden % h == 0 and h <= nhead)
        self.proj = nn.Linear(n_feat, hidden)
        enc_l = nn.TransformerEncoderLayer(hidden, nhead, hidden * 4, dropout, batch_first=True)
        self.tenc = nn.TransformerEncoder(enc_l, layers)
        self.fc_e = nn.Linear(hidden, latent)
        self.fc_d = nn.Linear(latent, hidden)
        dec_l = nn.TransformerDecoderLayer(hidden, nhead, hidden * 4, dropout, batch_first=True)
        self.tdec = nn.TransformerDecoder(dec_l, layers)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        p = self.proj(x)
        m = self.tenc(p)
        z = self.fc_e(m.mean(1))
        d = self.fc_d(z).unsqueeze(1).repeat(1, x.size(1), 1)
        return self.out(self.tdec(d, m))


DEEP_MODEL_CLASSES = {
    'LSTM-AE': LSTMAutoencoder,
    'CNN1D-AE': CNN1DAutoencoder,
    'Transformer-AE': TransformerAutoencoder,
}


def compute_vif(df_feat):
    X = df_feat.dropna().astype(float)
    return pd.DataFrame({
        'feature': X.columns,
        'VIF': [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values('VIF', ascending=False).reset_index(drop=True)


def iterative_vif_drop(df_feat, threshold, verbose=True):
    remaining = list(df_feat.columns)
    step = 0
    while True:
        step += 1
        vif = compute_vif(df_feat[remaining])
        top_vif = vif.iloc[0]['VIF']
        top_feat = vif.iloc[0]['feature']
        if top_vif <= threshold:
            if verbose:
                print(f'  [step {step}] Done — all VIF <= {threshold}')
            break
        if verbose:
            print(f'  [step {step}] Drop {top_feat:<15s}  VIF={top_vif:.2f}')
        remaining.remove(top_feat)
    return remaining, pd.DataFrame()


def get_errors(model, X_tensor, batch=512):
    model.eval()
    errs = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch):
            b = X_tensor[i:i + batch]
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                recon = model(b)
            errs.extend(((b - recon) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.array(errs)


def evaluate(errors, y_true, pct=THRESHOLD_PCT):
    errors = np.nan_to_num(errors, nan=0.0, posinf=0.0, neginf=0.0)
    thr = np.percentile(errors, pct)
    y_pred = (errors > thr).astype(int)
    uniq = len(np.unique(y_true))
    return dict(
        threshold=thr,
        y_pred=y_pred,
        errors=errors,
        f1=f1_score(y_true, y_pred, zero_division=0),
        f2=fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, errors) if uniq > 1 else 0.0,
        pr_auc=average_precision_score(y_true, errors) if uniq > 1 else 0.0,
    )


def make_windows(data, labels, win, step):
    X, y = [], []
    for s in range(0, len(data) - win, step):
        X.append(data[s:s + win])
        y.append(int(labels[s:s + win].max()))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def load_serving_config(checkpoint_dir):
    import json

    path = os.path.join(checkpoint_dir, 'model_serving_config.json')
    default = {
        'best_params': {
            'LSTM-AE': {'hidden': 64, 'latent': 32, 'layers': 2, 'dropout': 0.1},
            'CNN1D-AE': {'hidden': 64, 'latent': 32, 'layers': 2, 'dropout': 0.1},
            'Transformer-AE': {'hidden': 64, 'latent': 32, 'layers': 2, 'dropout': 0.1},
        },
        'window_configs': {
            'LSTM-AE': {'window_size': 50, 'threshold_pct': 95},
            'CNN1D-AE': {'window_size': 50, 'threshold_pct': 95},
            'Transformer-AE': {'window_size': 50, 'threshold_pct': 95},
        },
        'step_size': 10,
    }
    if not os.path.isfile(path):
        return default
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    for fam in ('best_params', 'window_configs'):
        if fam in raw:
            for k, v in raw[fam].items():
                default[fam].setdefault(k, {}).update(v)
    if 'step_size' in raw:
        default['step_size'] = int(raw['step_size'])
    return default


def _build_deep_model(model_name, n_feat, hidden, latent, layers, dropout):
    if model_name == 'Transformer-AE':
        return DEEP_MODEL_CLASSES[model_name](
            n_feat, hidden=hidden, latent=latent, layers=layers, dropout=dropout, nhead=4,
        )
    kw = dict(hidden=hidden, latent=latent, dropout=dropout)
    if model_name == 'LSTM-AE':
        kw['layers'] = layers
    return DEEP_MODEL_CLASSES[model_name](n_feat, **kw)


def _params_for_suffix(model_name, suffix, serving_cfg):
    """default.pt는 Sec.6 기본 하이퍼(DEFAULT_*); tuned/optuna_best는 serving JSON."""
    if suffix == 'default':
        return DEFAULT_HIDDEN, DEFAULT_LATENT, DEFAULT_LAYERS, DEFAULT_DROPOUT
    p = dict(serving_cfg['best_params'].get(model_name, {}))
    return (
        int(p.get('hidden', DEFAULT_HIDDEN)),
        int(p.get('latent', DEFAULT_LATENT)),
        int(p.get('layers', DEFAULT_LAYERS)),
        float(p.get('dropout', DEFAULT_DROPOUT)),
    )


def _dims_from_state_dict(model_name, state, serving_cfg, suffix):
    """가중치 텐서 shape에서 아키텍처 추정 (JSON/ suffix와 무관하게 .pt와 일치)."""
    if model_name == 'LSTM-AE' and 'enc.weight_ih_l0' in state:
        hidden = int(state['enc.weight_ih_l0'].shape[0] // 4)
        latent = int(state['fc_e.weight'].shape[0])
        layers = sum(1 for k in state if k.startswith('enc.weight_ih_l'))
        return hidden, latent, layers, DEFAULT_DROPOUT
    if model_name == 'CNN1D-AE' and 'enc.0.weight' in state:
        hidden = int(state['enc.0.weight'].shape[0])
        latent = int(state['fc_e.weight'].shape[0])
        return hidden, latent, DEFAULT_LAYERS, DEFAULT_DROPOUT
    if model_name == 'Transformer-AE' and 'proj.weight' in state:
        hidden = int(state['proj.weight'].shape[0])
        latent = int(state['fc_e.weight'].shape[0])
        n_layers = len({k.split('.')[1] for k in state if k.startswith('tenc.layers.')})
        layers = max(n_layers, DEFAULT_LAYERS)
        return hidden, latent, layers, DEFAULT_DROPOUT
    return _params_for_suffix(model_name, suffix, serving_cfg)


def load_deep_model(checkpoint_dir, model_name, suffix, n_feat, serving_cfg):
    pt = os.path.join(checkpoint_dir, f'{model_name}_{suffix}.pt')
    if not os.path.isfile(pt):
        return None
    try:
        state = torch.load(pt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(pt, map_location=device)
    hidden, latent, layers, dropout = _dims_from_state_dict(
        model_name, state, serving_cfg, suffix
    )
    model = _build_deep_model(model_name, n_feat, hidden, latent, layers, dropout)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def parse_optuna_log(log_path):
    trial_vals, best = [], None
    if not os.path.isfile(log_path):
        return trial_vals, best
    with open(log_path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re.search(r'New best Val F2=([\d.]+)', line)
            if m:
                trial_vals.append(float(m.group(1)))
            m2 = re.search(r'Optuna Best Val F2\s*:\s*([\d.]+)', line)
            if m2:
                best = float(m2.group(1))
    return trial_vals, best


def prepare_variant(variant):
    global OUTPUT_DIR, CHECKPOINT_DIR, MODEL_DIR, VIF_THRESH, CORR_THRESH
    global df_vis, valid_sensors, FINAL_FEATURES, N_FEATURES, vif_before, vif_after
    global X_train_all, y_train_all, X_val_all, y_val_all, X_test, y_test
    global X_test_t, X_val_t, results, BEST_MODEL_NAME, serving_cfg
    global study_best_value, trial_vals, final_errors, final_metrics, y_te_fin

    if variant == 'all':
        OUTPUT_DIR = str(REPO_ROOT / 'outputs/figures/all/')
        CHECKPOINT_DIR = str(REPO_ROOT / 'models/checkpoints_all/')
        VIF_THRESH = 9999.0
        CORR_THRESH = 1.0
    else:
        OUTPUT_DIR = str(REPO_ROOT / 'outputs/figures/vif/')
        CHECKPOINT_DIR = str(REPO_ROOT / 'models/checkpoints_vif/')
        VIF_THRESH = 50.0
        CORR_THRESH = 0.98

    MODEL_DIR = CHECKPOINT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    serving_cfg = load_serving_config(CHECKPOINT_DIR)

    feat_path = os.path.join(CHECKPOINT_DIR, 'final_features.pkl')
    scaler_path = os.path.join(CHECKPOINT_DIR, 'scaler.pkl')
    if not os.path.isfile(feat_path) or not os.path.isfile(scaler_path):
        raise FileNotFoundError(
            f'체크포인트에 scaler.pkl / final_features.pkl 필요: {CHECKPOINT_DIR}'
        )

    with open(feat_path, 'rb') as f:
        FINAL_FEATURES = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    N_FEATURES = len(FINAL_FEATURES)

    missing = df_raw[sensor_cols].isnull().mean().sort_values(ascending=False)
    drop_missing = missing[missing > MISSING_THRESH].index.tolist()
    valid_sensors = [c for c in sensor_cols if c not in drop_missing]
    df_vis = df_raw[valid_sensors + ['machine_status']].copy()

    vif_sample = df_vis[valid_sensors].dropna().sample(min(8000, len(df_vis)), random_state=SEED)
    vif_before = compute_vif(vif_sample)
    selected_by_vif, _ = iterative_vif_drop(vif_sample, VIF_THRESH, verbose=True)
    zero_var = [c for c in selected_by_vif if df_vis[c].std() < 1e-6]
    computed = [c for c in selected_by_vif if c not in zero_var]
    if set(computed) != set(FINAL_FEATURES):
        print(
            f'  [참고] VIF 재계산({len(computed)}) vs 체크포인트({N_FEATURES}) — 체크포인트 피처 사용'
        )
    vif_after = compute_vif(vif_sample[FINAL_FEATURES])

    df_proc = df_raw[FINAL_FEATURES + ['machine_status']].copy().reset_index(drop=True)
    label_map = {'NORMAL': 0, 'BROKEN': 1, 'RECOVERING': 1}
    df_proc['label'] = df_proc['machine_status'].map(label_map).fillna(0).astype(int)
    n = len(df_proc)
    i_tr, i_val = int(n * 0.55), int(n * 0.70)
    train_df = df_proc.iloc[:i_tr].copy()
    val_df = df_proc.iloc[i_tr:i_val].copy()
    test_df = df_proc.iloc[i_val:].copy()

    def _impute(df_part):
        df_part = df_part.copy()
        df_part[FINAL_FEATURES] = df_part[FINAL_FEATURES].ffill().fillna(0.0)
        return df_part

    train_df, val_df, test_df = _impute(train_df), _impute(val_df), _impute(test_df)

    def _safe_scaled(X):
        return np.nan_to_num(
            scaler.transform(X).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )

    train_scaled = _safe_scaled(train_df[FINAL_FEATURES])
    val_scaled = _safe_scaled(val_df[FINAL_FEATURES])
    test_scaled = _safe_scaled(test_df[FINAL_FEATURES])
    labels_train = train_df['label'].values
    labels_val = val_df['label'].values
    labels_test = test_df['label'].values

    win = WINDOW_SIZE
    step = serving_cfg.get('step_size', STEP_SIZE)
    X_train_all, y_train_all = make_windows(train_scaled, labels_train, win, step)
    X_val_all, y_val_all = make_windows(val_scaled, labels_val, win, step)
    X_test, y_test = make_windows(test_scaled, labels_test, win, step)
    X_test_t = torch.FloatTensor(X_test).to(device)
    X_val_t = torch.FloatTensor(X_val_all).to(device)

    results = {}
    for mname in DEEP_MODEL_CLASSES:
        model = load_deep_model(CHECKPOINT_DIR, mname, 'default', N_FEATURES, serving_cfg)
        if model is None:
            print(f'  skip (no .pt): {mname}_default.pt')
            continue
        met = evaluate(get_errors(model, X_test_t), y_test)
        met['val_f2'] = evaluate(get_errors(model, X_val_t), y_val_all)['f2']
        met['val_f1'] = evaluate(get_errors(model, X_val_t), y_val_all)['f1']
        met['val_pr_auc'] = evaluate(get_errors(model, X_val_t), y_val_all)['pr_auc']
        met['train_time'] = 0
        results[mname] = met

    if not results:
        raise FileNotFoundError(f'평가 가능한 *_default.pt 없음: {CHECKPOINT_DIR}')

    BEST_MODEL_NAME = max(results.keys(), key=lambda m: results[m]['val_f2'])

    logs = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.log')]
    log_path = os.path.join(CHECKPOINT_DIR, logs[0]) if logs else ''
    trial_vals, study_best_value = parse_optuna_log(log_path)

    wc = serving_cfg['window_configs'].get(
        BEST_MODEL_NAME, {'window_size': win, 'threshold_pct': THRESHOLD_PCT}
    )
    tuned = load_deep_model(CHECKPOINT_DIR, BEST_MODEL_NAME, 'tuned', N_FEATURES, serving_cfg)
    if tuned is None:
        tuned = load_deep_model(
            CHECKPOINT_DIR, BEST_MODEL_NAME, 'optuna_best', N_FEATURES, serving_cfg
        )
    if tuned is None:
        tuned = load_deep_model(
            CHECKPOINT_DIR, BEST_MODEL_NAME, 'default', N_FEATURES, serving_cfg
        )
    final_errors = get_errors(tuned, X_test_t)
    final_metrics = evaluate(final_errors, y_test, pct=wc.get('threshold_pct', THRESHOLD_PCT))
    y_te_fin = y_test

    print(f'\n=== {variant.upper()} | OUT={OUTPUT_DIR} ===')
    print(f'CKPT={CHECKPOINT_DIR} | features={N_FEATURES} | BEST={BEST_MODEL_NAME}')
