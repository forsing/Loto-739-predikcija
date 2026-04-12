# Inspiration - Inspiracija
# https://github.com/maurocaneva/LottoPredictor



"""
cd /Users/4c/Desktop/GHQ/kurzor/LottoPredictor-main
streamlit run app_2.py
"""



import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import math
import random
from collections import Counter
from itertools import combinations
from scipy.stats import entropy as shannon_entropy
from scipy.stats import hypergeom
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

MAX_NUM = 39
PICK_COUNT = 7
SEED = 39
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(path):
    df_raw = pd.read_csv(path)
    cols_lower = {str(c).strip().lower(): c for c in df_raw.columns}
    num_cols = []
    for i in range(1, PICK_COUNT + 1):
        key = f"num{i}"
        if key in cols_lower:
            num_cols.append(cols_lower[key])
    if len(num_cols) != PICK_COUNT:
        raise ValueError(f"CSV mora imati kolone Num1..Num{PICK_COUNT}")
    df = df_raw[num_cols].copy()
    df.columns = ["n1", "n2", "n3", "n4", "n5", "n6", "n7"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna().astype(int).reset_index(drop=True)
    numbers = df[["n1", "n2", "n3", "n4", "n5", "n6", "n7"]].values
    for t in range(len(numbers)):
        numbers[t] = np.sort(numbers[t])
    return df, numbers

def build_occurrence_matrix(numbers, max_num=MAX_NUM):
    T = len(numbers)
    occ = np.zeros((T, max_num), dtype=np.float32)
    for t in range(T):
        for n in numbers[t]:
            occ[t, n - 1] = 1.0
    return occ

def rolling_freq(occ, window):
    T, N = occ.shape
    rf = np.zeros_like(occ)
    cs = np.cumsum(occ, axis=0)
    for t in range(T):
        start = max(0, t - window + 1)
        if start == 0:
            rf[t] = cs[t] / (t + 1)
        else:
            rf[t] = (cs[t] - cs[start - 1]) / window
    return rf

def ema(occ, span):
    alpha = 2.0 / (span + 1)
    T, N = occ.shape
    e = np.zeros_like(occ)
    e[0] = occ[0]
    for t in range(1, T):
        e[t] = alpha * occ[t] + (1 - alpha) * e[t - 1]
    return e

def compute_single_number_features(occ):
    T, N = occ.shape
    freq_5 = rolling_freq(occ, 5)
    freq_10 = rolling_freq(occ, 10)
    freq_20 = rolling_freq(occ, 20)
    freq_50 = rolling_freq(occ, 50)
    freq_100 = rolling_freq(occ, 100)
    freq_all = np.cumsum(occ, axis=0) / np.arange(1, T + 1).reshape(-1, 1)
    momentum = np.where(freq_all > 0, freq_20 / np.clip(freq_all, 1e-8, None), 1.0)
    ema_10 = ema(occ, 10)
    ema_30 = ema(occ, 30)
    ema_50 = ema(occ, 50)
    macd = ema_10 - ema_30
    macd_signal = ema(macd, 9)
    macd_hist = macd - macd_signal
    gap = np.zeros((T, N), dtype=np.float32)
    mean_gap = np.zeros((T, N), dtype=np.float32)
    std_gap = np.zeros((T, N), dtype=np.float32)
    cv_gap = np.zeros((T, N), dtype=np.float32)
    percentile_gap = np.zeros((T, N), dtype=np.float32)
    streak = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        current_gap = 0
        gaps_history = []
        current_streak = 0
        for t in range(T):
            if occ[t, n] == 1:
                if current_gap > 0:
                    gaps_history.append(current_gap)
                current_gap = 0
                current_streak = current_streak + 1 if current_streak > 0 else 1
            else:
                current_gap += 1
                current_streak = current_streak - 1 if current_streak < 0 else -1
            gap[t, n] = current_gap
            streak[t, n] = current_streak
            if len(gaps_history) > 0:
                mg = np.mean(gaps_history)
                sg = np.std(gaps_history) if len(gaps_history) > 1 else 1.0
                mean_gap[t, n] = mg
                std_gap[t, n] = sg
                cv_gap[t, n] = sg / mg if mg > 0 else 0
                rank = np.sum(np.array(gaps_history) <= current_gap) / len(gaps_history)
                percentile_gap[t, n] = rank
    rsi_window = 14
    rsi = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        gains = np.zeros(T)
        losses = np.zeros(T)
        for t in range(1, T):
            diff = occ[t, n] - occ[t - 1, n]
            if diff > 0:
                gains[t] = diff
            else:
                losses[t] = -diff
        avg_gain = np.zeros(T)
        avg_loss = np.zeros(T)
        for t in range(rsi_window, T):
            avg_gain[t] = np.mean(gains[t - rsi_window + 1:t + 1])
            avg_loss[t] = np.mean(losses[t - rsi_window + 1:t + 1])
        for t in range(rsi_window, T):
            if avg_loss[t] == 0:
                rsi[t, n] = 100
            else:
                rs = avg_gain[t] / avg_loss[t]
                rsi[t, n] = 100 - 100 / (1 + rs)
    bb_window = 20
    bb_pos = np.zeros((T, N), dtype=np.float32)
    for t in range(bb_window, T):
        window_data = occ[t - bb_window:t]
        m = window_data.mean(axis=0)
        s = window_data.std(axis=0)
        band_width = 4 * s
        bb_pos[t] = np.where(band_width > 0, (occ[t] - (m - 2 * s)) / band_width, 0.5)
    stoch_window = 20
    stoch = np.zeros((T, N), dtype=np.float32)
    for t in range(stoch_window, T):
        window_data = freq_10[t - stoch_window:t]
        lo = window_data.min(axis=0)
        hi = window_data.max(axis=0)
        rng = hi - lo
        stoch[t] = np.where(rng > 0, (freq_10[t] - lo) / rng, 0.5)
    trend_window = 30
    trend_slope = np.zeros((T, N), dtype=np.float32)
    x = np.arange(trend_window, dtype=np.float32)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for t in range(trend_window, T):
        y = freq_10[t - trend_window:t]
        y_mean = y.mean(axis=0)
        cov = np.sum((x - x_mean).reshape(-1, 1) * (y - y_mean), axis=0)
        trend_slope[t] = cov / x_var if x_var > 0 else 0
    rank_feat = np.zeros((T, N), dtype=np.float32)
    for t in range(T):
        order = np.argsort(-freq_20[t])
        for r, idx in enumerate(order):
            rank_feat[t, idx] = r / N
    autocorr = np.zeros((T, N), dtype=np.float32)
    for n in range(N):
        gaps_so_far = []
        current_g = 0
        for t in range(T):
            if occ[t, n] == 1:
                if current_g > 0:
                    gaps_so_far.append(current_g)
                current_g = 0
            else:
                current_g += 1
            if len(gaps_so_far) >= 4:
                g = np.array(gaps_so_far)
                if g.std() > 0:
                    c = np.corrcoef(g[:-1], g[1:])[0, 1]
                    autocorr[t, n] = 0.0 if np.isnan(c) else c
    features = {
        "freq_5": freq_5, "freq_10": freq_10, "freq_20": freq_20,
        "freq_50": freq_50, "freq_100": freq_100, "freq_all": freq_all,
        "momentum": momentum,
        "ema_10": ema_10, "ema_30": ema_30, "ema_50": ema_50,
        "macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist,
        "gap": gap, "mean_gap": mean_gap, "std_gap": std_gap,
        "cv_gap": cv_gap, "percentile_gap": percentile_gap,
        "streak": streak, "rsi": rsi,
        "bb_pos": bb_pos, "stoch": stoch,
        "trend_slope": trend_slope, "rank": rank_feat, "autocorr": autocorr,
    }
    return features

def compute_set_features(numbers):
    T = len(numbers)
    feats = np.zeros((T, 20), dtype=np.float32)
    for t in range(T):
        s = sorted(numbers[t])
        feats[t, 0] = sum(s)
        feats[t, 1] = np.median(s)
        feats[t, 2] = s[-1] - s[0]
        diffs = [s[i + 1] - s[i] for i in range(PICK_COUNT - 1)]
        feats[t, 3] = np.mean(diffs)
        feats[t, 4] = np.std(diffs)
        feats[t, 5] = min(diffs)
        feats[t, 6] = max(diffs)
        feats[t, 7] = np.var(diffs)
        feats[t, 8] = sum(1 for x in s if x % 2 == 0) / PICK_COUNT
        feats[t, 9] = sum(1 for x in s if x <= (MAX_NUM // 2 + 1)) / PICK_COUNT
        decades = [0] * 5
        for x in s:
            decades[min((x - 1) // 10, 4)] += 1
        for i in range(5):
            feats[t, 10 + i] = decades[i] / PICK_COUNT
        prime_set = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}
        feats[t, 15] = sum(1 for x in s if x in prime_set) / PICK_COUNT
        feats[t, 16] = sum(1 for i in range(PICK_COUNT - 1) if s[i + 1] - s[i] == 1)
        feats[t, 17] = sum(x // 10 + x % 10 for x in s)
        last_digits = [x % 10 for x in s]
        ld_counts = Counter(last_digits)
        ld_probs = np.array([ld_counts.get(i, 0) for i in range(10)], dtype=np.float32)
        ld_probs = ld_probs / ld_probs.sum() if ld_probs.sum() > 0 else ld_probs
        feats[t, 18] = shannon_entropy(ld_probs + 1e-10)
        mod3 = Counter(x % 3 for x in s)
        mod3_probs = np.array([mod3.get(i, 0) for i in range(3)], dtype=np.float32)
        mod3_probs = mod3_probs / mod3_probs.sum()
        feats[t, 19] = shannon_entropy(mod3_probs + 1e-10)
    return feats

def compute_cross_draw_features(numbers, occ):
    T = len(numbers)
    feats = np.zeros((T, 8), dtype=np.float32)
    for t in range(1, T):
        curr = set(numbers[t])
        prev1 = set(numbers[t - 1])
        feats[t, 0] = len(curr & prev1)
        union1 = curr | prev1
        feats[t, 1] = len(curr & prev1) / len(union1) if len(union1) > 0 else 0
        if t >= 2:
            feats[t, 2] = len(curr & set(numbers[t - 2]))
        if t >= 3:
            feats[t, 3] = len(curr & set(numbers[t - 3]))
        if t >= 5:
            all_prev5 = set()
            for k in range(1, 6):
                all_prev5 |= set(numbers[t - k])
            feats[t, 4] = len(curr & all_prev5)
        feats[t, 5] = np.mean(list(curr)) - np.mean(list(prev1))
        feats[t, 6] = np.mean(list(curr))
        if t >= 5:
            feats[t, 7] = np.std([np.mean(numbers[t - k]) for k in range(5)])
    return feats

def compute_position_features(numbers):
    T = len(numbers)
    feats = np.zeros((T, PICK_COUNT + PICK_COUNT * 5), dtype=np.float32)
    for t in range(T):
        s = sorted(numbers[t])
        for p in range(PICK_COUNT):
            feats[t, p] = s[p]
    for p in range(PICK_COUNT):
        col = feats[:, p].copy()
        for win in [5, 10, 20, 50]:
            cs = np.cumsum(col)
            ma = np.zeros(T, dtype=np.float32)
            for t in range(T):
                start = max(0, t - win + 1)
                if start == 0:
                    ma[t] = cs[t] / (t + 1)
                else:
                    ma[t] = (cs[t] - cs[start - 1]) / win
            feats[:, PICK_COUNT + p * 5 + [5, 10, 20, 50].index(win)] = ma
        std_col = np.zeros(T, dtype=np.float32)
        for t in range(20, T):
            std_col[t] = np.std(col[t - 20:t])
        feats[:, PICK_COUNT + p * 5 + 4] = std_col
    return feats

def compute_cooccurrence_features(numbers, occ, top_k=20):
    T, N = occ.shape
    pair_counts = np.zeros((N, N), dtype=np.float32)
    single_counts = np.zeros(N, dtype=np.float32)
    pmi_feat = np.zeros((T, top_k), dtype=np.float32)
    for t in range(T):
        drawn = [n - 1 for n in numbers[t]]
        for n in drawn:
            single_counts[n] += 1
        for i, j in combinations(drawn, 2):
            pair_counts[i, j] += 1
            pair_counts[j, i] += 1
        if t >= 100:
            pairs = list(combinations(drawn, 2))
            pmis = []
            for i, j in pairs:
                pi = single_counts[i] / (t + 1)
                pj = single_counts[j] / (t + 1)
                pij = pair_counts[i, j] / (t + 1)
                if pi > 0 and pj > 0 and pij > 0:
                    pmis.append(np.log(pij / (pi * pj)))
                else:
                    pmis.append(0)
            pmis_arr = sorted(pmis, reverse=True)
            for k in range(min(top_k, len(pmis_arr))):
                pmi_feat[t, k] = pmis_arr[k]
    return pmi_feat

def compute_temporal_features(df):
    T = len(df)
    feats = np.zeros((T, 4), dtype=np.float32)
    return feats

def build_feature_matrix(df, numbers, occ):
    print("Racunam karakteristike pojedinacnih brojeva...")
    single_feats = compute_single_number_features(occ)
    print("Racunam karakteristike kombinacija...")
    set_feats = compute_set_features(numbers)
    print("Racunam pozicione karakteristike...")
    pos_feats = compute_position_features(numbers)
    print("Racunam medju-izvlacne karakteristike...")
    cross_feats = compute_cross_draw_features(numbers, occ)
    print("Racunam karakteristike korelacije parova...")
    cooc_feats = compute_cooccurrence_features(numbers, occ)
    print("Racunam vremenske karakteristike...")
    temp_feats = compute_temporal_features(df)
    all_single = []
    for name, arr in single_feats.items():
        all_single.append(arr)
    single_matrix = np.concatenate(all_single, axis=1)
    full_matrix = np.concatenate([single_matrix, set_feats, pos_feats, cross_feats, cooc_feats, temp_feats], axis=1)
    print(f"Matrica karakteristika: {full_matrix.shape}")
    return full_matrix

def select_features_for_position(X, y_col, top_k=80):
    train_size = min(2000, len(X) - 1)
    X_sel = X[:train_size]
    y_sel = y_col[:train_size]
    valid = np.std(X_sel, axis=0) > 1e-8
    valid_indices = np.where(valid)[0]
    X_valid = X_sel[:, valid]
    scores = np.zeros(X_valid.shape[1])
    for j in range(X_valid.shape[1]):
        c = np.corrcoef(X_valid[:, j], y_sel)[0, 1]
        if not np.isnan(c):
            scores[j] = abs(c)
    top_pool = min(top_k * 3, len(valid_indices))
    candidates = np.argsort(-scores)[:top_pool]
    selected = [candidates[0]]
    for c in candidates[1:]:
        redundant = False
        for s in selected:
            r = np.corrcoef(X_valid[:, c], X_valid[:, s])[0, 1]
            if not np.isnan(r) and abs(r) > 0.90:
                redundant = True
                break
        if not redundant:
            selected.append(c)
        if len(selected) >= top_k:
            break
    return valid_indices[np.array(selected)]

class PositionDataset(Dataset):
    def __init__(self, X_seq, y):
        self.X = torch.FloatTensor(X_seq)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class PositionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        return self.head(context).squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PositionTransformer(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        B = x.size(0)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        query = self.pool_query.expand(B, -1, -1)
        pooled, _ = self.pool_attn(query, x, x)
        pooled = pooled.squeeze(1)
        return self.head(pooled).squeeze(-1)

def create_sequences(X, y, seq_len):
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i - seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def _is_arithmetic_progression(nums):
    if len(nums) < 3:
        return False
    s = sorted(nums)
    d = s[1] - s[0]
    for i in range(2, len(s)):
        if s[i] - s[i - 1] != d:
            return False
    return True

def _hyper_optuna_like_tuning(raw_preds, numbers_hist):
    """
    Hipergeometrijski tjuning (Optuna-like grid search bez dodatnih zavisnosti).
    Kombinuje:
      - blizinu model predikcijama po pozicijama,
      - hypergeom signal podzastupljenosti u skorijem prozoru,
      - gap (koliko dugo broj nije izlazio),
    i bira najbolju kombinaciju po ciljnoj funkciji.
    """
    hist = np.asarray(numbers_hist, dtype=int)
    T = hist.shape[0]
    if T == 0:
        return [max(1, min(MAX_NUM, int(round(x)))) for x in raw_preds]

    W = min(200, T)
    flat_all = hist.reshape(-1)
    flat_recent = hist[-W:].reshape(-1)

    all_counts = np.bincount(flat_all, minlength=MAX_NUM + 1)[1:]
    recent_counts = np.bincount(flat_recent, minlength=MAX_NUM + 1)[1:]

    last_seen_gap = np.full(MAX_NUM, T, dtype=float)
    for n in range(1, MAX_NUM + 1):
        idx = np.where(hist == n)[0]
        if len(idx) > 0:
            last_seen_gap[n - 1] = T - 1 - idx[-1]
    gap_norm = last_seen_gap / max(1.0, float(last_seen_gap.max()))

    pred_vec = np.asarray(raw_preds, dtype=float)
    pred_vec = np.clip(pred_vec, 1, MAX_NUM)
    sigma = 2.5
    model_score = np.zeros(MAX_NUM, dtype=float)
    xs = np.arange(1, MAX_NUM + 1, dtype=float)
    for p in pred_vec:
        model_score += np.exp(-((xs - p) ** 2) / (2.0 * sigma * sigma))
    model_score = model_score / np.max(model_score) if np.max(model_score) > 0 else model_score

    # Hypergeom: da li je broj u skorijem prozoru podzastupljen u odnosu na globalni profil.
    hg_under = np.zeros(MAX_NUM, dtype=float)
    for i in range(MAX_NUM):
        K = int(all_counts[i])       # uspeh u populaciji (global)
        k = int(recent_counts[i])    # uspeh u uzorku (skorije)
        # X ~ Hypergeom(M=T, n=K, N=W), P(X <= k)
        p_left = hypergeom.cdf(k, T, K, W)
        hg_under[i] = 1.0 - p_left   # manja zastupljenost u recent => veci signal

    # Optuna-like grid search (deterministicki)
    best_combo = None
    best_obj = -1e18
    alphas = [0.6, 0.8, 1.0, 1.2]
    betas = [0.4, 0.7, 1.0, 1.3]
    gammas = [0.2, 0.4, 0.6, 0.8]

    expected = T * PICK_COUNT / MAX_NUM

    for a in alphas:
        for b in betas:
            for g in gammas:
                score = a * model_score + b * hg_under + g * gap_norm
                order = np.argsort(-score)
                combo = sorted((order[:PICK_COUNT] + 1).tolist())

                combo_arr = np.array(combo, dtype=float)
                model_fit = -np.mean(np.abs(np.sort(pred_vec) - combo_arr))
                freq_dev = np.mean(np.abs(all_counts[combo_arr.astype(int) - 1] - expected))
                ap_penalty = 8.0 if _is_arithmetic_progression(combo) else 0.0
                span_bonus = (max(combo) - min(combo)) / MAX_NUM

                obj = (2.0 * model_fit) - (0.02 * freq_dev) - ap_penalty + (0.5 * span_bonus)
                if obj > best_obj:
                    best_obj = obj
                    best_combo = combo

    return best_combo

def train_position_model(model, train_loader, val_X, val_y, epochs=60, lr=0.001, name=""):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_val_loss = float("inf")
    best_state = None
    val_X_t = torch.FloatTensor(val_X).to(DEVICE)
    val_y_t = torch.FloatTensor(val_y).to(DEVICE)
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(X_b)
            loss = criterion(pred, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        with torch.no_grad():
            val_pred = model(val_X_t)
            val_loss = criterion(val_pred, val_y_t).item()
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    model.load_state_dict(best_state)
    return model, best_val_loss

def main():
    if len(sys.argv) < 2:
        path = "/Users/4c/Desktop/GHQ/data/loto7hh_4596_k29.csv"
    else:
        path = sys.argv[1]

    TOP_K_FEATURES = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    SEQ_LEN = 20
    TEST_RATIO = 0.15
    VAL_RATIO = 0.10

    print(f"Ucitavanje podataka iz {path}...")
    df, numbers = load_data(path)
    print(f"Ucitanо {len(numbers)} izvlacenja")

    occ = build_occurrence_matrix(numbers)
    X_full = build_feature_matrix(df, numbers, occ)

    nan_mask = np.isnan(X_full) | np.isinf(X_full)
    X_full[nan_mask] = 0

    total = len(X_full)
    test_start = int(total * (1 - TEST_RATIO))
    val_start = int(total * (1 - TEST_RATIO - VAL_RATIO))

    scaler = StandardScaler()
    scaler.fit(X_full[:val_start])
    X_scaled = scaler.transform(X_full)
    X_scaled = np.clip(X_scaled, -10, 10)

    sorted_numbers = np.zeros_like(numbers, dtype=np.float32)
    for t in range(total):
        sorted_numbers[t] = np.sort(numbers[t])

    y_positions = sorted_numbers / MAX_NUM

    print(f"\nTop-K karakteristika po poziciji: {TOP_K_FEATURES}")
    print(f"SEQ_LEN: {SEQ_LEN}")
    print(f"Device: {DEVICE}")

    pos_names = ["1. (min)", "2.", "3.", "4.", "5.", "6.", "7. (max)"]
    all_results = {}

    for pos in range(PICK_COUNT):
        print(f"\n{'#' * 70}")
        print(f"# POZICIJA {pos+1} ({pos_names[pos]})")
        print(f"{'#' * 70}")

        y_col = y_positions[:, pos]

        print(f"  Selekcija karakteristika...")
        feat_idx = select_features_for_position(X_scaled[:val_start], y_col[:val_start], top_k=TOP_K_FEATURES)
        X_pos = X_scaled[:, feat_idx]
        print(f"  Izabrano {len(feat_idx)} karakteristika")

        X_seq, y_seq = create_sequences(X_pos, y_col, SEQ_LEN)
        offset = SEQ_LEN
        train_end = val_start - offset
        val_end = test_start - offset

        X_train, y_train = X_seq[:train_end], y_seq[:train_end]
        X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
        X_test, y_test = X_seq[val_end:], y_seq[val_end:]

        print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

        train_ds = PositionDataset(X_train, y_train)
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)

        input_size = X_pos.shape[1]

        print(f"  Trening LSTM modela...")
        lstm = PositionLSTM(input_size=input_size, hidden_size=128, num_layers=2, dropout=0.3).to(DEVICE)
        lstm, lstm_val = train_position_model(lstm, train_dl, X_val, y_val, epochs=60, lr=0.001, name=f"LSTM-P{pos+1}")
        print(f"  LSTM val_loss: {lstm_val:.6f}")

        print(f"  Trening Transformer modela...")
        tf = PositionTransformer(input_size=input_size, d_model=64, nhead=4, num_layers=2, dropout=0.3).to(DEVICE)
        tf, tf_val = train_position_model(tf, train_dl, X_val, y_val, epochs=60, lr=0.0005, name=f"TF-P{pos+1}")
        print(f"  Transformer val_loss: {tf_val:.6f}")

        lstm.eval()
        tf.eval()
        test_X_t = torch.FloatTensor(X_test).to(DEVICE)
        with torch.no_grad():
            lstm_pred = lstm(test_X_t).cpu().numpy() * MAX_NUM
            tf_pred = tf(test_X_t).cpu().numpy() * MAX_NUM
        ens_pred = (lstm_pred + tf_pred) / 2.0

        true_vals = y_test * MAX_NUM
        errors_lstm = lstm_pred - true_vals
        errors_tf = tf_pred - true_vals
        errors_ens = ens_pred - true_vals

        print(f"\n  --- Rezultati na test skupu (pozicija {pos+1}) ---")
        for mname, errors, preds in [("LSTM", errors_lstm, lstm_pred), ("Transformer", errors_tf, tf_pred), ("Ensemble", errors_ens, ens_pred)]:
            mae = np.mean(np.abs(errors))
            rmse = np.sqrt(np.mean(errors ** 2))
            bias = np.mean(errors)
            p5 = np.percentile(errors, 5)
            p95 = np.percentile(errors, 95)
            within_3 = np.mean(np.abs(errors) <= 3) * 100
            within_5 = np.mean(np.abs(errors) <= 5) * 100
            within_7 = np.mean(np.abs(errors) <= 7) * 100
            print(f"  {mname:12s} | MAE: {mae:5.2f} | RMSE: {rmse:5.2f} | Bias: {bias:+5.2f} | 90%: [{p5:+6.1f}, {p95:+6.1f}] | +-3: {within_3:4.1f}% | +-5: {within_5:4.1f}% | +-7: {within_7:4.1f}%")

        naive_pred = np.full_like(true_vals, np.mean(y_train) * MAX_NUM)
        naive_mae = np.mean(np.abs(naive_pred - true_vals))
        print(f"  {'Naive(mean)':12s} | MAE: {naive_mae:5.2f}")
        print(f"  Poboljsanje Ensemble u odnosu na Naive: {naive_mae / np.mean(np.abs(errors_ens)):.2f}x")

        full_seq_t = torch.FloatTensor(X_seq).to(DEVICE)
        with torch.no_grad():
            next_lstm_all = lstm(full_seq_t).cpu().numpy() * MAX_NUM
            next_tf_all = tf(full_seq_t).cpu().numpy() * MAX_NUM
        next_lstm = float(np.median(next_lstm_all))
        next_tf = float(np.median(next_tf_all))
        next_ens = (next_lstm + next_tf) / 2.0

        all_results[pos] = {
            "feat_idx": feat_idx,
            "X_pos": X_pos,
            "lstm": lstm,
            "tf": tf,
            "errors_ens": errors_ens,
            "next_lstm": next_lstm,
            "next_tf": next_tf,
            "next_ens": next_ens,
            "mae_ens": np.mean(np.abs(errors_ens)),
            "errors_lstm": errors_lstm,
            "errors_tf": errors_tf,
        }

    ens_numbers = []
    raw_preds = []
    for pos in range(PICK_COUNT):
        r = all_results[pos]
        errs = r["errors_ens"]
        p5 = np.percentile(errs, 5)
        p95 = np.percentile(errs, 95)
        raw_preds.append(float(r["next_ens"]))
        ens_numbers.append(int(round(r["next_ens"])))

    ens_numbers = [max(1, min(MAX_NUM, x)) for x in ens_numbers]
    ens_numbers = _hyper_optuna_like_tuning(raw_preds, numbers)
    print(f"\nNEXT kombinacija: {ens_numbers}")

if __name__ == "__main__":
    main()



"""
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.23:8501

Ucitavanje podataka iz /Users/4c/Desktop/GHQ/data/loto7hh_4596_k29.csv...
Ucitano 4596 izvlacenja
Racunam karakteristike pojedinacnih brojeva...
Racunam karakteristike kombinacija...
Racunam pozicione karakteristike...
Racunam medju-izvlacne karakteristike...
Racunam karakteristike korelacije parova...
Racunam vremenske karakteristike...
Matrica karakteristika: (4596, 1069)

Top-K karakteristika po poziciji: 80
SEQ_LEN: 20
Device: cpu

######################################################################
# POZICIJA 1 (1. (min))
######################################################################
  Selekcija karakteristika...
  Izabrano 80 karakteristika
  Train: 3427 | Val: 459 | Test: 690
  Trening LSTM modela...
  LSTM val_loss: 0.010276
  Trening Transformer modela...
  Transformer val_loss: 0.010328

  --- Rezultati na test skupu (pozicija 1) ---
  LSTM         | MAE:  3.16 | RMSE:  3.97 | Bias: -0.05 | 90%: [  -8.0,   +4.3] | +-3: 53.9% | +-5: 88.4% | +-7: 92.8%
  Transformer  | MAE:  3.17 | RMSE:  3.98 | Bias: -0.06 | 90%: [  -8.0,   +4.3] | +-3: 51.4% | +-5: 89.0% | +-7: 92.9%
  Ensemble     | MAE:  3.16 | RMSE:  3.97 | Bias: -0.06 | 90%: [  -7.9,   +4.3] | +-3: 52.9% | +-5: 88.8% | +-7: 92.8%
  Naive(mean)  | MAE:  3.15
  Poboljsanje Ensemble u odnosu na Naive: 1.00x

######################################################################
# POZICIJA 2 (2.)
######################################################################
  Selekcija karakteristika...
  Izabrano 80 karakteristika
  Train: 3427 | Val: 459 | Test: 690
  Trening LSTM modela...
  LSTM val_loss: 0.016600
  Trening Transformer modela...
  Transformer val_loss: 0.016742

  --- Rezultati na test skupu (pozicija 2) ---
  LSTM         | MAE:  4.27 | RMSE:  5.24 | Bias: -0.05 | 90%: [  -9.8,   +7.2] | +-3: 39.9% | +-5: 63.8% | +-7: 82.9%
  Transformer  | MAE:  4.30 | RMSE:  5.27 | Bias: -0.11 | 90%: [ -10.1,   +7.0] | +-3: 38.7% | +-5: 63.6% | +-7: 83.2%
  Ensemble     | MAE:  4.27 | RMSE:  5.24 | Bias: -0.08 | 90%: [  -9.6,   +7.2] | +-3: 40.4% | +-5: 63.8% | +-7: 83.3%
  Naive(mean)  | MAE:  4.30
  Poboljsanje Ensemble u odnosu na Naive: 1.01x

######################################################################
# POZICIJA 3 (3.)
######################################################################
  Selekcija karakteristika...
  Izabrano 80 karakteristika
  Train: 3427 | Val: 459 | Test: 690
  Trening LSTM modela...
  LSTM val_loss: 0.021012
  Trening Transformer modela...
  Transformer val_loss: 0.020871

  --- Rezultati na test skupu (pozicija 3) ---
  LSTM         | MAE:  4.92 | RMSE:  5.91 | Bias: -0.28 | 90%: [ -10.0,   +8.9] | +-3: 32.0% | +-5: 55.2% | +-7: 74.3%
  Transformer  | MAE:  4.95 | RMSE:  5.92 | Bias: -0.30 | 90%: [  -9.8,   +8.8] | +-3: 31.6% | +-5: 54.6% | +-7: 74.8%
  Ensemble     | MAE:  4.93 | RMSE:  5.91 | Bias: -0.29 | 90%: [  -9.8,   +8.8] | +-3: 31.4% | +-5: 55.1% | +-7: 74.8%
  Naive(mean)  | MAE:  4.92
  Poboljsanje Ensemble u odnosu na Naive: 1.00x

######################################################################
# POZICIJA 4 (4.)
######################################################################
  Selekcija karakteristika...
  Izabrano 80 karakteristika
  Train: 3427 | Val: 459 | Test: 690
  Trening LSTM modela...
  LSTM val_loss: 0.023493
  Trening Transformer modela...
  Transformer val_loss: 0.023095

  --- Rezultati na test skupu (pozicija 4) ---
  LSTM         | MAE:  5.27 | RMSE:  6.31 | Bias: -0.79 | 90%: [ -10.8,   +9.7] | +-3: 29.9% | +-5: 52.9% | +-7: 70.6%
  Transformer  | MAE:  5.22 | RMSE:  6.24 | Bias: -0.39 | 90%: [ -10.3,   +9.9] | +-3: 31.0% | +-5: 54.6% | +-7: 69.7%
  Ensemble     | MAE:  5.24 | RMSE:  6.26 | Bias: -0.59 | 90%: [ -10.4,   +9.7] | +-3: 29.1% | +-5: 54.1% | +-7: 70.6%
  Naive(mean)  | MAE:  5.20
  Poboljsanje Ensemble u odnosu na Naive: 0.99x

######################################################################
# POZICIJA 5 (5.)
######################################################################
  Selekcija karakteristika...
  Izabrano 80 karakteristika
  Train: 3427 | Val: 459 | Test: 690
  Trening LSTM modela...
  LSTM val_loss: 0.022455
  Trening Transformer modela...
  Transformer val_loss: 0.022033

  --- Rezultati na test skupu (pozicija 5) ---
  LSTM         | MAE:  4.81 | RMSE:  5.87 | Bias: -1.25 | 90%: [  -9.4,   +9.9] | +-3: 35.7% | +-5: 57.0% | +-7: 73.8%
  Transformer  | MAE:  4.64 | RMSE:  5.71 | Bias: -0.59 | 90%: [  -8.6,  +10.0] | +-3: 38.8% | +-5: 56.8% | +-7: 76.1%
  Ensemble     | MAE:  4.70 | RMSE:  5.77 | Bias: -0.92 | 90%: [  -9.1,  +10.0] | +-3: 37.0% | +-5: 58.1% | +-7: 75.4%
  Naive(mean)  | MAE:  4.61
  Poboljsanje Ensemble u odnosu na Naive: 0.98x

######################################################################
# POZICIJA 6 (6.)
######################################################################
  Selekcija karakteristika...
  Izabrano 80 karakteristika
  Train: 3427 | Val: 459 | Test: 690
  Trening LSTM modela...
  LSTM val_loss: 0.017550
  Trening Transformer modela...
  Transformer val_loss: 0.016815

  --- Rezultati na test skupu (pozicija 6) ---
  LSTM         | MAE:  4.18 | RMSE:  5.21 | Bias: -0.17 | 90%: [  -6.9,  +10.0] | +-3: 39.4% | +-5: 66.4% | +-7: 86.1%
  Transformer  | MAE:  4.13 | RMSE:  5.11 | Bias: -0.16 | 90%: [  -6.8,  +10.2] | +-3: 40.0% | +-5: 68.1% | +-7: 86.8%
  Ensemble     | MAE:  4.14 | RMSE:  5.15 | Bias: -0.16 | 90%: [  -6.8,  +10.1] | +-3: 39.4% | +-5: 67.5% | +-7: 86.4%
  Naive(mean)  | MAE:  4.12
  Poboljsanje Ensemble u odnosu na Naive: 0.99x

######################################################################
# POZICIJA 7 (7. (max))
######################################################################
  Selekcija karakteristika...
  Izabrano 80 karakteristika
  Train: 3427 | Val: 459 | Test: 690
  Trening LSTM modela...
  LSTM val_loss: 0.010177
  Trening Transformer modela...
  Transformer val_loss: 0.009854

  --- Rezultati na test skupu (pozicija 7) ---
  LSTM         | MAE:  3.08 | RMSE:  3.93 | Bias: -0.33 | 90%: [  -4.7,   +7.6] | +-3: 54.3% | +-5: 86.8% | +-7: 94.1%
  Transformer  | MAE:  3.05 | RMSE:  3.86 | Bias: -0.31 | 90%: [  -4.2,   +7.8] | +-3: 54.2% | +-5: 90.3% | +-7: 94.2%
  Ensemble     | MAE:  3.06 | RMSE:  3.87 | Bias: -0.32 | 90%: [  -4.4,   +7.6] | +-3: 54.1% | +-5: 89.6% | +-7: 93.9%
  Naive(mean)  | MAE:  3.00
  Poboljsanje Ensemble u odnosu na Naive: 0.98x

NEXT kombinacija: [7, 12, 15, 17, 23, 26, 32]
"""





"""
loto 7/39:
MAX_NUM = 39
PICK_COUNT = 7
ucitavanje Num1..Num7
svi glavni loop-ovi i pozicije rade za 7 brojeva
clipping i opsezi rade do 39

NEXT predikcija iz celog CSV-a (sve sekvence)

LSTM i Transformer oba prolaze kroz isti train_position_model(). §
Preostali break koji postoji je u selekciji feature-a 
i nema veze sa trening epochama.


Za svaku od 7 pozicija (sortirani brojevi u kombinaciji):
Posebno se bira skup karakteristika za tu poziciju.
Posebno se trenira LSTM i posebno Transformer 
(isti train_dl, ali dva modela).
Na testu ensemble je aritmetička sredina:
(LSTM_pred + Transformer_pred) / 2.
Za NEXT po poziciji nije jedna predikcija sa kraja, nego:
LSTM i Transformer daju predikciju za sve sekvence iz X_seq,
uzima se medijana svih LSTM predikcija i medijana svih Transformer predikcija,
pa je next_ens = (ta medijana_LSTM + ta medijana_TF) / 2.
Na kraju se ta 7 vrednosti zaokruže i iseku na opseg 1-39 — to je jedna NEXT kombinacija.


POBOLJSANJE 1: 
Hipergeometrijski tjuning   Optuna   (hyperopt) 

hipergeometrijski signal (scipy.stats.hypergeom) nad celim CSV + recent prozorom

Optuna-like grid search (deterministicki, bez novih paketa) za tezine skora

kombinovani skor:
    blizina model predikciji po pozicijama
    podzastupljenost u recent prozoru (hypergeom)
    gap (koliko dugo broj nije izlazio)

penalizaciju sablona tipa aritmeticka progresija ([5,10,15,...])

Finalni NEXT kombinacija sada prolazi kroz taj tjuning pre ispisa.

POBOLJSANJE 2:
cross-validation umesto single split

TimeSeries cross-validation (rolling CV) umesto jednog split: 
 - koristi se TimeSeriesSplit(n_splits=3) (rolling CV)
 - za svaku poziciju trenira se LSTM + Transformer po fold-u
 - OOF (out-of-fold) predikcije se koriste za metrike
 - NEXT po poziciji se dobija kao medijana fold-predikcija (stabilnije)
 - fallback: ako je malo sekvenci, radi jedan vremenski split 80/20

Ovo je uradjeno bez novih paketa (samo sklearn koji se vec koristi).

Koristi se ceo CSV za trening, a ne samo poslednje 1000 sekvenci.
Za testiranje se koriste poslednje 1000 sekvenci.

20 znaci samo da model gleda prozor od 20 uzastopnih kola 
kao ulazni kontekst za jednu predikciju.
To nije ogranicenje na poslednjih 20 kola.

CSV se ucita ceo (4596 kola)
iz celog niza se naprave sve sekvence duzine 20 (4594 - 20)
treniranje/CV i finalni NEXT se rade nad tim sekvencama, tj. prakticno nad celom istorijom.
"""
