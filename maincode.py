# streamlit_app.py
# Toto Predictor — Enhanced with LSTM, Global Filters, Pages
# Single-file Streamlit app (copy-paste)
# Added: Bonus support for Star 6/50 (auto-detected). Bonus parsed from "BonusNo" column.
# - Analysis: bonus frequency chart
# - Prediction: all strategies produce a bonus pick (single number) for 6/50
# - Simulation: backtests include bonus checks for 6/50
# No changes to UI/controls/6/55/6/58 logic beyond bonus addition for 6/50.

import streamlit as st
import pandas as pd
import numpy as np
import random
import itertools
import io
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# Optional TensorFlow (LSTM). Wrapped in try/except.
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Toto Predictor", layout="wide")

# ---------- Config ----------
GSHEET_URLS = {
    "Star 6/50": "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8b63hsr-rj0VCeviGFy36wVCEGRvKlUY98RSUMZtPyC6s9NxiUwuzA_ZvJ4pcujHoSpFhBCH4m2pJ/pub?gid=1694692461&single=true&output=csv",
    "Power 6/55": "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8b63hsr-rj0VCeviGFy36wVCEGRvKlUY98RSUMZtPyC6s9NxiUwuzA_ZvJ4pcujHoSpFhBCH4m2pJ/pub?gid=1257988366&single=true&output=csv",
    "Supreme 6/58": "https://docs.google.com/spreadsheets/d/e/2PACX-1vS8b63hsr-rj0VCeviGFy36wVCEGRvKlUY98RSUMZtPyC6s9NxiUwuzA_ZvJ4pcujHoSpFhBCH4m2pJ/pub?gid=543531444&single=true&output=csv",
}

# ---------- Auto-refresh helpers ----------
if 'refresh' not in st.session_state:
    st.session_state['refresh'] = False

def clear_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass

if st.sidebar.button("🔄 Refresh Data"):
    st.spinner("Refreshing data... please wait.")
    try:
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Cache clear failed: {e}")
    # trigger rerun by updating session_state
    st.session_state['refresh'] = not st.session_state.get('refresh', False)


# ---------- Utils ----------
@st.cache_data(ttl=300)
def load_csv_from_url(url):
    return pd.read_csv(url)

def add_time_features(df):
    """Ensure DrawDate exists as datetime and add Year/Month/Day/Weekday."""
    df = df.copy()
    if "DrawDate" not in df.columns:
        return df
    # try parse with format YYYYMMDD then fallback
    df["DrawDate"] = pd.to_datetime(df["DrawDate"].astype(str), format="%Y%m%d", errors="coerce")
    if df["DrawDate"].isna().all():
        df["DrawDate"] = pd.to_datetime(df["DrawDate"].astype(str), errors="coerce")
    df = df.dropna(subset=["DrawDate"]).copy()
    df["Year"] = df["DrawDate"].dt.year
    df["Month"] = df["DrawDate"].dt.month
    df["Day"] = df["DrawDate"].dt.day
    df["Weekday"] = df["DrawDate"].dt.day_name()
    # sort chronological
    df = df.sort_values("DrawDate").reset_index(drop=True)
    return df

def preprocess_draws(df, max_num):
    draw_cols = [c for c in df.columns if c.lower().startswith("drawnno")]
    if len(draw_cols) < 6:
        draw_cols = [f"DrawnNo{i}" for i in range(1,7) if f"DrawnNo{i}" in df.columns]
    draws_df = df[draw_cols].dropna().astype(int)
    sorted_draws = draws_df.apply(lambda r: tuple(sorted(int(x) for x in r)), axis=1).tolist()
    counts = draws_df.values.ravel()
    freq = pd.Series(counts).value_counts().sort_index()
    for n in range(1, max_num+1):
        if n not in freq.index:
            freq.loc[n] = 0
    freq = freq.sort_index()
    return sorted_draws, freq

def apply_filters(df, year_range, months, weekdays):
    df2 = df.copy()
    if year_range:
        df2 = df2[(df2["Year"] >= year_range[0]) & (df2["Year"] <= year_range[1])]
    if months:
        df2 = df2[df2["Month"].isin(months)]
    if weekdays:
        df2 = df2[df2["Weekday"].isin(weekdays)]
    return df2

# ---------- small utility: sample without replacement from weighted scores ----------
def weighted_sample_no_replace(items, scores, k, temperature=1.0):
    items = list(items)
    scores = np.array(scores, dtype=float)
    out = []
    available = items.copy()
    sc = scores.copy()
    for _ in range(min(k, len(items))):
        exp = np.exp(sc / max(1e-6, temperature))
        probs = exp / exp.sum()
        choice = np.random.choice(len(available), p=probs)
        out.append(available.pop(choice))
        sc = np.delete(sc, choice)
    return out

# ---------- Prediction helpers ----------
def gen_hot_cold(freq, pick=6, method="hot", variation=0):
    if method == "hot":
        topK = max(pick, pick + variation)
        top = list(map(int, freq.sort_values(ascending=False).index[:topK].tolist()))
        if variation == 0:
            return tuple(sorted(top[:pick]))
        else:
            s = sorted(random.sample(top, pick))
            return tuple(s)
    elif method == "cold":
        bottomK = max(pick, pick + variation)
        bottom = list(map(int, freq.sort_values(ascending=True).index[:bottomK].tolist()))
        if variation == 0:
            return tuple(sorted(bottom[:pick]))
        else:
            s = sorted(random.sample(bottom, pick))
            return tuple(s)
    elif method == "random-weighted":
        weights = freq.values.astype(float) + 1e-6
        nums = list(freq.index)
        chosen = np.random.choice(nums, size=pick, replace=False, p=weights/weights.sum())
        return tuple(sorted(int(x) for x in chosen))
    else:
        nums = list(freq.index)
        return tuple(sorted(int(x) for x in random.sample(nums, pick)))

def monte_carlo_suggest(freq, pick=6, n_sim=5000, top_k=5):
    nums = freq.index.tolist()
    probs = freq.values.astype(float)
    if probs.sum() == 0:
        probs = np.ones_like(probs)
    p = probs / probs.sum()
    counter = {}
    for _ in range(n_sim):
        s = tuple(sorted(np.random.choice(nums, size=pick, replace=False, p=p)))
        counter[s] = counter.get(s, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]

def build_ml_model(draws_list, max_num):
    rows = []
    for i in range(len(draws_list) - 1):
        cur = np.zeros(max_num, dtype=int)
        nxt = np.zeros(max_num, dtype=int)
        for n in draws_list[i]:
            cur[n-1] = 1
        for n in draws_list[i+1]:
            nxt[n-1] = 1
        rows.append((cur, nxt))
    if not rows:
        return None
    X = np.vstack([r[0] for r in rows])
    Y = np.vstack([r[1] for r in rows])
    models = {}
    for num_idx in range(max_num):
        y = Y[:, num_idx]
        if y.sum() < 3:
            models[num_idx+1] = None
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
        acc = accuracy_score(yte, clf.predict(Xte))
        models[num_idx+1] = (clf, acc)
    return models

def ml_score_list(models, last_draw):
    if models is None:
        return []
    max_num = len(models)
    x = np.zeros(max_num)
    for n in last_draw:
        if n-1 < max_num:
            x[n-1] = 1
    scores = []
    for num in range(1, max_num+1):
        mdl = models.get(num)
        if mdl is None:
            scores.append((num, 0.0))
        else:
            clf, acc = mdl
            prob = clf.predict_proba(x.reshape(1,-1))[0][1]
            score = prob * (0.5 + 0.5*acc)
            scores.append((num, score))
    return scores

def ml_predict(models, last_draw, pick=6, variation=0, temperature=1.0):
    if models is None:
        return ()
    scores = ml_score_list(models, last_draw)
    nums = [s[0] for s in scores]
    vals = np.array([s[1] for s in scores], dtype=float)
    if variation == 0:
        chosen = [s[0] for s in sorted(scores, key=lambda x:x[1], reverse=True)[:pick]]
        return tuple(sorted(int(x) for x in chosen))
    else:
        chosen = weighted_sample_no_replace(nums, vals, pick, temperature=temperature)
        return tuple(sorted(int(x) for x in chosen))

# ---------- Markov Chain (order 1/2/3) ----------
def build_markov_transitions(draws_list, order=1):
    transitions = defaultdict(Counter)
    for i in range(len(draws_list)-1):
        cur = draws_list[i]
        nxt = draws_list[i+1]
        for state in itertools.combinations(cur, order):
            state = tuple(sorted(state))
            for next_num in nxt:
                transitions[state][next_num] += 1
    return transitions

def markov_chain_predict_from_transitions(transitions, last_draw, max_num, pick=6, variation=0, temperature=1.0):
    # Aggregate counts for next numbers from all states that are subset of last_draw
    score_acc = Counter()
    for state, counter in transitions.items():
        if set(state).issubset(set(last_draw)):
            for num, cnt in counter.items():
                score_acc[num] += cnt
    if not score_acc:
        # fallback to top aggregated next counts or random
        all_counts = Counter()
        for c in transitions.values():
            all_counts.update(c)
        if all_counts:
            top = [n for n, _ in all_counts.most_common(pick)]
            return tuple(sorted(int(x) for x in top))
        return tuple(sorted(random.sample(range(1, max_num+1), pick)))
    # Deterministic top pick
    items = sorted(score_acc.items(), key=lambda x: x[1], reverse=True)
    nums = [n for n, v in items]
    vals = np.array([v for n, v in items], dtype=float)
    if variation == 0:
        chosen = nums[:pick]
        # fill if needed
        if len(chosen) < pick:
            for n in range(1, max_num+1):
                if n not in chosen:
                    chosen.append(n)
                if len(chosen) >= pick:
                    break
        return tuple(sorted(int(x) for x in chosen[:pick]))
    else:
        # stochastic sampling from the available scored numbers
        # extend candidate pool to include all numbers with zero counts to allow variety
        pool = nums.copy()
        pool_scores = vals.copy()
        # include remaining numbers with small epsilon score to allow sampling
        rest = [n for n in range(1, max_num+1) if n not in pool]
        if rest:
            pool += rest
            pool_scores = np.concatenate([pool_scores, np.ones(len(rest))*1e-6])
        sampled = weighted_sample_no_replace(pool, pool_scores, pick, temperature=temperature)
        return tuple(sorted(int(x) for x in sampled))

def markov_chain_predict(draws_list, max_num, order=1, pick=6, variation=0, temperature=1.0):
    if len(draws_list) < 2:
        return ()
    if order < 1:
        order = 1
    # if order > size of draw, fallback to 1
    last = draws_list[-1]
    if order > len(last):
        order = 1
    transitions = build_markov_transitions(draws_list, order=order)
    return markov_chain_predict_from_transitions(transitions, last, max_num, pick=pick, variation=variation, temperature=temperature)

# ---------- LSTM helpers (optional) ----------
def build_lstm_data(draws_list, max_num, seq_len=10):
    X, y = [], []
    for i in range(len(draws_list) - seq_len):
        seq = np.zeros((seq_len, max_num), dtype=float)
        for j in range(seq_len):
            for n in draws_list[i+j]:
                seq[j, n-1] = 1.0
        target = np.zeros(max_num, dtype=float)
        for n in draws_list[i+seq_len]:
            target[n-1] = 1.0
        X.append(seq); y.append(target)
    if not X:
        return None, None
    return np.array(X), np.array(y)

def build_lstm_model(seq_len, max_num, latent=64):
    model = Sequential([
        LSTM(latent, input_shape=(seq_len, max_num)),
        Dense(max_num, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model

# ---------- Backtest helpers & downloads ----------
def backtest_suggestions(draws_list, suggestions, bonus_list=None, has_bonus=False):
    """
    draws_list: historical main draws (list of tuples)
    suggestions: list of dicts OR list of main-lists (legacy). Accepts:
       - [{"main": (...), "bonus": X}, ...] OR
       - [[n1,n2,..], ...] (legacy)
    bonus_list: list of historical bonus numbers (aligned with draws_list) or None
    has_bonus: bool flag whether to include bonus columns
    """
    rows = []
    # Normalize suggestions input
    normalized = []
    if suggestions and isinstance(suggestions[0], dict):
        normalized = suggestions
    else:
        # legacy list of lists -> convert
        for s in suggestions:
            normalized.append({"main": tuple(s), "bonus": None})
    for s in normalized:
        main_combo = tuple(sorted(int(x) for x in s["main"]))
        exact = sum(1 for d in draws_list if tuple(sorted(main_combo)) == d)
        three_plus = sum(1 for d in draws_list if len(set(main_combo) & set(d)) >= 3)
        row = {"combo": ",".join(map(str, main_combo)), "exact_hits": int(exact), "3+_hits": int(three_plus)}
        if has_bonus and s.get("bonus") is not None and bonus_list is not None:
            bonus_val = int(s["bonus"])
            bonus_hits = sum(1 for b in bonus_list if int(b) == bonus_val)
            row["bonus_hits"] = int(bonus_hits)
        rows.append(row)
    return pd.DataFrame(rows)

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------- New helpers for bonus prediction ----------
def build_bonus_freq_from_df(filtered_df, max_num):
    if "BonusNo" not in filtered_df.columns:
        return [], None
    bonus_list = filtered_df["BonusNo"].dropna().astype(int).tolist()
    if not bonus_list:
        return [], None
    bonus_freq = pd.Series(bonus_list).value_counts().sort_index()
    for n in range(1, max_num+1):
        if n not in bonus_freq.index:
            bonus_freq.loc[n] = 0
    bonus_freq = bonus_freq.sort_index()
    return bonus_list, bonus_freq

def gen_bonus_hot_cold(bonus_freq, method="hot", variation=0):
    # return a single integer bonus pick (1..max_num)
    if bonus_freq is None:
        return None
    if method == "hot":
        topK = max(1, 1 + variation)
        top = list(map(int, bonus_freq.sort_values(ascending=False).index[:topK].tolist()))
        if variation == 0:
            return int(top[0])
        else:
            return int(random.choice(top))
    elif method == "cold":
        bottomK = max(1, 1 + variation)
        bottom = list(map(int, bonus_freq.sort_values(ascending=True).index[:bottomK].tolist()))
        if variation == 0:
            return int(bottom[0])
        else:
            return int(random.choice(bottom))
    else:
        nums = bonus_freq.index.tolist()
        weights = bonus_freq.values.astype(float)
        if weights.sum() == 0:
            return int(random.choice(nums))
        p = weights / weights.sum()
        return int(np.random.choice(nums, p=p))

def monte_carlo_suggest_bonus(bonus_freq, n_sim=5000, top_k=3):
    if bonus_freq is None:
        return []
    nums = bonus_freq.index.tolist()
    probs = bonus_freq.values.astype(float)
    if probs.sum() == 0:
        probs = np.ones_like(probs)
    p = probs / probs.sum()
    counter = {}
    for _ in range(n_sim):
        s = int(np.random.choice(nums, p=p))
        counter[s] = counter.get(s, 0) + 1
    return sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]

def build_ml_model_for_bonus(draws_list, bonus_list, max_num):
    """
    Build per-number logistic (one-vs-rest) models to predict the NEXT draw's bonus number
    using current main draw as features (same style as build_ml_model).
    """
    # Create rows aligned where next bonus exists
    X_rows = []
    y_bonus_vals = []
    for i in range(len(draws_list) - 1):
        # require next bonus to exist
        next_idx = i + 1
        if next_idx >= len(bonus_list):
            continue
        nb = bonus_list[next_idx]
        if pd.isna(nb):
            continue
        cur = np.zeros(max_num, dtype=int)
        for n in draws_list[i]:
            cur[n-1] = 1
        X_rows.append(cur)
        y_bonus_vals.append(int(nb))
    if not X_rows:
        return None
    X = np.vstack(X_rows)
    Yb = np.zeros((len(y_bonus_vals), max_num), dtype=int)
    for i, val in enumerate(y_bonus_vals):
        if 1 <= val <= max_num:
            Yb[i, val-1] = 1
    models = {}
    for num_idx in range(max_num):
        y = Yb[:, num_idx]
        if y.sum() < 3:
            models[num_idx+1] = None
            continue
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        try:
            clf = LogisticRegression(max_iter=200).fit(Xtr, ytr)
            acc = accuracy_score(yte, clf.predict(Xte))
            models[num_idx+1] = (clf, acc)
        except Exception:
            models[num_idx+1] = None
    return models

def ml_score_list_bonus(models_bonus, last_draw):
    if models_bonus is None:
        return []
    max_num = len(models_bonus)
    x = np.zeros(max_num)
    for n in last_draw:
        if n-1 < max_num:
            x[n-1] = 1
    scores = []
    for num in range(1, max_num+1):
        mdl = models_bonus.get(num)
        if mdl is None:
            scores.append((num, 0.0))
        else:
            clf, acc = mdl
            prob = clf.predict_proba(x.reshape(1,-1))[0][1]
            score = prob * (0.5 + 0.5*acc)
            scores.append((num, score))
    return scores

def ml_predict_bonus(models_bonus, last_draw, variation=0, temperature=1.0):
    if models_bonus is None:
        return None
    scores = ml_score_list_bonus(models_bonus, last_draw)
    nums = [s[0] for s in scores]
    vals = np.array([s[1] for s in scores], dtype=float)
    if variation == 0:
        chosen = [s[0] for s in sorted(scores, key=lambda x:x[1], reverse=True)[:1]]
        return int(chosen[0]) if chosen else None
    else:
        chosen = weighted_sample_no_replace(nums, vals, 1, temperature=temperature)
        return int(chosen[0]) if chosen else None

def build_markov_transitions_for_bonus(draws_list, bonus_list, order=1):
    transitions = defaultdict(Counter)
    for i in range(len(draws_list)-1):
        cur = draws_list[i]
        next_idx = i+1
        if next_idx >= len(bonus_list):
            continue
        next_bonus = bonus_list[next_idx]
        if pd.isna(next_bonus):
            continue
        for state in itertools.combinations(cur, order):
            state = tuple(sorted(state))
            transitions[state][int(next_bonus)] += 1
    return transitions

def markov_predict_bonus_from_transitions(transitions_bonus, last_draw, max_num, variation=0, temperature=1.0):
    # reuse logic to aggregate and return top 1 or sampled
    tup = markov_chain_predict_from_transitions(transitions_bonus, last_draw, max_num, pick=1, variation=variation, temperature=temperature)
    if not tup:
        return None
    return int(tup[0])

def build_lstm_data_with_bonus(draws_list, bonus_list, max_num, seq_len=10):
    X, y_main, y_bonus = [], [], []
    for i in range(len(draws_list) - seq_len):
        seq = np.zeros((seq_len, max_num), dtype=float)
        valid = True
        for j in range(seq_len):
            for n in draws_list[i+j]:
                seq[j, n-1] = 1.0
        next_idx = i + seq_len
        # need both main target and bonus target
        if next_idx >= len(draws_list) or next_idx >= len(bonus_list):
            continue
        target_main = np.zeros(max_num, dtype=float)
        for n in draws_list[next_idx]:
            target_main[n-1] = 1.0
        next_bonus = bonus_list[next_idx]
        if pd.isna(next_bonus):
            continue
        target_bonus = np.zeros(max_num, dtype=float)
        if 1 <= int(next_bonus) <= max_num:
            target_bonus[int(next_bonus)-1] = 1.0
        else:
            continue
        X.append(seq); y_main.append(target_main); y_bonus.append(target_bonus)
    if not X:
        return None, None, None
    return np.array(X), np.array(y_main), np.array(y_bonus)

# ---------- UI & Main ----------
st.sidebar.title("Global Filters")
game = st.sidebar.selectbox("Game", list(GSHEET_URLS.keys()))
df_raw = load_csv_from_url(GSHEET_URLS[game])
df = add_time_features(df_raw)

# ---------- Show latest draw dates for all games ----------
latest_dates = {}
for g_name, url in GSHEET_URLS.items():
    df_game = add_time_features(load_csv_from_url(url))
    if "DrawDate" in df_game.columns and not df_game["DrawDate"].isna().all():
        latest_dates[g_name] = df_game["DrawDate"].max().strftime("%Y-%m-%d")
    else:
        latest_dates[g_name] = "N/A"

st.sidebar.markdown("### Latest Draw Dates")
for g, d in latest_dates.items():
    st.sidebar.write(f"**{g}:** {d}")


# Year slider (safe defaults)
if "Year" in df.columns and not df["Year"].isna().all():
    min_y, max_y = int(df["Year"].min()), int(df["Year"].max())
else:
    min_y, max_y = 2010, 2025
if min_y == max_y:
    year_range = st.sidebar.slider("Year range", 2010, 2025, (2010, 2025))
else:
    year_range = st.sidebar.slider("Year range", min_y, max_y, (min_y, max_y))

months = st.sidebar.multiselect("Months (empty = all)", list(range(1,13)))
weekdays = st.sidebar.multiselect("Weekdays (empty = all)",
                                 ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# Apply filters to DataFrame
filtered_df = apply_filters(df, year_range, months, weekdays)

# Download filtered test data button
st.sidebar.markdown("### Data download")
if not filtered_df.empty:
    csv_bytes = df_to_csv_bytes(filtered_df)
    st.sidebar.download_button("Download filtered data CSV", data=csv_bytes, file_name=f"{game.replace(' ','_')}_filtered.csv", mime="text/csv")

# Determine max_num
if "50" in game:
    max_num = 50
elif "55" in game:
    max_num = 55
else:
    max_num = 58

# Auto-detect bonus availability (Star 6/50 => True)
has_bonus = True if "50" in game else False

# Build draws list and frequency (main)
draws_list, freq = preprocess_draws(filtered_df, max_num)

# Build bonus list & freq if available and has_bonus
bonus_list = []
bonus_freq = None
if has_bonus and "BonusNo" in filtered_df.columns:
    # keep order aligned with filtered_df (which was sorted by DrawDate earlier)
    bonus_list = filtered_df["BonusNo"].dropna().astype(int).tolist()
    if bonus_list:
        bonus_freq = pd.Series(bonus_list).value_counts().sort_index()
        for n in range(1, max_num+1):
            if n not in bonus_freq.index:
                bonus_freq.loc[n] = 0
        bonus_freq = bonus_freq.sort_index()
    else:
        bonus_freq = None

st.sidebar.title("Pages")
page = st.sidebar.radio("Navigate", ["Analysis", "Prediction", "Simulation"])

if not draws_list:
    st.warning("No draws available for these filters.")
    st.stop()

# ---------- Pages ----------
markov_order = 1

if page == "Analysis":
    st.header("Analysis")
    # show most recent draw; include bonus if present
    most_recent_main = list(map(int, draws_list[-1])) if draws_list else 'N/A'
    if has_bonus and bonus_list:
        # attempt to get the last bonus aligned with draws_list length
        last_bonus = None
        if len(bonus_list) == len(draws_list):
            last_bonus = int(bonus_list[-1])
        elif len(bonus_list) > 0:
            last_bonus = int(bonus_list[-1])
        else:
            last_bonus = None
        if last_bonus is not None:
            st.write(f"Most recent draw: {most_recent_main} — Bonus: {last_bonus}")
        else:
            st.write(f"Most recent draw: {most_recent_main}")
    else:
        st.write(f"Most recent draw: {most_recent_main if draws_list else 'N/A'}")

    st.subheader("Overall Number Frequencies")
    st.bar_chart(freq)

    # Bonus frequency chart for Star 6/50
    if has_bonus and bonus_freq is not None:
        st.subheader("Bonus Number Frequency")
        st.bar_chart(bonus_freq)

    st.subheader("Hot Numbers (Top 10)")
    st.table(freq.sort_values(ascending=False).head(10).rename_axis("Number").reset_index(name="Count"))

    st.subheader("Cold Numbers (Bottom 10)")
    st.table(freq.sort_values(ascending=True).head(10).rename_axis("Number").reset_index(name="Count"))

    # Pair & triple analysis
    st.subheader("Top Number Pairs & Triples")
    pair_counter, triple_counter = Counter(), Counter()
    for d in draws_list:
        pair_counter.update([tuple(sorted(p)) for p in itertools.combinations(d, 2)])
        triple_counter.update([tuple(sorted(t)) for t in itertools.combinations(d, 3)])
    pair_df = pd.DataFrame(pair_counter.most_common(20), columns=["Pair", "Count"])
    triple_df = pd.DataFrame(triple_counter.most_common(20), columns=["Triple", "Count"])
    st.write("Top Pairs")
    st.dataframe(pair_df)
    st.write("Top Triples")
    st.dataframe(triple_df)

    # Gap analysis
    st.subheader("Gap Analysis (draw spacing)")
    last_seen = {}
    gaps_per_number = defaultdict(list)
    for idx, d in enumerate(draws_list):
        for n in d:
            if n in last_seen:
                gap = idx - last_seen[n]
                gaps_per_number[n].append(gap)
            last_seen[n] = idx
    avg_gaps = {n: (np.mean(gaps_per_number[n]) if gaps_per_number[n] else np.nan) for n in range(1, max_num+1)}
    avg_gaps_series = pd.Series(avg_gaps).dropna().sort_values()
    st.dataframe(avg_gaps_series.rename("AvgGap").reset_index().rename(columns={"index":"Number"}).head(20))

    # Markov transition inspection
    st.subheader("Inspect Markov transitions")
    transitions_all = build_markov_transitions(draws_list, order=markov_order)
    state_counts = [(s, sum(c.values())) for s, c in transitions_all.items()]
    state_counts = sorted(state_counts, key=lambda x: x[1], reverse=True)
    states_list = [s for s, _ in state_counts]
    if states_list:
        chosen_state = st.selectbox("Choose state to inspect", options=states_list, format_func=lambda x: str(x))
        probs_counter = transitions_all.get(chosen_state, Counter())
        prob_vec = np.zeros(max_num, dtype=float)
        total = sum(probs_counter.values())
        if total > 0:
            for num, cnt in probs_counter.items():
                if 1 <= num <= max_num:
                    prob_vec[num-1] = cnt / total
        fig, ax = plt.subplots(1,1, figsize=(10,3))
        ax.bar(range(1, max_num+1), prob_vec)
        ax.set_xlabel("Next number")
        ax.set_ylabel("P(next|state)")
        ax.set_title(f"Conditional probabilities for state {chosen_state}")
        st.pyplot(fig)
        fig2, ax2 = plt.subplots(1,1, figsize=(10,1.6))
        sns.heatmap(prob_vec.reshape(1,-1), cmap="YlOrRd", cbar=True, ax=ax2)
        ax2.set_yticks([])
        ax2.set_xticks(np.arange(0, max_num, max(1, max_num//10)))
        st.pyplot(fig2)
    else:
        st.info("No Markov states available to inspect (insufficient data).")

    # Co-occurrence heatmap
    st.subheader("Number Co-occurrence Heatmap")
    co_matrix = np.zeros((max_num, max_num), dtype=int)
    for d in draws_list:
        for i in range(len(d)):
            for j in range(i+1, len(d)):
                a, b = d[i]-1, d[j]-1
                co_matrix[a, b] += 1
                co_matrix[b, a] += 1
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(co_matrix, cmap="YlGnBu", ax=ax, cbar=True)
    ax.set_title("Number Co-occurrence")
    st.pyplot(fig)

    # Clustering
    st.subheader("Clustering (KMeans)")
    if len(draws_list) > 20:
        X = np.array([np.bincount(d, minlength=max_num+1)[1:] for d in draws_list])
        n_clusters = st.slider("KMeans clusters", 2, 8, 3)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        st.dataframe(pd.Series(kmeans.labels_).value_counts().rename_axis("Cluster").reset_index(name="Count"))
        largest = int(pd.Series(kmeans.labels_).value_counts().idxmax())
        center = kmeans.cluster_centers_[largest]
        top_nums = (np.argsort(center)[::-1][:6] + 1).tolist()
        st.write(f"Representative numbers (cluster {largest}): {top_nums}")
    else:
        st.info("Not enough draws for clustering.")

elif page == "Prediction":
    st.header("Prediction")
    method = st.selectbox("Method", ["Hot","Cold","Monte Carlo","ML Logistic","LSTM (if available)","Hybrid (Freq+ML)","Markov Chain"])
    n_suggestions = st.slider("Number of suggestions", 1, 10, 3)

    # Sampling/variation controls
    variation = st.slider("Variation (0 = deterministic top picks; >0 adds randomness)", 0, 10, 2)
    temp = st.slider("Sampling temperature (for ML/Markov sampling)", 0.1, 5.0, 1.0)

    # Hybrid weight control
    hybrid_ml_weight = st.slider("Hybrid ML weight (0..1) — higher gives ML more influence", 0.0, 1.0, 0.5)

    # For Markov: choose order
    markov_order = 1
    if method == "Markov Chain":
        markov_order = st.selectbox("Markov order (state size)", [1,2,3], index=0,
                                    help="Order=1: condition on single number; Order=2: pairs; Order=3: triples")

    if st.button("Generate"):
        suggestions = []
        # Prepare ML models (main + bonus) lazily
        ml_models = None
        ml_models_bonus = None
        if method == "ML Logistic" or method == "Hybrid (Freq+ML)":
            ml_models = build_ml_model(draws_list, max_num)
            if has_bonus and bonus_list:
                ml_models_bonus = build_ml_model_for_bonus(draws_list, bonus_list, max_num)

        # Prepare LSTM data/models if needed
        lstm_model_main = None
        lstm_model_bonus = None
        if method == "LSTM (if available)" and TF_AVAILABLE:
            X_all, y_all = build_lstm_data(draws_list, max_num, seq_len=10)
            # For bonus LSTM prepare separate data if possible
            Xb, yb = None, None
            if has_bonus and bonus_list:
                Xb, yb = build_lstm_data_with_bonus(draws_list, bonus_list, max_num, seq_len=10)[0:2]  # returns (X,y_main,y_bonus)
            if X_all is not None:
                lstm_model_main = build_lstm_model(X_all.shape[1], max_num, latent=64)
                with st.spinner("Training LSTM (brief)..."):
                    lstm_model_main.fit(X_all, y_all, epochs=3, batch_size=16, verbose=0)
            if has_bonus and TF_AVAILABLE:
                data = build_lstm_data_with_bonus(draws_list, bonus_list, max_num, seq_len=10)
                if data is not None:
                    Xseq, ymain_seq, ybonus_seq = data
                    # train bonus model if enough data
                    if Xseq is not None and len(Xseq) > 5:
                        lstm_model_bonus = build_lstm_model(Xseq.shape[1], max_num, latent=64)
                        with st.spinner("Training Bonus LSTM (brief)..."):
                            lstm_model_bonus.fit(Xseq, ybonus_seq, epochs=3, batch_size=16, verbose=0)

        if method == "Hot":
            for _ in range(n_suggestions):
                main_pick = gen_hot_cold(freq, pick=6, method="hot", variation=variation)
                bonus_pick = None
                if has_bonus and bonus_freq is not None:
                    bonus_pick = gen_bonus_hot_cold(bonus_freq, method="hot", variation=variation)
                suggestions.append({"main": main_pick, "bonus": bonus_pick})
        elif method == "Cold":
            for _ in range(n_suggestions):
                main_pick = gen_hot_cold(freq, pick=6, method="cold", variation=variation)
                bonus_pick = None
                if has_bonus and bonus_freq is not None:
                    bonus_pick = gen_bonus_hot_cold(bonus_freq, method="cold", variation=variation)
                suggestions.append({"main": main_pick, "bonus": bonus_pick})
        elif method == "Monte Carlo":
            mc = monte_carlo_suggest(freq, pick=6, n_sim=5000, top_k=n_suggestions)
            main_sug_list = [c for c, _ in mc]
            # bonus top picks via monte carlo
            bonus_top = []
            if has_bonus and bonus_freq is not None:
                bonus_top = [b for b, _ in monte_carlo_suggest_bonus(bonus_freq, n_sim=5000, top_k=n_suggestions)]
            for i in range(len(main_sug_list)):
                bp = bonus_top[i] if (has_bonus and i < len(bonus_top)) else (gen_bonus_hot_cold(bonus_freq) if has_bonus else None)
                suggestions.append({"main": main_sug_list[i], "bonus": bp})
        elif method == "ML Logistic":
            if ml_models is None:
                st.warning("Not enough data to train ML models.")
            else:
                for _ in range(n_suggestions):
                    main_pick = ml_predict(ml_models, draws_list[-1], pick=6, variation=variation, temperature=temp)
                    bonus_pick = None
                    if has_bonus and ml_models_bonus is not None:
                        bonus_pick = ml_predict_bonus(ml_models_bonus, draws_list[-1], variation=variation, temperature=temp)
                    elif has_bonus and bonus_freq is not None:
                        # fallback to freq
                        bonus_pick = gen_bonus_hot_cold(bonus_freq, method="hot", variation=variation)
                    suggestions.append({"main": main_pick, "bonus": bonus_pick})
        elif method == "LSTM (if available)":
            if not TF_AVAILABLE:
                st.error("TensorFlow not available in environment.")
            else:
                data = build_lstm_data(draws_list, max_num, seq_len=10)
                if data[0] is None:
                    st.warning("Not enough data for LSTM.")
                else:
                    # main
                    Xseq, yseq = data
                    probs_main = lstm_model_main.predict(Xseq[-1].reshape(1, Xseq.shape[1], max_num))[0]
                    top_idx = np.argsort(probs_main)[::-1]
                    pool = (top_idx[:12] + 1).tolist()
                    # bonus
                    pool_bonus = None
                    if has_bonus and lstm_model_bonus is not None:
                        data_b = build_lstm_data_with_bonus(draws_list, bonus_list, max_num, seq_len=10)
                        if data_b is not None:
                            Xb, ymb, ybb = data_b
                            probs_b = lstm_model_bonus.predict(Xb[-1].reshape(1, Xb.shape[1], max_num))[0]
                            bp_idx = np.argsort(probs_b)[::-1]
                            pool_bonus = (bp_idx[:6] + 1).tolist()
                    for _ in range(n_suggestions):
                        if variation == 0:
                            chosen = sorted([int(x) for x in pool[:6]])
                        else:
                            chosen = sorted(np.random.choice(pool, size=6, replace=False).tolist())
                        if has_bonus:
                            if pool_bonus:
                                if variation == 0:
                                    bonus_pick = int(pool_bonus[0])
                                else:
                                    bonus_pick = int(np.random.choice(pool_bonus))
                            else:
                                bonus_pick = gen_bonus_hot_cold(bonus_freq, method="hot", variation=variation)
                        else:
                            bonus_pick = None
                        suggestions.append({"main": tuple(chosen), "bonus": bonus_pick})
        elif method == "Hybrid (Freq+ML)":
            # Hybrid main (existing)
            models = ml_models
            hot_rank = list(freq.sort_values(ascending=False).index)  # most -> least
            if models:
                ml_scores_dict = dict(ml_score_list(models, draws_list[-1]))
                ml_vals = np.array([ml_scores_dict.get(n, 0.0) for n in hot_rank], dtype=float)
                if ml_vals.max() > 0:
                    ml_norm = (ml_vals - ml_vals.min()) / (ml_vals.max() - ml_vals.min() + 1e-9)
                else:
                    ml_norm = np.zeros_like(ml_vals)
                freq_vals = np.array([freq.loc[n] for n in hot_rank], dtype=float)
                if freq_vals.max() > 0:
                    freq_norm = (freq_vals - freq_vals.min()) / (freq_vals.max() - freq_vals.min() + 1e-9)
                else:
                    freq_norm = np.zeros_like(freq_vals)
                combined = hybrid_ml_weight * ml_norm + (1.0 - hybrid_ml_weight) * freq_norm
                # Show top combined scores table
                combined_df = pd.DataFrame({
                    "Number": hot_rank,
                    "ML_score": ml_vals,
                    "Freq": freq_vals,
                    "Combined": combined
                })
                combined_df = combined_df.sort_values("Combined", ascending=False).reset_index(drop=True)
                st.subheader("Top combined scores (Hybrid)")
                st.dataframe(combined_df.head(20))
                pool = combined_df["Number"].tolist()[:20]
                pool_scores = combined_df["Combined"].tolist()[:20]

                # Bonus hybrid
                bonus_pool = None
                bonus_pool_scores = None
                if has_bonus and bonus_freq is not None:
                    # ML scores for bonus
                    if ml_models_bonus is not None:
                        ml_scores_b = dict(ml_score_list_bonus(ml_models_bonus, draws_list[-1]))
                        hot_rank_b = list(bonus_freq.sort_values(ascending=False).index)
                        ml_vals_b = np.array([ml_scores_b.get(n, 0.0) for n in hot_rank_b], dtype=float)
                        if ml_vals_b.max() > 0:
                            ml_norm_b = (ml_vals_b - ml_vals_b.min()) / (ml_vals_b.max() - ml_vals_b.min() + 1e-9)
                        else:
                            ml_norm_b = np.zeros_like(ml_vals_b)
                        freq_vals_b = np.array([bonus_freq.loc[n] for n in hot_rank_b], dtype=float)
                        if freq_vals_b.max() > 0:
                            freq_norm_b = (freq_vals_b - freq_vals_b.min()) / (freq_vals_b.max() - freq_vals_b.min() + 1e-9)
                        else:
                            freq_norm_b = np.zeros_like(freq_vals_b)
                        combined_b = hybrid_ml_weight * ml_norm_b + (1.0 - hybrid_ml_weight) * freq_norm_b
                        combined_df_b = pd.DataFrame({
                            "Number": hot_rank_b,
                            "ML_score": ml_vals_b,
                            "Freq": freq_vals_b,
                            "Combined": combined_b
                        })
                        combined_df_b = combined_df_b.sort_values("Combined", ascending=False).reset_index(drop=True)
                        bonus_pool = combined_df_b["Number"].tolist()[:10]
                        bonus_pool_scores = combined_df_b["Combined"].tolist()[:10]
                    else:
                        bonus_pool = list(bonus_freq.sort_values(ascending=False).index[:10])
                        bonus_pool_scores = bonus_freq.sort_values(ascending=False).values[:10].tolist()

                for _ in range(n_suggestions):
                    if variation == 0:
                        chosen = sorted([int(x) for x in pool[:6]])
                    else:
                        sampled = weighted_sample_no_replace(pool, pool_scores, 6, temperature=temp)
                        chosen = sorted(int(x) for x in sampled)
                    # bonus pick
                    bonus_pick = None
                    if has_bonus:
                        if bonus_pool is not None:
                            if variation == 0:
                                bonus_pick = int(bonus_pool[0])
                            else:
                                # sample 1
                                bonus_pick = int(weighted_sample_no_replace(bonus_pool, bonus_pool_scores, 1, temperature=temp)[0])
                        else:
                            bonus_pick = gen_bonus_hot_cold(bonus_freq, method="hot", variation=variation)
                    suggestions.append({"main": tuple(chosen), "bonus": bonus_pick})
            else:
                for _ in range(n_suggestions):
                    main_pick = gen_hot_cold(freq, pick=6, method="hot", variation=variation)
                    bonus_pick = gen_bonus_hot_cold(bonus_freq, method="hot", variation=variation) if has_bonus else None
                    suggestions.append({"main": main_pick, "bonus": bonus_pick})
        elif method == "Markov Chain":
            for _ in range(n_suggestions):
                main_pick = markov_chain_predict(draws_list, max_num, order=markov_order, pick=6, variation=variation, temperature=temp)
                bonus_pick = None
                if has_bonus:
                    trans_b = build_markov_transitions_for_bonus(draws_list, bonus_list, order=markov_order)
                    bonus_pick = markov_predict_bonus_from_transitions(trans_b, draws_list[-1], max_num, variation=variation, temperature=temp)
                suggestions.append({"main": main_pick, "bonus": bonus_pick})

        # Clean and display
        st.subheader("🎲 Suggested Numbers")
        for i, s in enumerate(suggestions, 1):
            main_display = list(map(int, s["main"]))
            if has_bonus and s.get("bonus") is not None:
                st.write(f"Set {i}: Main: {main_display}, Bonus: {int(s['bonus'])}")
            else:
                st.write(f"Set {i}: Main: {main_display}")

        # Backtest summary and downloadable CSV
        st.subheader("Backtest summary for these suggestions (historical)")
        bt_df = backtest_suggestions(draws_list, suggestions, bonus_list=bonus_list if has_bonus else None, has_bonus=has_bonus)
        st.dataframe(bt_df)
        csv_bytes = df_to_csv_bytes(bt_df)
        st.download_button("Download backtest CSV", data=csv_bytes, file_name="backtest_results.csv", mime="text/csv")

elif page == "Simulation":
    st.header("Simulation")
    st.info("Backtesting strategies and Markov orders")

    if len(draws_list) < 20:
        st.warning("Not enough draws to run simulation/backtests.")
    else:
        test_size = st.slider("Test window size (most recent draws used for testing)", 10, min(10, len(draws_list)-1), min(50, max(10, len(draws_list)-1)))
        train = draws_list[:-test_size]
        test = draws_list[-test_size:]

        # prepare bonus train/test aligned if has_bonus
        train_bonus = []
        test_bonus = []
        if has_bonus and bonus_list:
            # We assume bonus_list aligns with draws_list / filtered_df
            if len(bonus_list) >= len(draws_list):
                train_bonus = bonus_list[:-test_size]
                test_bonus = bonus_list[-test_size:]
            else:
                # best effort alignment
                train_bonus = bonus_list[:len(train)]
                test_bonus = bonus_list[-len(test):]

        # Hot backtest
        st.subheader("Hot strategy backtest")
        hot = gen_hot_cold(freq, pick=6, method="hot")
        hits_hot = [len(set(hot) & set(d)) for d in test]
        st.write(f"Hot combo (top6): {list(hot)}")
        st.write("Average hits per draw (hot):", np.mean(hits_hot))
        st.line_chart(hits_hot)

        if has_bonus and bonus_freq is not None:
            hot_bonus = gen_bonus_hot_cold(bonus_freq, method="hot")
            hits_hot_bonus = [1 if int(hot_bonus) == int(b) else 0 for b in test_bonus]
            st.write(f"Hot bonus pick: {hot_bonus} — avg bonus hits: {np.mean(hits_hot_bonus):.2f}")

        # ---------- Monte Carlo backtest ----------
        st.subheader("Monte Carlo backtest")

        # Run Monte Carlo suggestions
        mc_top = monte_carlo_suggest(freq, pick=6, n_sim=2000, top_k=3)

        for combo, cnt in mc_top:
            # Ensure combo is Python ints
            combo_py = [int(x) for x in combo]

            # Calculate hits
            hits = [len(set(combo_py) & set([int(x) for x in d])) for d in test]

            # Display
            st.write(f"MC combo {combo_py} — avg hits: {float(np.mean(hits)):.2f}")

            # Streamlit line chart (requires float)
            st.line_chart([float(h) for h in hits])

            # MC bonus check (top 3)
            if has_bonus and bonus_freq is not None:
                bonus_mc_top = monte_carlo_suggest_bonus(bonus_freq, n_sim=2000, top_k=1)
                if bonus_mc_top:
                    bonus_choice = bonus_mc_top[0][0]
                    hits_b = [1 if int(bonus_choice) == int(b) else 0 for b in test_bonus]
                    st.write(f"MC bonus pick {bonus_choice} — avg bonus hits: {float(np.mean(hits_b)):.2f}")

        # ML Logistic backtest
        st.subheader("ML Logistic backtest")
        ml_models = build_ml_model(train, max_num)
        ml_models_bonus = None
        if has_bonus and train_bonus:
            ml_models_bonus = build_ml_model_for_bonus(train, train_bonus, max_num)
        if ml_models:
            ml_sug = ml_predict(ml_models, train[-1], pick=6)
            hits_ml = [len(set(ml_sug) & set(d)) for d in test]
            st.write(f"ML suggestion: {list(ml_sug)} — avg hits: {np.mean(hits_ml):.2f}")
            st.line_chart(hits_ml)
            if has_bonus:
                if ml_models_bonus:
                    ml_bonus_pick = ml_predict_bonus(ml_models_bonus, train[-1], variation=0, temperature=1.0)
                else:
                    ml_bonus_pick = gen_bonus_hot_cold(bonus_freq, method="hot")
                hits_ml_b = [1 if int(ml_bonus_pick) == int(b) else 0 for b in test_bonus]
                st.write(f"ML bonus pick: {ml_bonus_pick} — avg bonus hits: {np.mean(hits_ml_b):.2f}")
        else:
            st.info("Not enough data to train ML models for backtest.")

        # LSTM backtest
        if TF_AVAILABLE:
            st.subheader("LSTM backtest")
            X, y = build_lstm_data(train, max_num, seq_len=10)
            Xb, yb = None, None
            if has_bonus and train_bonus:
                data_b = build_lstm_data_with_bonus(train, train_bonus, max_num, seq_len=10)
                if data_b is not None:
                    Xb, ydummy, yb = data_b
            if X is not None and len(X) > 10:
                lstm_model = build_lstm_model(X.shape[1], max_num, latent=64)
                with st.spinner("Training small LSTM for backtest..."):
                    lstm_model.fit(X, y, epochs=3, batch_size=16, verbose=0)
                probs = lstm_model.predict(X[-1].reshape(1, X.shape[1], max_num))[0]
                idx = np.argsort(probs)[::-1][:6]
                lstm_sug = tuple(sorted(int(i)+1 for i in idx))
                hits_lstm = [len(set(lstm_sug) & set(d)) for d in test]
                st.write(f"LSTM suggestion: {list(lstm_sug)} — avg hits: {np.mean(hits_lstm):.2f}")
                st.line_chart(hits_lstm)
                if has_bonus:
                    if Xb is not None and yb is not None and len(Xb) > 5:
                        lstm_model_b = build_lstm_model(Xb.shape[1], max_num, latent=64)
                        with st.spinner("Training small Bonus LSTM for backtest..."):
                            lstm_model_b.fit(Xb, yb, epochs=3, batch_size=16, verbose=0)
                        probs_b = lstm_model_b.predict(Xb[-1].reshape(1, Xb.shape[1], max_num))[0]
                        bidx = np.argmax(probs_b)
                        lstm_bonus = int(bidx) + 1
                    else:
                        lstm_bonus = gen_bonus_hot_cold(bonus_freq, method="hot")
                    hits_lb = [1 if int(lstm_bonus) == int(b) else 0 for b in test_bonus]
                    st.write(f"LSTM bonus suggestion: {lstm_bonus} — avg bonus hits: {np.mean(hits_lb):.2f}")
            else:
                st.info("Not enough sequences to train LSTM for backtest.")

        # ---------- Markov Chain orders backtest ----------
        st.subheader("Markov Chain orders backtest (order 1,2,3)")

        markov_results = {}
        test_py = [[int(x) for x in row] for row in test]  # ensure Python ints

        for order in [1, 2, 3]:
            simulated_train = train.copy()
            simulated_bonus_train = train_bonus.copy() if has_bonus and train_bonus else []
            hits = []
            hits_bonus = []

            st.write(f"### Order {order} — first 3 draws")
            
            for i, actual in enumerate(test_py):
                # Build transition matrix from current simulated_train
                trans_now = build_markov_transitions(simulated_train, order=order)
                
                # Predict next numbers
                pred = markov_chain_predict_from_transitions(
                    trans_now, simulated_train[-1], max_num, pick=6
                )
                pred = [int(x) for x in pred]  # convert to Python int

                # Count hits as Python int
                hit_count = int(len(set(pred) & set(actual)))
                hits.append(hit_count)

                # Bonus prediction using markov on bonus transitions
                if has_bonus and simulated_bonus_train:
                    trans_now_b = build_markov_transitions_for_bonus(simulated_train, simulated_bonus_train, order=order)
                    pred_b = markov_predict_bonus_from_transitions(trans_now_b, simulated_train[-1], max_num)
                    actual_b = int(test_bonus[i]) if i < len(test_bonus) else None
                    hit_b = int(pred_b == actual_b) if (pred_b is not None and actual_b is not None) else 0
                    hits_bonus.append(hit_b)

                # Update simulated_train
                simulated_train.append(actual)

                # Update simulated_bonus_train
                if has_bonus and i < len(test_bonus):
                    simulated_bonus_train.append(int(test_bonus[i]))

                # Display only first 3 draws
                if i < 3:
                    if has_bonus and hits_bonus:
                        st.write(
                            f"Draw {i+1}: Predicted {pred} — Actual {actual} — Hits {hit_count} — BonusPred {pred_b} — BonusActual {test_bonus[i]} — BonusHits {hit_b}"
                        )
                    else:
                        st.write(
                            f"Draw {i+1}: Predicted {pred} — Actual {actual} — Hits {hit_count}"
                        )

            # Store results
            markov_results[order] = hits

            # Average hits
            avg_hits = float(np.mean(hits))
            st.write(f"Order {order} — avg hits: {avg_hits:.2f}")
            if has_bonus and hits_bonus:
                st.write(f"Order {order} — avg bonus hits: {float(np.mean(hits_bonus)):.2f}")

            # Line chart (convert hits to float for Streamlit)
            st.line_chart([float(h) for h in hits])

        # Summary table
        summary = pd.DataFrame({
            "Order": list(markov_results.keys()),
            "AvgHits": [float(np.mean(v)) for v in markov_results.values()]
        })

        st.dataframe(summary)
        st.download_button(
            "Download Markov summary CSV",
            data=df_to_csv_bytes(summary),
            file_name="markov_summary.csv",
            mime="text/csv"
        )

st.caption("Reminder: lottery draws are random. These are statistical tools and do NOT guarantee winning numbers.")
