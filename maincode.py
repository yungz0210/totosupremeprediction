# streamlit_app.py
# Toto Predictor — Enhanced with LSTM, Global Filters, Pages
# Single-file Streamlit app (copy-paste)

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
def clear_cache():
    try:
        st.cache_data.clear()
    except Exception:
        pass

if st.sidebar.button("🔄 Refresh Data"):
    clear_cache()
    st.experimental_rerun()

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
def backtest_suggestions(draws_list, suggestions):
    rows = []
    for s in suggestions:
        exact = sum(1 for d in draws_list if tuple(sorted(s)) == d)
        three_plus = sum(1 for d in draws_list if len(set(s) & set(d)) >= 3)
        rows.append({"combo": ",".join(map(str,s)), "exact_hits": exact, "3+_hits": three_plus})
    return pd.DataFrame(rows)

def df_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")

# ---------- UI & Main ----------
st.sidebar.title("Global Filters")
game = st.sidebar.selectbox("Game", list(GSHEET_URLS.keys()))
df_raw = load_csv_from_url(GSHEET_URLS[game])
df = add_time_features(df_raw)

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

# Build draws list and frequency
draws_list, freq = preprocess_draws(filtered_df, max_num)

st.sidebar.title("Pages")
page = st.sidebar.radio("Navigate", ["Analysis", "Prediction", "Simulation"])

if not draws_list:
    st.warning("No draws available for these filters.")
    st.stop()

# ---------- Pages ----------
if page == "Analysis":
    st.header("Analysis")
    st.write(f"Most recent draw: {list(map(int, draws_list[-1])) if draws_list else 'N/A'}")

    st.subheader("Overall Number Frequencies")
    st.bar_chart(freq)

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
    all_gaps = []
    for v in gaps_per_number.values():
        all_gaps.extend(v)
    if all_gaps:
        st.bar_chart(pd.Series(all_gaps).value_counts().sort_index())

    # Co-occurrence heatmap
    st.subheader("Number Co-occurrence Heatmap")
    import matplotlib.pyplot as plt
    import seaborn as sns
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

    # Yearly/monthly/weekday trends
    st.subheader("Yearly Trends")
    if "Year" in filtered_df.columns:
        st.line_chart(filtered_df.groupby("Year").size())
    st.subheader("Monthly Counts")
    st.bar_chart(filtered_df.groupby("Month").size().reindex(range(1,13), fill_value=0))
    st.subheader("Weekday Counts")
    st.bar_chart(filtered_df.groupby("Weekday").size().reindex(
        ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"], fill_value=0
    ))

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

    # If Markov mode show transition inspection UI
    if method == "Markov Chain":
        st.subheader("Inspect Markov transitions")
        transitions_all = build_markov_transitions(draws_list, order=markov_order)
        # list available states, show top states by count
        state_counts = [(s, sum(c.values())) for s, c in transitions_all.items()]
        state_counts = sorted(state_counts, key=lambda x: x[1], reverse=True)
        states_list = [s for s, _ in state_counts]
        if states_list:
            chosen_state = st.selectbox("Choose state to inspect", options=states_list, format_func=lambda x: str(x))
            # compute probability vector for next numbers
            probs_counter = transitions_all.get(chosen_state, Counter())
            prob_vec = np.zeros(max_num, dtype=float)
            total = sum(probs_counter.values())
            if total > 0:
                for num, cnt in probs_counter.items():
                    if 1 <= num <= max_num:
                        prob_vec[num-1] = cnt / total
            # plot bar chart & heatmap
            fig, ax = plt.subplots(1,1, figsize=(10,3))
            ax.bar(range(1, max_num+1), prob_vec)
            ax.set_xlabel("Next number")
            ax.set_ylabel("P(next|state)")
            ax.set_title(f"Conditional probabilities for state {chosen_state}")
            st.pyplot(fig)
            # small heatmap 1 x N
            fig2, ax2 = plt.subplots(1,1, figsize=(10,1.6))
            sns.heatmap(prob_vec.reshape(1,-1), cmap="YlOrRd", cbar=True, ax=ax2)
            ax2.set_yticks([])
            ax2.set_xticks(np.arange(0, max_num, max(1, max_num//10)))
            st.pyplot(fig2)
        else:
            st.info("No Markov states available to inspect (insufficient data).")

    if st.button("Generate"):
        suggestions = []
        if method == "Hot":
            for _ in range(n_suggestions):
                suggestions.append(gen_hot_cold(freq, pick=6, method="hot", variation=variation))
        elif method == "Cold":
            for _ in range(n_suggestions):
                suggestions.append(gen_hot_cold(freq, pick=6, method="cold", variation=variation))
        elif method == "Monte Carlo":
            mc = monte_carlo_suggest(freq, pick=6, n_sim=5000, top_k=n_suggestions)
            suggestions = [c for c, _ in mc]
        elif method == "ML Logistic":
            models = build_ml_model(draws_list, max_num)
            if models is None:
                st.warning("Not enough data to train ML models.")
            else:
                for _ in range(n_suggestions):
                    suggestions.append(ml_predict(models, draws_list[-1], pick=6, variation=variation, temperature=temp))
        elif method == "LSTM (if available)":
            if not TF_AVAILABLE:
                st.error("TensorFlow not available in environment.")
            else:
                X, y = build_lstm_data(draws_list, max_num, seq_len=10)
                if X is None:
                    st.warning("Not enough data for LSTM.")
                else:
                    model = build_lstm_model(X.shape[1], max_num, latent=64)
                    with st.spinner("Training LSTM (brief)..."):
                        model.fit(X, y, epochs=3, batch_size=16, verbose=0)
                    probs = model.predict(X[-1].reshape(1, X.shape[1], max_num))[0]
                    top_idx = np.argsort(probs)[::-1]
                    pool = (top_idx[:12] + 1).tolist()
                    for _ in range(n_suggestions):
                        if variation == 0:
                            chosen = sorted([int(x) for x in pool[:6]])
                        else:
                            chosen = sorted(np.random.choice(pool, size=6, replace=False).tolist())
                        suggestions.append(tuple(chosen))
        elif method == "Hybrid (Freq+ML)":
            models = build_ml_model(draws_list, max_num)
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
                for _ in range(n_suggestions):
                    if variation == 0:
                        chosen = sorted([int(x) for x in pool[:6]])
                    else:
                        sampled = weighted_sample_no_replace(pool, pool_scores, 6, temperature=temp)
                        chosen = sorted(int(x) for x in sampled)
                    suggestions.append(tuple(chosen))
            else:
                for _ in range(n_suggestions):
                    suggestions.append(gen_hot_cold(freq, pick=6, method="hot", variation=variation))
        elif method == "Markov Chain":
            for _ in range(n_suggestions):
                suggestions.append(markov_chain_predict(draws_list, max_num, order=markov_order, pick=6, variation=variation, temperature=temp))

        # Clean and display
        clean_suggestions = [list(map(int, s)) for s in suggestions]
        st.subheader("🎲 Suggested Numbers")
        for i, s in enumerate(clean_suggestions, 1):
            st.write(f"Set {i}: {s}")

        # Backtest summary and downloadable CSV
        st.subheader("Backtest summary for these suggestions (historical)")
        bt_df = backtest_suggestions(draws_list, clean_suggestions)
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

        # Hot backtest
        st.subheader("Hot strategy backtest")
        hot = gen_hot_cold(freq, pick=6, method="hot")
        hits_hot = [len(set(hot) & set(d)) for d in test]
        st.write(f"Hot combo (top6): {list(hot)}")
        st.write("Average hits per draw (hot):", np.mean(hits_hot))
        st.line_chart(hits_hot)

        # Monte Carlo backtest
        st.subheader("Monte Carlo backtest")
        mc_top = monte_carlo_suggest(freq, pick=6, n_sim=2000, top_k=3)
        for combo, cnt in mc_top:
            hits = [len(set(combo) & set(d)) for d in test]
            st.write(f"MC combo {list(combo)} — avg hits: {np.mean(hits):.2f}")
            st.line_chart(hits)

        # ML Logistic backtest
        st.subheader("ML Logistic backtest")
        ml_models = build_ml_model(train, max_num)
        if ml_models:
            ml_sug = ml_predict(ml_models, train[-1], pick=6)
            hits_ml = [len(set(ml_sug) & set(d)) for d in test]
            st.write(f"ML suggestion: {list(ml_sug)} — avg hits: {np.mean(hits_ml):.2f}")
            st.line_chart(hits_ml)
        else:
            st.info("Not enough data to train ML models for backtest.")

        # LSTM backtest
        if TF_AVAILABLE:
            st.subheader("LSTM backtest")
            X, y = build_lstm_data(train, max_num, seq_len=10)
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
            else:
                st.info("Not enough sequences to train LSTM for backtest.")

        # Markov orders backtest
        st.subheader("Markov Chain orders backtest (order 1,2,3)")
        markov_results = {}
        for order in [1,2,3]:
            simulated_train = train.copy()
            hits = []
            for i, actual in enumerate(test):
                trans_now = build_markov_transitions(simulated_train, order=order)
                pred = markov_chain_predict_from_transitions(trans_now, simulated_train[-1], max_num, pick=6)
                hits.append(len(set(pred) & set(actual)))
                simulated_train = simulated_train + [actual]
            markov_results[order] = hits
            st.write(f"Order {order} — avg hits: {np.mean(hits):.2f}")
            st.line_chart(hits)

        summary = pd.DataFrame({
            "Order": list(markov_results.keys()),
            "AvgHits": [np.mean(v) for v in markov_results.values()]
        })
        st.dataframe(summary)

        # Download summary CSV
        csv_bytes = df_to_csv_bytes(summary)
        st.download_button("Download Markov summary CSV", data=csv_bytes, file_name="markov_summary.csv", mime="text/csv")

st.caption("Reminder: lottery draws are random. These are statistical tools and do NOT guarantee winning numbers.")
