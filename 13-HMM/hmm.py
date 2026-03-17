import os

# Work around Streamlit watcher issues on some Python/Windows environments.
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import streamlit as st
import pandas as pd
import numpy as np
import kagglehub
from hmmlearn import hmm

# -------------------------
# Load Dataset and Train HMM
# -------------------------
@st.cache_resource(show_spinner="Loading Dataset and Training HMM...")
def build_model():
    # Download dataset from Kaggle
    path = kagglehub.dataset_download("sumanthvrao/daily-climate-time-series-data")
    data = pd.read_csv(os.path.join(path, "DailyDelhiClimateTrain.csv"))

    feature_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]
    X = data[feature_cols].to_numpy(dtype=float)

    feature_means = X.mean(axis=0)
    feature_stds = X.std(axis=0)
    feature_stds = np.where(feature_stds > 0, feature_stds, 1.0)

    X_scaled = (X - feature_means) / feature_stds

    n_states = 3

    def train_candidate(seed):
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=400,
            min_covar=1e-3,
            random_state=seed,
            params="mc",
            init_params="",
        )

        model.startprob_ = np.full(n_states, 1.0 / n_states)
        model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)

        base_mean = X_scaled.mean(axis=0)
        means_init = np.tile(base_mean, (n_states, 1))
        means_init[:, 0] = np.quantile(X_scaled[:, 0], [0.2, 0.5, 0.8])
        model.means_ = means_init

        base_var = np.var(X_scaled, axis=0) + 1e-3
        model.covars_ = np.tile(base_var, (n_states, 1))

        model.fit(X_scaled)

        states = model.predict(X_scaled)
        counts = np.bincount(states, minlength=n_states)
        balance_ratio = counts.min() / max(counts.max(), 1)
        ll_per_sample = model.score(X_scaled) / len(X_scaled)
        quality = ll_per_sample + 0.5 * balance_ratio

        return model, quality

    best_model = None
    best_quality = -np.inf
    for seed in [7, 21, 42, 84, 126]:
        candidate_model, quality = train_candidate(seed)
        if quality > best_quality:
            best_model = candidate_model
            best_quality = quality

    model = best_model

    train_states = model.predict(X_scaled)
    state_counts = np.bincount(train_states, minlength=n_states).astype(float)
    state_balance_weights = len(X_scaled) / (n_states * np.maximum(state_counts, 1.0))
    state_balance_weights = np.clip(state_balance_weights, 0.7, 1.5)

    temp_state_order = np.argsort(model.means_[:, 0])
    cold_state = int(temp_state_order[0])
    moderate_state = int(temp_state_order[1])
    hot_state = int(temp_state_order[2])

    state_label_map = {
        cold_state: "Cold",
        moderate_state: "Moderate",
        hot_state: "Hot",
    }

    cold_threshold, hot_threshold = np.quantile(data["meantemp"].to_numpy(dtype=float), [0.33, 0.66])

    return (
        model,
        feature_cols,
        feature_means,
        feature_stds,
        state_label_map,
        state_balance_weights,
        (cold_state, moderate_state, hot_state),
        float(cold_threshold),
        float(hot_threshold),
    )


(
    model,
    feature_cols,
    feature_means,
    feature_stds,
    state_label_map,
    state_balance_weights,
    state_roles,
    cold_threshold,
    hot_threshold,
) = build_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("Weather State Detection using HMM")
st.subheader("Enter Weather Inputs to Predict Hidden Weather State")

temp_input = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=50.0,
    value=25.0,
    step=0.1
)

humidity_input = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=60.0,
    step=0.1
)

wind_input = st.number_input(
    "Wind Speed",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.1
)

pressure_input = st.number_input(
    "Mean Pressure",
    min_value=900.0,
    max_value=1200.0,
    value=1015.0,
    step=0.1
)

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Weather State"):
    obs_raw = np.array([[temp_input, humidity_input, wind_input, pressure_input]], dtype=float)
    obs = (obs_raw - feature_means) / feature_stds

    _, posteriors = model.score_samples(obs)
    eps = 1e-12
    cold_state, moderate_state, hot_state = state_roles

    scores = np.log(posteriors[0] + eps) + np.log(state_balance_weights + eps)

    temp_means_raw = model.means_[:, 0] * feature_stds[0] + feature_means[0]
    covars = model.covars_
    if covars.ndim == 2:
        temp_vars_scaled = np.maximum(covars[:, 0], 1e-6)
    else:
        temp_vars_scaled = np.maximum(covars[:, 0, 0], 1e-6)
    temp_vars_raw = temp_vars_scaled * (feature_stds[0] ** 2)

    temp_loglik = -0.5 * (
        np.log(2 * np.pi * np.maximum(temp_vars_raw, 1e-6))
        + ((temp_input - temp_means_raw) ** 2) / np.maximum(temp_vars_raw, 1e-6)
    )
    scores += 0.8 * temp_loglik

    if temp_input <= cold_threshold:
        scores[cold_state] += 0.3
        scores[hot_state] -= 0.2
    elif temp_input >= hot_threshold:
        scores[hot_state] += 0.3
        scores[cold_state] -= 0.2
    else:
        scores[moderate_state] += 0.15

    hidden_state = int(np.argmax(scores))
    predicted_label = state_label_map[hidden_state]

    st.success(f"The predicted weather state is: **{predicted_label}**")