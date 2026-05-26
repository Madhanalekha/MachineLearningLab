import streamlit as st
import numpy as np
import pandas as pd
import kagglehub
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


@st.cache_resource
def load_stock_data():
    path = kagglehub.dataset_download("jacksoncrow/stock-market-dataset")

    preferred_files = ["AAPL.csv", "AAPL_data.csv"]
    csv_files = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".csv"):
                csv_files.append(os.path.join(root, file))

    for preferred in preferred_files:
        for file_path in csv_files:
            if os.path.basename(file_path).lower() == preferred.lower():
                df = pd.read_csv(file_path)
                if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                    return df

    for file_path in csv_files:
        df = pd.read_csv(file_path)
        if {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
            return df

    raise ValueError("No suitable stock CSV found with required OHLCV columns.")


class TradingRL:
    def __init__(self):
        self.actions = ["Buy", "Sell", "Hold"]
        self.q_table = {}

        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state(self, price, prev_price):
        if price > prev_price:
            return "UP"
        elif price < prev_price:
            return "DOWN"
        else:
            return "STABLE"

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.actions))
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))

        best_next = np.max(self.q_table[next_state])

        self.q_table[state][action] += self.alpha * (
            reward + self.gamma * best_next - self.q_table[state][action]
        )

    def train(self, prices):
        for i in range(1, len(prices)-1):
            state = self.get_state(prices[i], prices[i-1])
            next_state = self.get_state(prices[i+1], prices[i])

            action = self.choose_action(state)

            if action == 0:  # Buy
                reward = prices[i+1] - prices[i]
            elif action == 1:  # Sell
                reward = prices[i] - prices[i+1]
            else:  # Hold
                reward = 0

            self.update(state, action, reward, next_state)

    def predict(self, price, prev_price):
        state = self.get_state(price, prev_price)
        action = self.choose_action(state)
        return self.actions[action], self.q_table[state]

@st.cache_resource
def train_models(df):
    df = df.dropna()

    df['Price_Change'] = df['Close'].diff()
    df['Trend'] = df['Price_Change'].apply(lambda x: 1 if x > 0 else 0)

    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features]
    y = df['Trend']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier()
    rf.fit(X_scaled, y)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)

    return rf, kmeans, scaler


def main():
    st.set_page_config(page_title="Smart Trading AI", layout="wide")

    st.title("📈 Smart Stock Trading Intelligence System")
    st.caption("Reinforcement Learning based Trading Advisor")

    df = load_stock_data()
    st.success("✅ Dataset Loaded from KaggleHub")

    rf, kmeans, scaler = train_models(df)

    rl_agent = TradingRL()
    rl_agent.train(df['Close'].values)


    st.header("📊 Enter Stock Details")

    col1, col2 = st.columns(2)

    with col1:
        open_p = st.number_input("Open Price", value=100.0)
        high_p = st.number_input("High Price", value=105.0)
        low_p = st.number_input("Low Price", value=95.0)

    with col2:
        volume = st.number_input("Volume", value=1000000.0)
        prev_price = st.number_input("Previous Close", value=100.0)
        curr_price = st.number_input("Current Price", value=102.0)

    if st.button("🔍 Analyze & Recommend"):
        user_data = np.array([[open_p, high_p, low_p, volume]])
        user_scaled = scaler.transform(user_data)


        trend_pred = rf.predict(user_scaled)[0]
        cluster = kmeans.predict(user_scaled)[0]

        trend_label = "📈 Uptrend" if trend_pred == 1 else "📉 Downtrend"

        st.subheader("📊 Market Analysis")

        col3, col4 = st.columns(2)

        with col3:
            st.metric("Trend Prediction", trend_label)

        with col4:
            st.metric("Market Cluster", f"Cluster {cluster}")


        action, q_values = rl_agent.predict(curr_price, prev_price)

        st.subheader("🤖 RL Trading Recommendation")

        st.success(f"Recommended Action: **{action}**")

        action_df = pd.DataFrame({
            "Action": ["Buy", "Sell", "Hold"],
            "Q-Value": q_values
        })

        st.bar_chart(action_df.set_index("Action"))


        st.subheader("📉 Price Trend Visualization")

        fig, ax = plt.subplots()
        ax.plot(df['Close'].head(100))
        ax.set_title("Stock Price Trend")
        st.pyplot(fig)


def run_cli_demo():
    print("Running CLI demo for Smart Trading AI...")

    df = load_stock_data()

    rl_agent = TradingRL()
    rl_agent.train(df['Close'].values)

    prev_price = float(df['Close'].iloc[-2])
    curr_price = float(df['Close'].iloc[-1])

    action, q_values = rl_agent.predict(curr_price, prev_price)

    print(f"Latest prices -> prev: {prev_price:.2f}, current: {curr_price:.2f}")
    print(f"Recommended Action: {action}")
    print(f"Q-values [Buy, Sell, Hold]: {q_values}")

def is_running_in_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False

if __name__ == "__main__":
    if is_running_in_streamlit():
        main()
    else:
        run_cli_demo()