import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Stock Predictor",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("📈 AI Stock Market Predictor Dashboard")

st.markdown("Real-Time Stock Analysis & AI Prediction System")

# ---------------- SIDEBAR ----------------

st.sidebar.header("Dashboard Controls")

stock = st.sidebar.text_input(
    "Enter Stock Symbol",
    "AAPL"
)

show_data = st.sidebar.checkbox("Show Raw Data", True)

show_ma = st.sidebar.checkbox("Show Moving Average", True)

show_volume = st.sidebar.checkbox("Show Volume Chart", True)

show_prediction = st.sidebar.checkbox(
    "Show AI Prediction",
    True
)

# ---------------- DOWNLOAD DATA ----------------

data = yf.download(
    stock,
    start="2020-01-01"
)

# ---------------- COMPANY INFO ----------------

ticker = yf.Ticker(stock)

info = ticker.info

# ---------------- METRICS ----------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Current Price",
        f"${info.get('currentPrice', 'N/A')}"
    )

with col2:
    st.metric(
        "Market Cap",
        info.get('marketCap', 'N/A')
    )

with col3:
    st.metric(
        "52W High",
        info.get('fiftyTwoWeekHigh', 'N/A')
    )

# ---------------- SHOW DATA ----------------

if show_data:
    st.subheader("📊 Stock Data")
    st.dataframe(data.tail())

# ---------------- CANDLESTICK CHART ----------------

st.subheader("📉 Candlestick Chart")

fig = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Candlestick'
)])

# ---------------- MOVING AVERAGE ----------------

if show_ma:

    data['MA50'] = data['Close'].rolling(50).mean()

    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['MA50'],
        mode='lines',
        name='50-Day MA'
    ))

# ---------------- LAYOUT ----------------

fig.update_layout(
    height=700,
    xaxis_rangeslider_visible=False,
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- VOLUME CHART ----------------

if show_volume:

    st.subheader("📦 Trading Volume")

    volume_fig = go.Figure()

    volume_fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume'
    ))

    volume_fig.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        volume_fig,
        use_container_width=True
    )

# ---------------- PREPARE DATA ----------------

dataset = data['Close'].values
dataset = dataset.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))

scaled_data = scaler.fit_transform(dataset)

x_train = []
y_train = []

for i in range(60, len(scaled_data)):

    x_train.append(
        scaled_data[i-60:i, 0]
    )

    y_train.append(
        scaled_data[i, 0]
    )

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(
    x_train,
    (
        x_train.shape[0],
        x_train.shape[1],
        1
    )
)

# ---------------- BUILD MODEL ----------------

model = Sequential()

model.add(LSTM(
    units=50,
    return_sequences=True,
    input_shape=(
        x_train.shape[1],
        1
    )
))

model.add(LSTM(units=50))

model.add(Dense(units=1))

model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# ---------------- TRAIN MODEL ----------------

with st.spinner("Training AI Model..."):

    model.fit(
        x_train,
        y_train,
        epochs=1,
        batch_size=32,
        verbose=0
    )

# ---------------- PREDICTION ----------------

x_test = x_train[-100:]

predicted_prices = model.predict(
    x_test,
    verbose=0
)

predicted_prices = scaler.inverse_transform(
    predicted_prices
)

# ---------------- PREDICTION GRAPH ----------------

if show_prediction:

    st.subheader("🤖 AI Price Prediction")

    prediction_fig = go.Figure()

    prediction_fig.add_trace(go.Scatter(
        y=predicted_prices.flatten(),
        mode='lines',
        name='Predicted Prices'
    ))

    prediction_fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        prediction_fig,
        use_container_width=True
    )

# ---------------- BUY/SELL SIGNAL ----------------

latest_prediction = float(predicted_prices[-1][0])

latest_actual = float(data['Close'].iloc[-1])

st.subheader("📢 AI Recommendation")

if latest_prediction > latest_actual:

    st.success("✅ BUY SIGNAL")

else:

    st.error("❌ SELL SIGNAL")

# ---------------- COMPANY DETAILS ----------------

st.subheader("🏢 Company Information")

st.write(info.get('longBusinessSummary', 'No Information'))

# ---------------- FOOTER ----------------

st.markdown("---")

st.markdown(
    "Made with ❤️ using Streamlit, TensorFlow & AI"
)