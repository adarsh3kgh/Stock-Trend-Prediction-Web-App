import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = load_model("Stock Predictions Model.keras")

# ---------------- HEADER ----------------
st.markdown(
    """
    <h1 style='text-align: center; color: #00BFFF;'>
        📈 AI Stock Trend Prediction App
    </h1>
    """,
    unsafe_allow_html=True
)

st.write("Predict future stock trends using Deep Learning.")

# ---------------- SIDEBAR ----------------
st.sidebar.header("Settings")

stock = st.sidebar.text_input("Enter Stock Symbol", "AAPL")

start_date = st.sidebar.date_input(
    "Start Date",
    pd.to_datetime("2015-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    pd.to_datetime("today")
)

# ---------------- FETCH DATA ----------------
with st.spinner("Fetching stock data..."):

    data = yf.download(stock, start=start_date, end=end_date)

if data.empty:
    st.error("Invalid stock symbol or no data found.")
    st.stop()

# ---------------- STOCK DATA ----------------
st.subheader(f"{stock} Stock Data")

st.dataframe(data.tail(10), use_container_width=True)

# ---------------- CLOSE PRICE CHART ----------------
st.subheader("Closing Price")

fig = plt.figure(figsize=(12, 6))
plt.plot(data.Close, label='Closing Price')
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

st.pyplot(fig)

# ---------------- MOVING AVERAGES ----------------
ma50 = data.Close.rolling(50).mean()
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()

st.subheader("Moving Averages")

fig2 = plt.figure(figsize=(12, 6))

plt.plot(data.Close, label='Close Price')
plt.plot(ma50, label='MA50')
plt.plot(ma100, label='MA100')
plt.plot(ma200, label='MA200')

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()

st.pyplot(fig2)

# ---------------- TRAIN TEST SPLIT ----------------
data_train = pd.DataFrame(data.Close[: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)

final_df = pd.concat([past_100_days, data_test], ignore_index=True)

input_data = scaler.fit_transform(final_df)

# ---------------- PREPARE DATA ----------------
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# ---------------- PREDICTION ----------------
predictions = model.predict(x_test)

scale_factor = 1 / scaler.scale_[0]

predictions = predictions * scale_factor
y_test = y_test * scale_factor

# ---------------- PREDICTION GRAPH ----------------
st.subheader("Actual vs Predicted Price")

fig3 = plt.figure(figsize=(12, 6))

plt.plot(y_test, label='Actual Price')
plt.plot(predictions, label='Predicted Price')

plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()

st.pyplot(fig3)

# ---------------- METRICS ----------------
latest_actual = float(y_test[-1])
latest_prediction = float(predictions[-1][0])

change = latest_prediction - latest_actual

col1, col2, col3 = st.columns(3)

col1.metric("Latest Actual Price", f"${latest_actual:.2f}")
col2.metric("Predicted Price", f"${latest_prediction:.2f}")
col3.metric("Difference", f"${change:.2f}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with Streamlit + TensorFlow + Yahoo Finance")
