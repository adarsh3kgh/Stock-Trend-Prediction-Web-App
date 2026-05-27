```markdown
# 📈 StockSight AI — Stock Trend Prediction Web App

A deep learning-powered stock trend prediction app built with **Streamlit**, **TensorFlow/Keras**, and **Yahoo Finance**.

🔗 **Live Demo:** [stock-trend-prediction-web-app-3.streamlit.app](https://stock-trend-prediction-web-app-3.streamlit.app/)

---

## 🚀 Features

- 🔍 Search any stock by ticker symbol (e.g. `AAPL`, `TSLA`, `RELIANCE.NS`)
- 📊 Interactive closing price chart with historical data
- 📉 Toggleable Moving Averages — MA50, MA100, MA200
- 🤖 LSTM deep learning model predicts future price trends
- 📈 Actual vs Predicted price comparison chart
- 💡 Live metrics — latest price, 52-week high/low, prediction signal (Bullish/Bearish)
- 🌙 Dark-themed, responsive UI

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Model | LSTM (TensorFlow / Keras) |
| Data Source | Yahoo Finance (`yfinance`) |
| Data Processing | NumPy, Pandas, Scikit-learn |
| Visualization | Matplotlib |
| Deployment | Streamlit Cloud |

---

## 🧠 How It Works

1. Historical stock data is fetched from Yahoo Finance
2. Data is split 80/20 into training and test sets
3. A **2-layer LSTM** model trained on 100-day rolling windows predicts the next closing price
4. Predicted prices are inverse-scaled and plotted against actual prices
5. A **Bullish / Bearish / Neutral** signal is generated from the difference

---

## 📁 Project Structure

```
stock-trend-prediction-web-app/
│
├── app.py                        # Main Streamlit application
├── Stock Predictions Model.keras # Pre-trained LSTM model
├── requirements.txt              # Python dependencies
├── .python-version               # Pins Python 3.11 for deployment
└── README.md                     # You are here
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/stock-trend-prediction-web-app.git
cd stock-trend-prediction-web-app

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit==1.35.0
numpy==1.26.4
pandas==2.2.2
yfinance==0.2.54
tensorflow==2.16.1
scikit-learn==1.5.0
matplotlib==3.8.4
```

---

## ⚠️ Disclaimer

> This app is built for **educational purposes only**.
> Predictions made by this model are **not financial advice**.
> Do not make investment decisions based on this tool.

---

## 🙌 Acknowledgements

- [Yahoo Finance](https://finance.yahoo.com/) for free stock data via `yfinance`
- [Streamlit](https://streamlit.io/) for the deployment platform
- [TensorFlow](https://www.tensorflow.org/) for the deep learning framework

---

## 📬 Contact

Built with ❤️ — feel free to open an issue or submit a PR!
```
