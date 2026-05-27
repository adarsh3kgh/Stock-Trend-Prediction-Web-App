import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="StockSight AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #111827 100%);
    border-right: 1px solid #1e293b;
  }
  section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

  /* Main background */
  .stApp { background-color: #0a0e1a; }

  /* Hero banner */
  .hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: '';
    position: absolute;
    top: -40%;
    left: 50%;
    transform: translateX(-50%);
    width: 500px;
    height: 300px;
    background: radial-gradient(ellipse, rgba(56,189,248,0.12) 0%, transparent 70%);
    pointer-events: none;
  }
  .hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    color: #f8fafc;
    letter-spacing: -1px;
    margin: 0;
  }
  .hero-title span { color: #38bdf8; }
  .hero-sub {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.5rem;
    letter-spacing: 0.05em;
    font-weight: 300;
  }

  /* Section headers */
  .section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 0.4rem;
  }
  .section-title {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 1rem;
  }

  /* Metric cards */
  .metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
  }
  .metric-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    position: relative;
    overflow: hidden;
  }
  .metric-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
    background: #38bdf8;
    border-radius: 4px 0 0 4px;
  }
  .metric-card.green::after { background: #4ade80; }
  .metric-card.red::after   { background: #f87171; }
  .metric-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-family: 'Space Mono', monospace; font-size: 1.65rem; font-weight: 700; color: #f8fafc; }
  .metric-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
  }
  .badge-up   { background: rgba(74,222,128,0.15); color: #4ade80; }
  .badge-down { background: rgba(248,113,113,0.15); color: #f87171; }
  .badge-flat { background: rgba(148,163,184,0.15); color: #94a3b8; }

  /* Info box */
  .info-box {
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #94a3b8;
  }

  /* Chart container */
  .chart-wrap {
    background: #0d1321;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
  }

  /* Hide default streamlit chrome */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem; }

  /* Divider */
  .fancy-divider {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #334155, transparent);
    margin: 2rem 0;
  }

  /* Ticker badge */
  .ticker-tag {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    padding: 4px 14px;
    border-radius: 999px;
    margin-left: 10px;
    vertical-align: middle;
  }

  /* Streamlit native overrides */
  .stMetric { background: #111827 !important; border-radius: 12px; padding: 1rem !important; border: 1px solid #1e293b; }
  div[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
  .stSpinner > div { border-top-color: #38bdf8 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0d1321",
    "axes.facecolor":    "#0d1321",
    "axes.edgecolor":    "#1e293b",
    "axes.labelcolor":   "#64748b",
    "xtick.color":       "#475569",
    "ytick.color":       "#475569",
    "text.color":        "#94a3b8",
    "grid.color":        "#1e293b",
    "grid.linewidth":    0.6,
    "legend.framealpha": 0.0,
    "legend.labelcolor": "#94a3b8",
    "font.family":       "monospace",
    "font.size":         10,
})

PALETTE = {
    "close":  "#38bdf8",
    "ma50":   "#818cf8",
    "ma100":  "#fb923c",
    "ma200":  "#f472b6",
    "actual": "#4ade80",
    "pred":   "#f87171",
}

# ─────────────────────────────────────────────
#  LOAD MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_keras_model():
    return load_model("Stock Predictions Model.keras")

model = load_keras_model()

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <p class="hero-title">📡 Stock<span>Sight</span> AI</p>
  <p class="hero-sub">Deep Learning · LSTM · Yahoo Finance · Real-time Predictions</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    stock = st.text_input("🔍 Stock Symbol", value="AAPL", help="e.g. AAPL, TSLA, GOOGL, RELIANCE.NS").upper().strip()

    st.markdown("**📅 Date Range**")
    col_s, col_e = st.columns(2)
    with col_s:
        start_date = st.date_input("From", value=date(2015, 1, 1), label_visibility="collapsed")
    with col_e:
        end_date = st.date_input("To", value=date.today(), label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**📊 Moving Averages**")
    show_ma50  = st.checkbox("MA 50",  value=True)
    show_ma100 = st.checkbox("MA 100", value=True)
    show_ma200 = st.checkbox("MA 200", value=True)

    st.markdown("---")
    run_btn = st.button("🚀 Analyse & Predict", use_container_width=True, type="primary")

    st.markdown("""
    <div class="info-box">
      Model: LSTM (2-layer)<br>
      Trained on 80% historical data<br>
      Window: 100 days
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FETCH DATA
# ─────────────────────────────────────────────
if not run_btn:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color: #334155;">
      <p style="font-size:3rem;">📊</p>
      <p style="font-family: 'Space Mono', monospace; font-size:1rem;">Enter a stock symbol and click <b style="color:#38bdf8">Analyse & Predict</b></p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner(f"Fetching data for **{stock}**…"):
    data = yf.download(stock, start=start_date, end=end_date, progress=False)

if data.empty:
    st.error(f"❌ No data found for **{stock}**. Check the symbol and try again.")
    st.stop()

# ─────────────────────────────────────────────
#  STOCK OVERVIEW
# ─────────────────────────────────────────────
close = data["Close"].squeeze()
latest_close  = float(close.iloc[-1])
prev_close    = float(close.iloc[-2])
price_change  = latest_close - prev_close
pct_change    = (price_change / prev_close) * 100
high_52w      = float(close.rolling(252).max().iloc[-1])
low_52w       = float(close.rolling(252).min().iloc[-1])

badge_cls  = "badge-up"   if price_change >= 0 else "badge-down"
badge_text = f"▲ {pct_change:+.2f}%" if price_change >= 0 else f"▼ {pct_change:+.2f}%"
card_cls   = "green"      if price_change >= 0 else "red"

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card {card_cls}">
    <div class="metric-label">Latest Close</div>
    <div class="metric-value">${latest_close:,.2f}</div>
    <span class="metric-badge {badge_cls}">{badge_text}</span>
  </div>
  <div class="metric-card">
    <div class="metric-label">52-Week High</div>
    <div class="metric-value">${high_52w:,.2f}</div>
  </div>
  <div class="metric-card">
    <div class="metric-label">52-Week Low</div>
    <div class="metric-value">${low_52w:,.2f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  RAW DATA TABLE
# ─────────────────────────────────────────────
with st.expander(f"📋 Raw Data — {stock}  (last 10 rows)", expanded=False):
    st.dataframe(
        data.tail(10).style.format("{:.2f}"),
        use_container_width=True
    )

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHART HELPERS
# ─────────────────────────────────────────────
def styled_fig(height=5):
    fig, ax = plt.subplots(figsize=(13, height))
    ax.grid(axis="y", linestyle="--")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=30, ha="right")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    return fig, ax

# ─────────────────────────────────────────────
#  CLOSING PRICE CHART
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">Chart 01</div><div class="section-title">Closing Price History</div>', unsafe_allow_html=True)

fig1, ax1 = styled_fig()
ax1.plot(close.index, close.values, color=PALETTE["close"], linewidth=1.6, label="Close")
ax1.fill_between(close.index, close.values, alpha=0.06, color=PALETTE["close"])
ax1.set_ylabel("Price (USD)")
ax1.legend()
st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.pyplot(fig1, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
plt.close(fig1)

# ─────────────────────────────────────────────
#  MOVING AVERAGES
# ─────────────────────────────────────────────
if show_ma50 or show_ma100 or show_ma200:
    st.markdown('<div class="section-label">Chart 02</div><div class="section-title">Moving Averages</div>', unsafe_allow_html=True)

    ma50  = close.rolling(50).mean()
    ma100 = close.rolling(100).mean()
    ma200 = close.rolling(200).mean()

    fig2, ax2 = styled_fig()
    ax2.plot(close.index, close.values, color=PALETTE["close"], linewidth=1.4, label="Close", alpha=0.85)
    if show_ma50:
        ax2.plot(ma50.index,  ma50.values,  color=PALETTE["ma50"],  linewidth=1.3, linestyle="--", label="MA 50")
    if show_ma100:
        ax2.plot(ma100.index, ma100.values, color=PALETTE["ma100"], linewidth=1.3, linestyle="--", label="MA 100")
    if show_ma200:
        ax2.plot(ma200.index, ma200.values, color=PALETTE["ma200"], linewidth=1.3, linestyle="--", label="MA 200")
    ax2.set_ylabel("Price (USD)")
    ax2.legend()
    st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
    st.pyplot(fig2, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    plt.close(fig2)

st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PREDICTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-label">LSTM Model</div><div class="section-title">Actual vs Predicted Price</div>', unsafe_allow_html=True)

with st.spinner("Running LSTM inference…"):
    data_train = pd.DataFrame(close[: int(len(close) * 0.80)])
    data_test  = pd.DataFrame(close[int(len(close) * 0.80):])

    scaler       = MinMaxScaler(feature_range=(0, 1))
    past_100     = data_train.tail(100)
    final_df     = pd.concat([past_100, data_test], ignore_index=True)
    input_scaled = scaler.fit_transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_scaled.shape[0]):
        x_test.append(input_scaled[i-100:i])
        y_test.append(input_scaled[i, 0])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    predictions  = model.predict(x_test, verbose=0)
    scale_factor = 1 / scaler.scale_[0]
    predictions  = predictions * scale_factor
    y_test_vals  = y_test * scale_factor

fig3, ax3 = styled_fig()
ax3.plot(y_test_vals,     color=PALETTE["actual"], linewidth=1.5, label="Actual Price",    alpha=0.9)
ax3.plot(predictions,     color=PALETTE["pred"],   linewidth=1.5, label="Predicted Price", alpha=0.9, linestyle="--")
ax3.fill_between(range(len(y_test_vals)), y_test_vals, predictions.flatten(),
                 alpha=0.06, color="#94a3b8")
ax3.set_xlabel("Time Step")
ax3.set_ylabel("Price (USD)")
ax3.legend()

ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
ax3.spines[["top","right"]].set_visible(False)

st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
st.pyplot(fig3, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
plt.close(fig3)

# ─────────────────────────────────────────────
#  PREDICTION METRICS
# ─────────────────────────────────────────────
actual_last = float(y_test_vals[-1])
pred_last   = float(predictions[-1][0])
diff        = pred_last - actual_last
diff_pct    = (diff / actual_last) * 100

signal = "🟢 BULLISH" if diff > 0 else ("🔴 BEARISH" if diff < 0 else "⚪ NEUTRAL")
sig_color = "#4ade80" if diff > 0 else ("#f87171" if diff < 0 else "#94a3b8")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Actual (Last)",    f"${actual_last:,.2f}")
col2.metric("Predicted (Last)", f"${pred_last:,.2f}")
col3.metric("Δ Difference",     f"${diff:+.2f}",  delta=f"{diff_pct:+.2f}%")
col4.metric("Model Signal",     signal)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown('<hr class="fancy-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#334155; font-size:0.8rem; padding-bottom:1rem;">
  StockSight AI &nbsp;·&nbsp; Built with Streamlit + TensorFlow + Yahoo Finance &nbsp;·&nbsp;
  <span style="color:#1e293b">Not financial advice</span>
</div>
""", unsafe_allow_html=True)
