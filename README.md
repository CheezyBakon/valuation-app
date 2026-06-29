# Dynamic Valuation Model 📊

A professional-grade equity valuation app built in Python + Streamlit.
Enter any publicly traded ticker and get a full institutional-quality valuation in seconds.

---

## Features

### Valuation Methods
- **DCF — 2-Stage** Standard 5-year explicit period + terminal value
- **DCF — 3-Stage** High growth → fade → terminal (Gordon Growth)
- **DCF — Monte Carlo** 10,000-path probabilistic simulation with P10/P50/P90 output
- **Comparable Company Analysis (Comps)** EV/EBITDA, EV/Revenue, P/E implied prices vs sector peers
- **Dividend Discount Model (DDM)** Gordon Growth + Multi-Stage for dividend-paying stocks

### Outputs
- **Football Field Chart** — banker-style range bar chart across all methods
- **Sensitivity Heatmap** — WACC × growth rate grid (color-coded vs current price)
- **Terminal Growth Sensitivity** — price vs terminal growth at 3 WACC levels
- **Monte Carlo Distribution** — histogram with current price overlay
- **Comps Scatter** — EV/EBITDA vs revenue growth peer positioning
- **WACC Decomposition** — CAPM cost of equity, after-tax debt cost, weights
- **Valuation Summary Table** — bear/base/bull across all methods with upside %

### Data Sources
- **Yahoo Finance** (`yfinance`) — price, financials, key statistics
- **FRED** — 10-year US Treasury rate (live, auto-refreshed)

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Usage

1. Enter any ticker (e.g. `AAPL`, `NVDA`, `JPM`, `NEE`, `AMZN`) in the sidebar
2. Select DCF model: 2-Stage / 3-Stage / Monte Carlo / All Three
3. Adjust growth, WACC, and terminal growth assumptions
4. Click **Run Valuation**
5. Navigate tabs: DCF → Comps → DDM → Football Field → Sensitivity

### Sidebar Options
| Setting | Description |
|---|---|
| Growth Yr 1–5 | Stage 1 FCF CAGR assumption |
| Terminal Growth | Long-run perpetuity growth rate (default 2.5%) |
| Fade Growth (3-Stage) | Mid-period growth between stage 1 and terminal |
| WACC Override | Set to 0 for auto-CAPM calculation |
| Projection Years | Explicit forecast horizon (5–10 years) |
| MC Growth Std Dev | Volatility of growth assumption in Monte Carlo |
| MC WACC Std Dev | Volatility of WACC in Monte Carlo |
| Simulations | Monte Carlo path count (1K–25K) |
| Custom Comp Tickers | Override auto-sector comps with your own list |

---

## Running in Google Colab

```python
# Install
!pip install streamlit yfinance plotly scipy -q

# Tunnel setup
!pip install pyngrok -q
from pyngrok import ngrok

# Run
import subprocess, threading
def run():
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])
t = threading.Thread(target=run); t.daemon = True; t.start()

public_url = ngrok.connect(8501)
print(f"Open: {public_url}")
```

---

## Quant Concepts Implemented

| Concept | Where Used |
|---|---|
| Discounted Cash Flow (DCF) | Tab 1 — all three models |
| CAPM (Capital Asset Pricing Model) | WACC → cost of equity |
| Geometric Brownian Motion | Monte Carlo FCF path simulation |
| Gordon Growth Model | Terminal value + DDM |
| Comparable Multiples | EV/EBITDA, EV/Revenue, P/E comps |
| Football Field Analysis | Tab 4 — range bar chart |
| Sensitivity Analysis | Tab 5 — 2D heatmap + terminal growth chart |
| Probability of Upside | Monte Carlo: % paths above current price |
| Mean Reversion (3-Stage) | Fade from high growth to terminal rate |

---

## Notes

- FCF is sourced directly from the cash flow statement. If unavailable, 50% EBITDA is used as a proxy (flagged in the UI).
- WACC is computed via CAPM with a 5.5% equity risk premium. Override via sidebar if you have a custom cost of capital.
- Comps are auto-selected by GICS sector. Override with any tickers in the sidebar.
- DDM is only meaningful for dividend-paying stocks. For non-payers, a reverse DDM shows the implied dividend required to justify the current price.
- All data is cached for 15 minutes (financials) and 60 minutes (risk-free rate). 

---

*For educational and research use only. Not investment advice.*
