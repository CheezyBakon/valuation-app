# Dynamic Valuation Model 

Enter any publicly traded ticker and get a full valuation in seconds.

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

## Usage

1. Enter any ticker in the sidebar
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

***Notes

- FCF is sourced directly from the cash flow statement. If unavailable, 50% EBITDA is used as a proxy (flagged in the UI).
- WACC is computed via CAPM with a 5.5% equity risk premium. Override via sidebar if you have a custom cost of capital.
- Comps are auto-selected by GICS sector. Override with any tickers in the sidebar.
- DDM is only meaningful for dividend-paying stocks. For non-payers, a reverse DDM shows the implied dividend required to justify the current price.
- All data is cached for 15 minutes (financials) and 60 minutes (risk-free rate). 

---

*For educational and research use only. Not investment advice.*
