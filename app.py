import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats

warnings.filterwarnings("ignore")

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dynamic Valuation Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0;}
  .sub-header {font-size: 0.95rem; color: #6b7280; margin-bottom: 1.5rem;}
  .metric-card {background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
                padding: 1rem 1.25rem; margin: 0.25rem 0;}
  .metric-label {font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em;}
  .metric-value {font-size: 1.5rem; font-weight: 700; color: #1a1a2e;}
  .metric-delta {font-size: 0.8rem;}
  .section-title {font-size: 1.1rem; font-weight: 600; color: #1a1a2e;
                  border-left: 3px solid #3b82f6; padding-left: 0.75rem; margin: 1.5rem 0 0.75rem;}
  .tag {display: inline-block; background: #eff6ff; color: #1d4ed8; border-radius: 4px;
        padding: 2px 8px; font-size: 0.75rem; font-weight: 500; margin: 2px;}
  .upside {color: #16a34a; font-weight: 600;}
  .downside {color: #dc2626; font-weight: 600;}
  .neutral {color: #6b7280; font-weight: 500;}
  div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0;
                                border-radius: 10px; padding: 0.75rem 1rem;}
  .stTabs [data-baseweb="tab"] {font-size: 0.85rem; font-weight: 500;}
  .stTabs [aria-selected="true"] {color: #3b82f6 !important;}
</style>
""", unsafe_allow_html=True)

# ─── FRED risk-free rate ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        df = pd.read_csv(url, parse_dates=["DATE"])
        df = df[df["DGS10"] != "."]
        return float(df["DGS10"].iloc[-1]) / 100
    except:
        return 0.043

# ─── Data fetching ───────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def fetch_company_data(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="5y")
    fin = t.financials
    cf = t.cashflow
    bs = t.balance_sheet
    return info, hist, fin, cf, bs

def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in [None, 0, "None", "N/A"]:
            return d[k]
    return default

def get_income_series(fin, label_options):
    for lbl in label_options:
        for col in (fin.index if fin is not None and not fin.empty else []):
            if lbl.lower() in col.lower():
                return fin.loc[col]
    return None

def extract_financials(info, fin, cf, bs):
    data = {}
    data["revenue"] = safe_get(info, "totalRevenue", default=0)
    data["ebitda"] = safe_get(info, "ebitda", default=0)
    data["net_income"] = safe_get(info, "netIncomeToCommon", default=0)
    data["total_debt"] = safe_get(info, "totalDebt", default=0)
    data["cash"] = safe_get(info, "totalCash", default=0)
    data["shares"] = safe_get(info, "sharesOutstanding", default=1)
    data["price"] = safe_get(info, "currentPrice", default=0)
    data["beta"] = safe_get(info, "beta", default=1.0)
    data["mkt_cap"] = safe_get(info, "marketCap", default=0)
    data["dividend"] = safe_get(info, "dividendRate", default=0) or 0
    data["payout_ratio"] = safe_get(info, "payoutRatio", default=0) or 0
    data["pe"] = safe_get(info, "trailingPE", default=None)
    data["fwd_pe"] = safe_get(info, "forwardPE", default=None)
    data["ev_ebitda"] = safe_get(info, "enterpriseToEbitda", default=None)
    data["ev_revenue"] = safe_get(info, "enterpriseToRevenue", default=None)
    data["roe"] = safe_get(info, "returnOnEquity", default=0) or 0
    data["roic"] = safe_get(info, "returnOnAssets", default=0) or 0
    data["sector"] = safe_get(info, "sector", default="Unknown")
    data["industry"] = safe_get(info, "industry", default="Unknown")
    data["name"] = safe_get(info, "longName", default=ticker)
    data["currency"] = safe_get(info, "currency", default="USD")
    data["country"] = safe_get(info, "country", default="")
    data["description"] = safe_get(info, "longBusinessSummary", default="")

    # Historical revenue growth
    rev_hist = []
    if fin is not None and not fin.empty:
        for col in fin.index:
            if "revenue" in col.lower() or "total revenue" in col.lower():
                series = fin.loc[col].dropna()
                rev_hist = series.values[::-1].tolist()
                break
    data["rev_hist"] = rev_hist

    # FCF
    fcf = 0
    if cf is not None and not cf.empty:
        for idx in cf.index:
            if "free cash flow" in idx.lower():
                vals = cf.loc[idx].dropna()
                if len(vals) > 0:
                    fcf = float(vals.iloc[0])
                    break
        if fcf == 0:
            ocf_val, capex_val = 0, 0
            for idx in cf.index:
                if "operating" in idx.lower() and "cash" in idx.lower():
                    vals = cf.loc[idx].dropna()
                    if len(vals) > 0:
                        ocf_val = float(vals.iloc[0])
                if "capital" in idx.lower() or "capex" in idx.lower():
                    vals = cf.loc[idx].dropna()
                    if len(vals) > 0:
                        capex_val = float(vals.iloc[0])
            fcf = ocf_val + capex_val  # capex usually negative

    data["fcf"] = fcf
    data["enterprise_value"] = safe_get(info, "enterpriseValue", default=data["mkt_cap"])
    return data

# ─── WACC calculation ─────────────────────────────────────────────────────────
def compute_wacc(info_data, rf):
    beta = max(info_data["beta"] or 1.0, 0.3)
    erp = 0.055
    cost_equity = rf + beta * erp
    total_debt = info_data["total_debt"] or 0
    mkt_cap = info_data["mkt_cap"] or 1
    tax_rate = 0.21
    cost_debt = safe_get({}, default=0.05)  # approx
    if "interestExpense" in str(info_data):
        pass
    cost_debt = max(rf + 0.015, 0.04)
    total_cap = mkt_cap + total_debt
    e_weight = mkt_cap / total_cap if total_cap > 0 else 1.0
    d_weight = total_debt / total_cap if total_cap > 0 else 0.0
    wacc = e_weight * cost_equity + d_weight * cost_debt * (1 - tax_rate)
    return round(wacc, 4), round(cost_equity, 4), round(cost_debt, 4), round(e_weight, 4), round(d_weight, 4)

# ─── Historical revenue growth ────────────────────────────────────────────────
def estimate_growth_from_history(rev_hist):
    if len(rev_hist) >= 2:
        growth_rates = []
        for i in range(1, len(rev_hist)):
            if rev_hist[i - 1] and rev_hist[i - 1] != 0:
                g = (rev_hist[i] - rev_hist[i - 1]) / abs(rev_hist[i - 1])
                growth_rates.append(g)
        if growth_rates:
            return np.median(growth_rates)
    return 0.08

# ─── DCF Engines ─────────────────────────────────────────────────────────────
def dcf_2stage(fcf, g1, wacc, terminal_g, years=5, debt=0, cash=0, shares=1):
    if shares == 0:
        return 0, 0, []
    fcfs = []
    pv_fcfs = []
    f = fcf
    for i in range(1, years + 1):
        f = f * (1 + g1)
        pv = f / (1 + wacc) ** i
        fcfs.append(f)
        pv_fcfs.append(pv)
    tv = fcfs[-1] * (1 + terminal_g) / (wacc - terminal_g) if wacc > terminal_g else 0
    pv_tv = tv / (1 + wacc) ** years
    ev = sum(pv_fcfs) + pv_tv
    equity_val = ev - debt + cash
    per_share = equity_val / shares if shares else 0
    return per_share, ev, pv_fcfs

def dcf_3stage(fcf, g1, g2, wacc, terminal_g, y1=3, y2=4, debt=0, cash=0, shares=1):
    fcfs, pv_fcfs = [], []
    f = fcf
    for i in range(1, y1 + 1):
        f = f * (1 + g1)
        pv = f / (1 + wacc) ** i
        fcfs.append(f)
        pv_fcfs.append(pv)
    # fade linearly g1 -> terminal_g over y2 years
    fade_rates = np.linspace(g1, g2, y2)
    for i, gr in enumerate(fade_rates, start=y1 + 1):
        f = f * (1 + gr)
        pv = f / (1 + wacc) ** i
        fcfs.append(f)
        pv_fcfs.append(pv)
    tv = fcfs[-1] * (1 + terminal_g) / (wacc - terminal_g) if wacc > terminal_g else 0
    pv_tv = tv / (1 + wacc) ** (y1 + y2)
    ev = sum(pv_fcfs) + pv_tv
    equity_val = ev - debt + cash
    per_share = equity_val / shares if shares else 0
    return per_share, ev, pv_fcfs

def dcf_monte_carlo(fcf, g_mean, g_std, wacc_mean, wacc_std, terminal_g, n=10000,
                    years=5, debt=0, cash=0, shares=1):
    np.random.seed(42)
    results = []
    for _ in range(n):
        g = np.random.normal(g_mean, g_std)
        w = max(np.random.normal(wacc_mean, wacc_std), terminal_g + 0.01)
        tg = np.random.normal(terminal_g, 0.005)
        tg = min(tg, w - 0.01)
        f = fcf
        pv_sum = 0
        for i in range(1, years + 1):
            f = f * (1 + g)
            pv_sum += f / (1 + w) ** i
        tv = f * (1 + tg) / (w - tg) if w > tg else 0
        pv_tv = tv / (1 + w) ** years
        ev = pv_sum + pv_tv
        eq = (ev - debt + cash) / shares if shares else 0
        results.append(eq)
    return np.array(results)

# ─── DDM ────────────────────────────────────────────────────────────────────
def ddm_gordon(div, g, cost_eq):
    if cost_eq <= g or div == 0:
        return None
    return div * (1 + g) / (cost_eq - g)

def ddm_multistage(div, g1, g2, cost_eq, years=5):
    if cost_eq <= g2 or div == 0:
        return None
    pvs = []
    d = div
    for i in range(1, years + 1):
        d = d * (1 + g1)
        pvs.append(d / (1 + cost_eq) ** i)
    tv = d * (1 + g2) / (cost_eq - g2)
    pv_tv = tv / (1 + cost_eq) ** years
    return sum(pvs) + pv_tv

# ─── Comps ───────────────────────────────────────────────────────────────────
SECTOR_COMPS = {
    "Technology": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "NKE", "MCD"],
    "Healthcare": ["JNJ", "UNH", "PFE", "ABBV", "MRK"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "T"],
    "Industrials": ["CAT", "BA", "HON", "UPS", "GE"],
    "Consumer Defensive": ["PG", "KO", "PEP", "WMT", "COST"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "Utilities": ["NEE", "DUK", "SO", "D", "EXC"],
    "Real Estate": ["AMT", "PLD", "CCI", "EQIX", "SPG"],
    "Basic Materials": ["LIN", "APD", "ECL", "NEM", "FCX"],
}

@st.cache_data(ttl=3600)
def fetch_comps_data(tickers, exclude):
    rows = []
    for t in tickers:
        if t == exclude.upper():
            continue
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker": t,
                "Name": info.get("shortName", t),
                "EV/EBITDA": info.get("enterpriseToEbitda"),
                "EV/Revenue": info.get("enterpriseToRevenue"),
                "P/E": info.get("trailingPE"),
                "Fwd P/E": info.get("forwardPE"),
                "Mkt Cap ($B)": round((info.get("marketCap") or 0) / 1e9, 1),
                "Rev Growth": info.get("revenueGrowth"),
                "EBITDA Margin": info.get("ebitdaMargins"),
            })
        except:
            pass
    return pd.DataFrame(rows)

def comps_implied_price(info_data, comps_df):
    results = {}
    ebitda = info_data["ebitda"]
    revenue = info_data["revenue"]
    net_inc = info_data["net_income"]
    debt = info_data["total_debt"]
    cash = info_data["cash"]
    shares = info_data["shares"]

    def ev_to_equity(ev):
        return (ev - debt + cash) / shares if shares else 0

    if ebitda and ebitda > 0:
        med_ev_ebitda = comps_df["EV/EBITDA"].dropna().median()
        if pd.notna(med_ev_ebitda):
            results["EV/EBITDA Comps"] = ev_to_equity(ebitda * med_ev_ebitda)

    if revenue and revenue > 0:
        med_ev_rev = comps_df["EV/Revenue"].dropna().median()
        if pd.notna(med_ev_rev):
            results["EV/Revenue Comps"] = ev_to_equity(revenue * med_ev_rev)

    if net_inc and net_inc > 0:
        med_pe = comps_df["P/E"].dropna().median()
        if pd.notna(med_pe):
            results["P/E Comps"] = (net_inc * med_pe) / shares if shares else 0

    return results

# ─── Sensitivity table ───────────────────────────────────────────────────────
def sensitivity_table(fcf, base_wacc, base_g, terminal_g, debt, cash, shares, years=5):
    wacc_range = np.arange(base_wacc - 0.02, base_wacc + 0.025, 0.005)
    g_range = np.arange(base_g - 0.04, base_g + 0.045, 0.01)
    rows = {}
    for g in g_range:
        row = {}
        for w in wacc_range:
            if w <= terminal_g:
                row[round(w, 3)] = np.nan
                continue
            price, _, _ = dcf_2stage(fcf, g, w, terminal_g, years, debt, cash, shares)
            row[round(w, 3)] = round(price, 2)
        rows[round(g, 3)] = row
    df = pd.DataFrame(rows).T
    df.index.name = "Growth Rate"
    df.columns.name = "WACC"
    return df

# ─── Formatting helpers ──────────────────────────────────────────────────────
def fmt_billions(v):
    if v is None or v == 0:
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v/1e12:.2f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.2f}M"
    return f"${v:,.0f}"

def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"

def fmt_x(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.1f}x"

def upside_color(implied, current):
    if not implied or not current or current == 0:
        return "neutral"
    pct = (implied - current) / current
    return "upside" if pct > 0.05 else ("downside" if pct < -0.05 else "neutral")

# ════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ════════════════════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">📊 Dynamic Valuation Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">DCF · Comparable Company Analysis · Dividend Discount Model · Football Field · Monte Carlo</p>', unsafe_allow_html=True)

# ─── Sidebar inputs ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Company")
    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. MSFT, NVDA, JPM").upper().strip()
    run_btn = st.button("▶  Run Valuation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### ⚙️ DCF Assumptions")
    dcf_scenario = st.selectbox("DCF Model", ["2-Stage", "3-Stage", "Monte Carlo", "All Three"])

    col1, col2 = st.columns(2)
    with col1:
        g1_pct = st.number_input("Growth Yr 1–5 (%)", value=10.0, step=0.5, format="%.1f")
    with col2:
        terminal_g_pct = st.number_input("Terminal Growth (%)", value=2.5, step=0.1, format="%.1f")

    if "3-Stage" in dcf_scenario or dcf_scenario == "All Three":
        g2_pct = st.number_input("Fade Growth Yr 6–10 (%)", value=5.0, step=0.5, format="%.1f")
    else:
        g2_pct = 5.0

    wacc_override = st.number_input("WACC Override (0 = auto)", value=0.0, step=0.1, format="%.1f")
    projection_years = st.slider("Projection Years", 5, 10, 5)

    if "Monte Carlo" in dcf_scenario or dcf_scenario == "All Three":
        st.markdown("**Monte Carlo Settings**")
        mc_g_std = st.slider("Growth Std Dev (%)", 1.0, 15.0, 5.0) / 100
        mc_wacc_std = st.slider("WACC Std Dev (%)", 0.5, 5.0, 1.5) / 100
        n_sims = st.select_slider("Simulations", options=[1000, 5000, 10000, 25000], value=10000)

    st.divider()
    st.markdown("### 🏷️ Comps Settings")
    manual_tickers = st.text_input("Custom Comp Tickers (comma-sep)", placeholder="e.g. MSFT,GOOGL,META")

    st.divider()
    st.markdown("### 📐 Display")
    show_raw = st.checkbox("Show raw financial data", value=False)

# ─── Main logic ─────────────────────────────────────────────────────────────
if not run_btn and "last_ticker" not in st.session_state:
    st.info("👈 Enter a ticker in the sidebar and click **Run Valuation** to begin.")
    st.stop()

if run_btn:
    st.session_state["last_ticker"] = ticker
else:
    ticker = st.session_state.get("last_ticker", "AAPL")

with st.spinner(f"Fetching data for **{ticker}**..."):
    try:
        info, hist, fin, cf, bs = fetch_company_data(ticker)
        rf = get_risk_free_rate()
        d = extract_financials(info, fin, cf, bs)
    except Exception as e:
        st.error(f"Could not fetch data for **{ticker}**. Check the ticker and try again. ({e})")
        st.stop()

if not d["price"] or d["price"] == 0:
    st.error(f"No price data found for **{ticker}**. Please verify the ticker.")
    st.stop()

# ─── Compute WACC & key rates ────────────────────────────────────────────────
wacc, cost_eq, cost_debt, e_wt, d_wt = compute_wacc(d, rf)
if wacc_override > 0:
    wacc = wacc_override / 100

g1 = g1_pct / 100
g2 = g2_pct / 100
terminal_g = terminal_g_pct / 100
terminal_g = min(terminal_g, wacc - 0.005)

hist_growth = estimate_growth_from_history(d["rev_hist"])

# ─── Company header ───────────────────────────────────────────────────────────
st.markdown(f"## {d['name']} &nbsp;<span style='font-size:1rem;color:#6b7280;'>({ticker})</span>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.markdown(f"<span class='tag'>{d['sector']}</span> <span class='tag'>{d['industry']}</span> <span class='tag'>{d['country']}</span>", unsafe_allow_html=True)
    if d["description"]:
        with st.expander("Business description"):
            st.write(d["description"][:800] + "...")

# ─── Key financials row ──────────────────────────────────────────────────────
st.markdown('<p class="section-title">Key Financials</p>', unsafe_allow_html=True)
cols = st.columns(7)
metrics = [
    ("Stock Price", f"${d['price']:,.2f}", ""),
    ("Market Cap", fmt_billions(d["mkt_cap"]), ""),
    ("Enterprise Value", fmt_billions(d["enterprise_value"]), ""),
    ("Revenue", fmt_billions(d["revenue"]), ""),
    ("EBITDA", fmt_billions(d["ebitda"]), ""),
    ("FCF", fmt_billions(d["fcf"]), ""),
    ("Net Income", fmt_billions(d["net_income"]), ""),
]
for col, (label, val, delta) in zip(cols, metrics):
    col.metric(label, val)

cols2 = st.columns(7)
metrics2 = [
    ("Beta", f"{d['beta']:.2f}" if d["beta"] else "N/A", ""),
    ("P/E (TTM)", fmt_x(d["pe"]), ""),
    ("Fwd P/E", fmt_x(d["fwd_pe"]), ""),
    ("EV/EBITDA", fmt_x(d["ev_ebitda"]), ""),
    ("EV/Revenue", fmt_x(d["ev_revenue"]), ""),
    ("ROE", fmt_pct(d["roe"]), ""),
    ("WACC", fmt_pct(wacc), ""),
]
for col, (label, val, _) in zip(cols2, metrics2):
    col.metric(label, val)

# ─── Valuation tabs ──────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Valuation Analysis</p>', unsafe_allow_html=True)

if d["fcf"] == 0 or d["fcf"] is None:
    fcf_base = d["ebitda"] * 0.5 if d["ebitda"] else d["net_income"] or 1e8
    st.warning("⚠️ FCF not directly available — using 50% EBITDA as proxy for FCF.")
else:
    fcf_base = d["fcf"]

debt = d["total_debt"] or 0
cash = d["cash"] or 0
shares = d["shares"] or 1
current_price = d["price"]

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 DCF Analysis", "🏢 Comps", "💰 DDM", "🎯 Football Field", "📊 Sensitivity"])

# ══════════════════════════════════════════════════════════
# TAB 1 — DCF
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    **Inputs used:** &nbsp; FCF = {fmt_billions(fcf_base)} &nbsp;|&nbsp;
    WACC = {fmt_pct(wacc)} &nbsp;|&nbsp;
    Stage 1 Growth = {fmt_pct(g1)} &nbsp;|&nbsp;
    Terminal Growth = {fmt_pct(terminal_g)} &nbsp;|&nbsp;
    Risk-Free Rate = {fmt_pct(rf)} (10Y UST) &nbsp;|&nbsp;
    Beta = {d["beta"]:.2f}
    """)

    all_dcf_results = {}

    # 2-Stage
    if dcf_scenario in ["2-Stage", "All Three"]:
        ps_2s, ev_2s, pv_fcfs_2s = dcf_2stage(fcf_base, g1, wacc, terminal_g,
                                               projection_years, debt, cash, shares)
        all_dcf_results["2-Stage DCF"] = ps_2s

        if dcf_scenario == "2-Stage":
            up_2s = (ps_2s - current_price) / current_price * 100 if current_price else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Implied Share Price", f"${ps_2s:,.2f}")
            c2.metric("Current Price", f"${current_price:,.2f}")
            c3.metric("Upside / Downside", f"{up_2s:+.1f}%",
                      delta_color="normal" if up_2s > 0 else "inverse")

            # FCF build chart
            years_list = [f"Yr {i+1}" for i in range(projection_years)]
            fig = go.Figure()
            fig.add_trace(go.Bar(name="PV of FCF", x=years_list, y=pv_fcfs_2s,
                                 marker_color="#3b82f6"))
            tv_pv = ps_2s * shares - sum(pv_fcfs_2s) + debt - cash
            fig.add_trace(go.Bar(name="PV of Terminal Value",
                                 x=[f"Terminal"],
                                 y=[max(tv_pv, 0)], marker_color="#8b5cf6"))
            fig.update_layout(title="DCF Value Build (PV of Cash Flows)",
                              barmode="group", height=350,
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(gridcolor="#f0f0f0"),
                              yaxis=dict(title="Value ($)", gridcolor="#f0f0f0",
                                         tickformat="$,.0f"))
            st.plotly_chart(fig, use_container_width=True)

    # 3-Stage
    if dcf_scenario in ["3-Stage", "All Three"]:
        ps_3s, ev_3s, pv_fcfs_3s = dcf_3stage(fcf_base, g1, g2, wacc, terminal_g,
                                               3, projection_years - 3, debt, cash, shares)
        all_dcf_results["3-Stage DCF"] = ps_3s

        if dcf_scenario == "3-Stage":
            up_3s = (ps_3s - current_price) / current_price * 100 if current_price else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Implied Share Price", f"${ps_3s:,.2f}")
            c2.metric("Current Price", f"${current_price:,.2f}")
            c3.metric("Upside / Downside", f"{up_3s:+.1f}%",
                      delta_color="normal" if up_3s > 0 else "inverse")

    # Monte Carlo
    if dcf_scenario in ["Monte Carlo", "All Three"]:
        mc_results = dcf_monte_carlo(
            fcf_base, g1, mc_g_std, wacc, mc_wacc_std, terminal_g,
            n=n_sims, years=projection_years, debt=debt, cash=cash, shares=shares
        )
        mc_results = mc_results[np.isfinite(mc_results)]
        mc_p10 = np.percentile(mc_results, 10)
        mc_p50 = np.percentile(mc_results, 50)
        mc_p90 = np.percentile(mc_results, 90)
        all_dcf_results["Monte Carlo (P50)"] = mc_p50
        all_dcf_results["Monte Carlo (P10)"] = mc_p10
        all_dcf_results["Monte Carlo (P90)"] = mc_p90

        if dcf_scenario == "Monte Carlo":
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("P10 (Bear)", f"${mc_p10:,.2f}")
            c2.metric("P50 (Base)", f"${mc_p50:,.2f}")
            c3.metric("P90 (Bull)", f"${mc_p90:,.2f}")
            c4.metric("Probability > Current",
                      f"{(mc_results > current_price).mean()*100:.0f}%")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=mc_results, nbinsx=80, name="Simulations",
            marker_color="#3b82f6", opacity=0.7
        ))
        fig_mc.add_vline(x=current_price, line_dash="dash", line_color="#dc2626",
                         annotation_text=f"Current: ${current_price:.2f}", annotation_position="top right")
        fig_mc.add_vline(x=mc_p50, line_dash="dot", line_color="#16a34a",
                         annotation_text=f"P50: ${mc_p50:.2f}", annotation_position="top left")
        fig_mc.update_layout(
            title=f"Monte Carlo DCF — {n_sims:,} Simulations",
            xaxis_title="Implied Share Price ($)", yaxis_title="Frequency",
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # All Three summary
    if dcf_scenario == "All Three":
        st.markdown("#### DCF Model Comparison")
        comp_data = {
            "Model": ["2-Stage DCF", "3-Stage DCF", "Monte Carlo P10", "Monte Carlo P50", "Monte Carlo P90"],
            "Implied Price": [all_dcf_results.get("2-Stage DCF", np.nan),
                              all_dcf_results.get("3-Stage DCF", np.nan),
                              mc_p10, mc_p50, mc_p90],
            "Upside (%)": []
        }
        for p in comp_data["Implied Price"]:
            if p and current_price:
                comp_data["Upside (%)"].append(f"{(p-current_price)/current_price*100:+.1f}%")
            else:
                comp_data["Upside (%)"].append("N/A")
        comp_data["Implied Price"] = [f"${p:,.2f}" if p else "N/A" for p in comp_data["Implied Price"]]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)

        fig_mc2 = go.Figure()
        fig_mc2.add_trace(go.Histogram(x=mc_results, nbinsx=80, marker_color="#3b82f6", opacity=0.65, name="MC Simulations"))
        fig_mc2.add_vline(x=current_price, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Current ${current_price:.2f}", annotation_position="top right")
        fig_mc2.add_vline(x=all_dcf_results.get("2-Stage DCF", 0), line_dash="solid",
                          line_color="#7c3aed", annotation_text="2-Stage")
        fig_mc2.add_vline(x=all_dcf_results.get("3-Stage DCF", 0), line_dash="solid",
                          line_color="#0891b2", annotation_text="3-Stage")
        fig_mc2.update_layout(title="All DCF Models vs Monte Carlo Distribution",
                              height=380, plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(title="Implied Share Price ($)", gridcolor="#f0f0f0"),
                              yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig_mc2, use_container_width=True)

    # WACC decomposition
    with st.expander("WACC Decomposition"):
        wd = pd.DataFrame({
            "Component": ["Cost of Equity", "After-tax Cost of Debt", "Blended WACC"],
            "Rate": [fmt_pct(cost_eq), fmt_pct(cost_debt * 0.79), fmt_pct(wacc)],
            "Weight": [fmt_pct(e_wt), fmt_pct(d_wt), "100%"],
            "Contribution": [fmt_pct(cost_eq * e_wt), fmt_pct(cost_debt * 0.79 * d_wt), fmt_pct(wacc)],
        })
        st.dataframe(wd, hide_index=True, use_container_width=True)
        st.caption(f"CAPM: Rf {fmt_pct(rf)} + Beta {d['beta']:.2f} × ERP 5.5% = Cost of Equity {fmt_pct(cost_eq)}")

# ══════════════════════════════════════════════════════════
# TAB 2 — COMPS
# ══════════════════════════════════════════════════════════
with tab2:
    sector = d["sector"]
    base_comps = SECTOR_COMPS.get(sector, ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
    if manual_tickers:
        extra = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()]
        base_comps = list(dict.fromkeys(extra + base_comps))

    with st.spinner("Fetching comparable companies..."):
        comps_df = fetch_comps_data(tuple(base_comps[:8]), ticker)

    if comps_df.empty:
        st.warning("Could not fetch comp data. Try custom tickers.")
    else:
        st.markdown(f"**Sector:** {sector} &nbsp;|&nbsp; **Comps Universe:** {', '.join(comps_df['Ticker'].tolist())}")

        # Display comps table
        display_cols = ["Ticker", "Name", "Mkt Cap ($B)", "EV/EBITDA", "EV/Revenue", "P/E", "Fwd P/E", "EBITDA Margin", "Rev Growth"]
        display_df = comps_df[display_cols].copy()
        for col in ["EV/EBITDA", "EV/Revenue", "P/E", "Fwd P/E"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
        for col in ["EBITDA Margin", "Rev Growth"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")

        # Add subject company row
        subj_row = pd.DataFrame([{
            "Ticker": f"★ {ticker}",
            "Name": d["name"][:20],
            "Mkt Cap ($B)": round((d["mkt_cap"] or 0) / 1e9, 1),
            "EV/EBITDA": fmt_x(d["ev_ebitda"]),
            "EV/Revenue": fmt_x(d["ev_revenue"]),
            "P/E": fmt_x(d["pe"]),
            "Fwd P/E": fmt_x(d["fwd_pe"]),
            "EBITDA Margin": fmt_pct(d["ebitda"] / d["revenue"]) if d["revenue"] else "N/A",
            "Rev Growth": fmt_pct(hist_growth),
        }])
        full_table = pd.concat([display_df, subj_row], ignore_index=True)
        st.dataframe(full_table, use_container_width=True, hide_index=True)

        # Comps-implied prices
        comps_prices = comps_implied_price(d, comps_df)
        if comps_prices:
            st.markdown("#### Comps-Implied Share Price")
            cp_cols = st.columns(len(comps_prices))
            for col, (method, price) in zip(cp_cols, comps_prices.items()):
                upside_pct = (price - current_price) / current_price * 100
                col.metric(method, f"${price:,.2f}", f"{upside_pct:+.1f}%",
                           delta_color="normal" if upside_pct > 0 else "inverse")

        # Multiple scatter
        fig_comps = go.Figure()
        for _, row in comps_df.iterrows():
            fig_comps.add_trace(go.Scatter(
                x=[row["Rev Growth"] * 100 if pd.notna(row["Rev Growth"]) else None],
                y=[row["EV/EBITDA"] if pd.notna(row["EV/EBITDA"]) else None],
                mode="markers+text",
                text=[row["Ticker"]],
                textposition="top center",
                marker=dict(size=row["Mkt Cap ($B)"] ** 0.4 * 4 if row["Mkt Cap ($B)"] else 8,
                            color="#3b82f6", opacity=0.7),
                name=row["Ticker"], showlegend=False
            ))
        if d["ev_ebitda"]:
            fig_comps.add_trace(go.Scatter(
                x=[hist_growth * 100], y=[d["ev_ebitda"]],
                mode="markers+text", text=[f"★ {ticker}"],
                textposition="top center",
                marker=dict(size=14, color="#dc2626", symbol="star"),
                name=ticker, showlegend=False
            ))
        fig_comps.update_layout(
            title="EV/EBITDA vs Revenue Growth — Comps Scatter",
            xaxis_title="Revenue Growth (%)", yaxis_title="EV/EBITDA (x)",
            height=400, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig_comps, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — DDM
# ══════════════════════════════════════════════════════════
with tab3:
    div = d["dividend"] or 0
    if div == 0:
        st.warning(f"**{ticker}** does not pay a dividend — DDM not applicable. Showing implied required dividend for target prices.")
        # Reverse DDM: what dividend would justify current price?
        implied_div = current_price * (cost_eq - terminal_g) / (1 + terminal_g)
        st.metric("Implied Annual Dividend (for fair value at current price)", f"${implied_div:.2f}/share")
        st.info("DDM works best for dividend-paying stocks (utilities, banks, consumer staples, REITs).")
    else:
        st.markdown(f"**Annual Dividend:** ${div:.2f} &nbsp;|&nbsp; **Payout Ratio:** {fmt_pct(d['payout_ratio'])} &nbsp;|&nbsp; **Cost of Equity:** {fmt_pct(cost_eq)}")

        gordon_val = ddm_gordon(div, terminal_g, cost_eq)
        ms_val = ddm_multistage(div, g1, terminal_g, cost_eq, 5)

        c1, c2, c3 = st.columns(3)
        if gordon_val:
            up = (gordon_val - current_price) / current_price * 100
            c1.metric("Gordon Growth Model", f"${gordon_val:,.2f}", f"{up:+.1f}%",
                      delta_color="normal" if up > 0 else "inverse")
        if ms_val:
            up2 = (ms_val - current_price) / current_price * 100
            c2.metric("Multi-Stage DDM", f"${ms_val:,.2f}", f"{up2:+.1f}%",
                      delta_color="normal" if up2 > 0 else "inverse")
        c3.metric("Current Price", f"${current_price:,.2f}", "")

        # Sensitivity: DDM vs required return
        ke_range = np.arange(cost_eq - 0.03, cost_eq + 0.035, 0.005)
        ddm_vals = [ddm_gordon(div, terminal_g, ke) or 0 for ke in ke_range]
        fig_ddm = go.Figure()
        fig_ddm.add_trace(go.Scatter(x=ke_range * 100, y=ddm_vals, mode="lines+markers",
                                     line=dict(color="#3b82f6", width=2), name="DDM Value"))
        fig_ddm.add_hline(y=current_price, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Current: ${current_price:.2f}")
        fig_ddm.update_layout(
            title="DDM Implied Value vs Required Return (Cost of Equity)",
            xaxis_title="Required Return (%)", yaxis_title="Implied Price ($)",
            height=350, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0")
        )
        st.plotly_chart(fig_ddm, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — FOOTBALL FIELD
# ══════════════════════════════════════════════════════════
with tab4:
    bars = []

    # DCF ranges
    if "2-Stage DCF" in all_dcf_results:
        ps = all_dcf_results["2-Stage DCF"]
        bear = ps * 0.80; bull = ps * 1.20
        bars.append(("2-Stage DCF", bear, bull, ps))

    if "3-Stage DCF" in all_dcf_results:
        ps = all_dcf_results["3-Stage DCF"]
        bear = ps * 0.80; bull = ps * 1.20
        bars.append(("3-Stage DCF", bear, bull, ps))

    if "Monte Carlo (P10)" in all_dcf_results:
        bars.append(("Monte Carlo DCF",
                     all_dcf_results["Monte Carlo (P10)"],
                     all_dcf_results["Monte Carlo (P90)"],
                     all_dcf_results["Monte Carlo (P50)"]))

    # Comps ranges
    if not comps_df.empty:
        comps_prices_ff = comps_implied_price(d, comps_df)
        ev_ebitda_vals = comps_df["EV/EBITDA"].dropna()
        if len(ev_ebitda_vals) >= 2 and d["ebitda"] and d["ebitda"] > 0:
            def ev_to_eq(ev):
                return (ev - debt + cash) / shares if shares else 0
            bars.append(("EV/EBITDA Comps",
                         ev_to_eq(d["ebitda"] * ev_ebitda_vals.quantile(0.25)),
                         ev_to_eq(d["ebitda"] * ev_ebitda_vals.quantile(0.75)),
                         ev_to_eq(d["ebitda"] * ev_ebitda_vals.median())))
        ev_rev_vals = comps_df["EV/Revenue"].dropna()
        if len(ev_rev_vals) >= 2 and d["revenue"] and d["revenue"] > 0:
            bars.append(("EV/Revenue Comps",
                         ev_to_eq(d["revenue"] * ev_rev_vals.quantile(0.25)),
                         ev_to_eq(d["revenue"] * ev_rev_vals.quantile(0.75)),
                         ev_to_eq(d["revenue"] * ev_rev_vals.median())))
        pe_vals = comps_df["P/E"].dropna()
        if len(pe_vals) >= 2 and d["net_income"] and d["net_income"] > 0:
            bars.append(("P/E Comps",
                         d["net_income"] * pe_vals.quantile(0.25) / shares,
                         d["net_income"] * pe_vals.quantile(0.75) / shares,
                         d["net_income"] * pe_vals.median() / shares))

    # DDM
    if div > 0 and gordon_val:
        bars.append(("DDM (Gordon Growth)", gordon_val * 0.85, gordon_val * 1.15, gordon_val))

    if not bars:
        st.warning("Not enough data to render football field.")
    else:
        # Filter out nonsensical values
        bars = [(name, lo, hi, mid) for name, lo, hi, mid in bars
                if lo > 0 and hi > 0 and hi < current_price * 10 and lo < current_price * 10]

        fig_ff = go.Figure()
        colors = ["#3b82f6", "#6366f1", "#8b5cf6", "#0891b2", "#059669",
                  "#d97706", "#dc2626", "#db2777"]

        for i, (name, lo, hi, mid) in enumerate(bars):
            color = colors[i % len(colors)]
            fig_ff.add_trace(go.Bar(
                name=name,
                x=[hi - lo],
                y=[name],
                base=[lo],
                orientation="h",
                marker=dict(color=color, opacity=0.7),
                text=f"${lo:,.1f} – ${hi:,.1f}",
                textposition="inside",
                insidetextanchor="middle",
                hovertemplate=f"<b>{name}</b><br>Low: ${lo:,.2f}<br>Mid: ${mid:,.2f}<br>High: ${hi:,.2f}<extra></extra>"
            ))
            # Midpoint marker
            fig_ff.add_trace(go.Scatter(
                x=[mid], y=[name], mode="markers",
                marker=dict(color="white", size=10, line=dict(color=color, width=2)),
                showlegend=False, hoverinfo="skip"
            ))

        fig_ff.add_vline(x=current_price, line_dash="solid", line_color="#dc2626", line_width=2,
                         annotation_text=f"Current: ${current_price:.2f}",
                         annotation_position="top right",
                         annotation=dict(font=dict(color="#dc2626", size=12)))

        all_lo = [b[1] for b in bars]; all_hi = [b[2] for b in bars]
        x_min = max(0, min(all_lo) * 0.85)
        x_max = max(all_hi) * 1.15

        fig_ff.update_layout(
            title=f"{d['name']} ({ticker}) — Valuation Football Field",
            xaxis=dict(title="Implied Share Price ($)", range=[x_min, x_max],
                       tickprefix="$", gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(autorange="reversed"),
            height=max(350, 80 + len(bars) * 60),
            plot_bgcolor="white", paper_bgcolor="white",
            barmode="overlay", showlegend=False,
            margin=dict(l=160, r=40, t=60, b=60)
        )
        st.plotly_chart(fig_ff, use_container_width=True)

        # Summary table
        st.markdown("#### Implied Price Summary")
        summary = []
        for name, lo, hi, mid in bars:
            upside_pct = (mid - current_price) / current_price * 100
            summary.append({
                "Method": name,
                "Bear Case": f"${lo:,.2f}",
                "Base Case": f"${mid:,.2f}",
                "Bull Case": f"${hi:,.2f}",
                "Upside to Base": f"{upside_pct:+.1f}%",
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════
# TAB 5 — SENSITIVITY
# ══════════════════════════════════════════════════════════
with tab5:
    st.markdown("#### DCF Sensitivity: Implied Price by WACC & Growth Rate")
    st.caption("Base values highlighted. Green = above current price, Red = below.")

    sens_df = sensitivity_table(fcf_base, wacc, g1, terminal_g, debt, cash, shares, projection_years)

    # Plotly heatmap
    z_vals = sens_df.values.astype(float)
    x_labels = [f"{v*100:.1f}%" for v in sens_df.columns]
    y_labels = [f"{v*100:.1f}%" for v in sens_df.index]

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_vals, x=x_labels, y=y_labels,
        colorscale=[
            [0.0, "#dc2626"],
            [0.3, "#fca5a5"],
            [0.5, "#fef3c7"],
            [0.7, "#86efac"],
            [1.0, "#16a34a"]
        ],
        zmid=current_price,
        text=[[f"${v:.0f}" if np.isfinite(v) else "N/A" for v in row] for row in z_vals],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Growth: %{y}<br>WACC: %{x}<br>Price: %{text}<extra></extra>"
    ))
    fig_heat.update_layout(
        xaxis_title="WACC", yaxis_title="FCF Growth Rate (Yr 1–5)",
        height=420, margin=dict(l=80, r=20, t=30, b=60)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # Terminal growth sensitivity
    st.markdown("#### Terminal Growth Rate Sensitivity")
    tg_range = np.arange(0.01, 0.04, 0.005)
    wacc_pts = [wacc - 0.01, wacc, wacc + 0.01]
    fig_tg = go.Figure()
    for w in wacc_pts:
        prices = []
        for tg in tg_range:
            if w <= tg:
                prices.append(np.nan)
                continue
            p, _, _ = dcf_2stage(fcf_base, g1, w, tg, projection_years, debt, cash, shares)
            prices.append(p)
        fig_tg.add_trace(go.Scatter(
            x=tg_range * 100, y=prices, mode="lines+markers",
            name=f"WACC = {w*100:.1f}%",
            line=dict(width=2)
        ))
    fig_tg.add_hline(y=current_price, line_dash="dash", line_color="#dc2626",
                     annotation_text=f"Current: ${current_price:.2f}")
    fig_tg.update_layout(
        xaxis_title="Terminal Growth Rate (%)", yaxis_title="Implied Price ($)",
        height=350, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig_tg, use_container_width=True)

    # Raw data expander
    if show_raw:
        with st.expander("Raw Sensitivity Table (2-Stage DCF)"):
            fmt_df = sens_df.copy()
            fmt_df.columns = [f"WACC {v*100:.1f}%" for v in fmt_df.columns]
            fmt_df.index = [f"Growth {v*100:.1f}%" for v in fmt_df.index]
            st.dataframe(fmt_df.style.format("${:.2f}"), use_container_width=True)

# ─── Footer ────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Data: Yahoo Finance · FRED (10Y UST: {rf*100:.2f}%) · "
    "For educational/research purposes only. Not investment advice. "
    f"Last refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}"
)
