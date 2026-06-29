import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from scipy import stats

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dynamic Valuation Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header {font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0;}
  .sub-header {font-size: 0.95rem; color: #6b7280; margin-bottom: 1.5rem;}
  .section-title {font-size: 1.05rem; font-weight: 600; color: #1a1a2e;
                  border-left: 3px solid #3b82f6; padding-left: 0.75rem;
                  margin: 1.5rem 0 0.75rem;}
  .tag {display: inline-block; background: #eff6ff; color: #1d4ed8; border-radius: 4px;
        padding: 2px 8px; font-size: 0.75rem; font-weight: 500; margin: 2px;}
  div[data-testid="stMetric"] {background: #f8fafc; border: 1px solid #e2e8f0;
                                border-radius: 10px; padding: 0.75rem 1rem;}
</style>
""", unsafe_allow_html=True)


# ── FRED risk-free rate ───────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        df = pd.read_csv(url, parse_dates=["DATE"])
        df = df[df["DGS10"] != "."]
        return float(df["DGS10"].iloc[-1]) / 100
    except Exception:
        return 0.043


# ── Data fetching ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=900)
def fetch_company_data(ticker: str):
    t = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="5y")
    fin  = t.financials
    cf   = t.cashflow
    bs   = t.balance_sheet
    return info, hist, fin, cf, bs


def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in [None, "None", "N/A"]:
            val = d[k]
            if isinstance(val, (int, float)) and val == 0:
                continue
            return val
    return default


def extract_financials(info, fin, cf, bs, ticker_sym):
    d = {}
    d["revenue"]          = info.get("totalRevenue") or 0
    d["ebitda"]           = info.get("ebitda") or 0
    d["net_income"]       = info.get("netIncomeToCommon") or 0
    d["total_debt"]       = info.get("totalDebt") or 0
    d["cash"]             = info.get("totalCash") or 0
    d["shares"]           = info.get("sharesOutstanding") or 1
    d["price"]            = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    d["beta"]             = info.get("beta") or 1.0
    d["mkt_cap"]          = info.get("marketCap") or 0
    d["dividend"]         = info.get("dividendRate") or 0
    d["payout_ratio"]     = info.get("payoutRatio") or 0
    d["pe"]               = info.get("trailingPE")
    d["fwd_pe"]           = info.get("forwardPE")
    d["ev_ebitda"]        = info.get("enterpriseToEbitda")
    d["ev_revenue"]       = info.get("enterpriseToRevenue")
    d["roe"]              = info.get("returnOnEquity") or 0
    d["sector"]           = info.get("sector") or "Unknown"
    d["industry"]         = info.get("industry") or "Unknown"
    d["name"]             = info.get("longName") or info.get("shortName") or ticker_sym
    d["currency"]         = info.get("currency") or "USD"
    d["country"]          = info.get("country") or ""
    d["description"]      = info.get("longBusinessSummary") or ""
    d["enterprise_value"] = info.get("enterpriseValue") or d["mkt_cap"]

    # Historical revenue for growth estimation
    rev_hist = []
    if fin is not None and not fin.empty:
        for idx in fin.index:
            if "revenue" in idx.lower():
                series = fin.loc[idx].dropna()
                rev_hist = list(series.values[::-1])
                break
    d["rev_hist"] = rev_hist

    # FCF: try direct line, fall back to OCF + capex
    fcf = 0
    if cf is not None and not cf.empty:
        for idx in cf.index:
            if "free cash flow" in idx.lower():
                vals = cf.loc[idx].dropna()
                if len(vals):
                    fcf = float(vals.iloc[0])
                break
        if fcf == 0:
            ocf, capex = 0, 0
            for idx in cf.index:
                if "operating" in idx.lower() and "cash" in idx.lower():
                    vals = cf.loc[idx].dropna()
                    if len(vals):
                        ocf = float(vals.iloc[0])
                if "capital" in idx.lower() or "capex" in idx.lower():
                    vals = cf.loc[idx].dropna()
                    if len(vals):
                        capex = float(vals.iloc[0])
            fcf = ocf + capex  # capex is typically negative

    d["fcf"] = fcf

    # Null-guard all numeric fields
    for key in ["revenue", "ebitda", "net_income", "total_debt", "cash",
                "mkt_cap", "enterprise_value", "fcf", "dividend", "payout_ratio",
                "roe", "beta"]:
        if d[key] is None:
            d[key] = 0

    return d


# ── WACC ──────────────────────────────────────────────────────────────────────
def compute_wacc(d, rf):
    beta      = max(float(d["beta"] or 1.0), 0.3)
    erp       = 0.055
    cost_eq   = rf + beta * erp
    total_dbt = d["total_debt"] or 0
    mkt_cap   = d["mkt_cap"] or 1
    cost_dbt  = max(rf + 0.015, 0.04)
    tax       = 0.21
    total_cap = mkt_cap + total_dbt
    ew        = mkt_cap   / total_cap if total_cap > 0 else 1.0
    dw        = total_dbt / total_cap if total_cap > 0 else 0.0
    wacc      = ew * cost_eq + dw * cost_dbt * (1 - tax)
    return round(wacc, 4), round(cost_eq, 4), round(cost_dbt, 4), round(ew, 4), round(dw, 4)


def estimate_growth(rev_hist):
    if len(rev_hist) >= 2:
        rates = []
        for i in range(1, len(rev_hist)):
            if rev_hist[i - 1] and rev_hist[i - 1] != 0:
                rates.append((rev_hist[i] - rev_hist[i - 1]) / abs(rev_hist[i - 1]))
        if rates:
            return np.median(rates)
    return 0.08


# ── DCF engines ───────────────────────────────────────────────────────────────
def dcf_2stage(fcf, g1, wacc, tg, years=5, debt=0, cash=0, shares=1):
    if shares == 0 or fcf == 0:
        return 0, 0, []
    fcfs, pvs = [], []
    f = fcf
    for i in range(1, years + 1):
        f = f * (1 + g1)
        pvs.append(f / (1 + wacc) ** i)
        fcfs.append(f)
    tv    = fcfs[-1] * (1 + tg) / (wacc - tg) if wacc > tg else 0
    pv_tv = tv / (1 + wacc) ** years
    ev    = sum(pvs) + pv_tv
    return (ev - debt + cash) / shares, ev, pvs


def dcf_3stage(fcf, g1, g2, wacc, tg, y1=3, y2=4, debt=0, cash=0, shares=1):
    if shares == 0 or fcf == 0:
        return 0, 0, []
    fcfs, pvs = [], []
    f = fcf
    for i in range(1, y1 + 1):
        f = f * (1 + g1)
        pvs.append(f / (1 + wacc) ** i)
        fcfs.append(f)
    for i, gr in enumerate(np.linspace(g1, g2, y2), start=y1 + 1):
        f = f * (1 + gr)
        pvs.append(f / (1 + wacc) ** i)
        fcfs.append(f)
    tv    = fcfs[-1] * (1 + tg) / (wacc - tg) if wacc > tg else 0
    pv_tv = tv / (1 + wacc) ** (y1 + y2)
    ev    = sum(pvs) + pv_tv
    return (ev - debt + cash) / shares, ev, pvs


def dcf_monte_carlo(fcf, g_mean, g_std, wacc_mean, wacc_std, tg,
                    n=10000, years=5, debt=0, cash=0, shares=1):
    np.random.seed(42)
    out = []
    for _ in range(n):
        g  = np.random.normal(g_mean, g_std)
        w  = max(np.random.normal(wacc_mean, wacc_std), tg + 0.01)
        tgr = min(np.random.normal(tg, 0.005), w - 0.01)
        f, pv_sum = fcf, 0
        for i in range(1, years + 1):
            f = f * (1 + g)
            pv_sum += f / (1 + w) ** i
        tv    = f * (1 + tgr) / (w - tgr) if w > tgr else 0
        pv_tv = tv / (1 + w) ** years
        ev    = pv_sum + pv_tv
        out.append((ev - debt + cash) / shares if shares else 0)
    return np.array(out)


# ── DDM ───────────────────────────────────────────────────────────────────────
def ddm_gordon(div, g, ke):
    if ke <= g or div == 0:
        return None
    return div * (1 + g) / (ke - g)


def ddm_multistage(div, g1, g2, ke, years=5):
    if ke <= g2 or div == 0:
        return None
    d, pvs = div, []
    for i in range(1, years + 1):
        d = d * (1 + g1)
        pvs.append(d / (1 + ke) ** i)
    tv    = d * (1 + g2) / (ke - g2)
    pv_tv = tv / (1 + ke) ** years
    return sum(pvs) + pv_tv


# ── Comps universe ────────────────────────────────────────────────────────────
SECTOR_COMPS = {
    "Technology":             ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "Consumer Cyclical":      ["AMZN", "TSLA", "HD",   "NKE",  "MCD"],
    "Healthcare":             ["JNJ",  "UNH",  "PFE",  "ABBV", "MRK"],
    "Financial Services":     ["JPM",  "BAC",  "WFC",  "GS",   "MS"],
    "Communication Services": ["GOOGL","META", "NFLX", "DIS",  "T"],
    "Industrials":            ["CAT",  "BA",   "HON",  "UPS",  "GE"],
    "Consumer Defensive":     ["PG",   "KO",   "PEP",  "WMT",  "COST"],
    "Energy":                 ["XOM",  "CVX",  "COP",  "SLB",  "EOG"],
    "Utilities":              ["NEE",  "DUK",  "SO",   "D",    "EXC"],
    "Real Estate":            ["AMT",  "PLD",  "CCI",  "EQIX", "SPG"],
    "Basic Materials":        ["LIN",  "APD",  "ECL",  "NEM",  "FCX"],
}


@st.cache_data(ttl=3600)
def fetch_comps(tickers, exclude):
    rows = []
    for t in tickers:
        if t == exclude.upper():
            continue
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker":       t,
                "Name":         info.get("shortName", t),
                "EV/EBITDA":    info.get("enterpriseToEbitda"),
                "EV/Revenue":   info.get("enterpriseToRevenue"),
                "P/E":          info.get("trailingPE"),
                "Fwd P/E":      info.get("forwardPE"),
                "Mkt Cap ($B)": round((info.get("marketCap") or 0) / 1e9, 1),
                "Rev Growth":   info.get("revenueGrowth"),
                "EBITDA Margin":info.get("ebitdaMargins"),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def comps_implied(d, comps_df):
    results = {}
    debt, cash, shares = d["total_debt"], d["cash"], d["shares"]

    def ev2eq(ev):
        return (ev - debt + cash) / shares if shares else 0

    if d["ebitda"] and d["ebitda"] > 0:
        med = comps_df["EV/EBITDA"].dropna().median()
        if pd.notna(med):
            results["EV/EBITDA Comps"] = ev2eq(d["ebitda"] * med)

    if d["revenue"] and d["revenue"] > 0:
        med = comps_df["EV/Revenue"].dropna().median()
        if pd.notna(med):
            results["EV/Revenue Comps"] = ev2eq(d["revenue"] * med)

    if d["net_income"] and d["net_income"] > 0:
        med = comps_df["P/E"].dropna().median()
        if pd.notna(med):
            results["P/E Comps"] = (d["net_income"] * med) / shares if shares else 0

    return results


# ── Sensitivity grid ──────────────────────────────────────────────────────────
def sensitivity_table(fcf, base_wacc, base_g, tg, debt, cash, shares, years):
    wacc_range = np.arange(base_wacc - 0.02, base_wacc + 0.025, 0.005)
    g_range    = np.arange(base_g    - 0.04, base_g    + 0.045, 0.010)
    rows = {}
    for g in g_range:
        row = {}
        for w in wacc_range:
            if w <= tg:
                row[round(w, 3)] = np.nan
            else:
                p, _, _ = dcf_2stage(fcf, g, w, tg, years, debt, cash, shares)
                row[round(w, 3)] = round(p, 2)
        rows[round(g, 3)] = row
    df = pd.DataFrame(rows).T
    df.index.name   = "Growth Rate"
    df.columns.name = "WACC"
    return df


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt_B(v):
    if v is None:
        return "N/A"
    sign = "-$" if v < 0 else "$"
    av = abs(v)
    if av >= 1e12: return f"{sign}{av/1e12:.2f}T"
    if av >= 1e9:  return f"{sign}{av/1e9:.1f}B"
    if av >= 1e6:  return f"{sign}{av/1e6:.0f}M"
    return f"{sign}{av:,.0f}"

def fmt_pct(v):
    return "N/A" if v is None else f"{v*100:.1f}%"

def fmt_x(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.1f}x"


# ═════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ═════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">Dynamic Valuation Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">DCF · Comparable Company Analysis · Dividend Discount Model · Football Field · Monte Carlo</p>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Company")
    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. MSFT, NVDA, JPM").upper().strip()
    run_btn = st.button("Run Valuation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### DCF Assumptions")
    dcf_scenario = st.selectbox("DCF Model", ["2-Stage", "3-Stage", "Monte Carlo", "All Three"])

    col1, col2 = st.columns(2)
    with col1:
        g1_pct = st.number_input("Growth Yr 1-5 (%)", value=10.0, step=0.5, format="%.1f")
    with col2:
        terminal_g_pct = st.number_input("Terminal Growth (%)", value=2.5, step=0.1, format="%.1f")

    if "3-Stage" in dcf_scenario or dcf_scenario == "All Three":
        g2_pct = st.number_input("Fade Growth Yr 6-10 (%)", value=5.0, step=0.5, format="%.1f")
    else:
        g2_pct = 5.0

    wacc_override    = st.number_input("WACC Override (0 = auto)", value=0.0, step=0.1, format="%.1f")
    projection_years = st.slider("Projection Years", 5, 10, 5)

    # Monte Carlo defaults always defined
    mc_g_std    = 0.05
    mc_wacc_std = 0.015
    n_sims      = 10000

    if "Monte Carlo" in dcf_scenario or dcf_scenario == "All Three":
        st.markdown("**Monte Carlo Settings**")
        mc_g_std    = st.slider("Growth Std Dev (%)", 1.0, 15.0, 5.0) / 100
        mc_wacc_std = st.slider("WACC Std Dev (%)", 0.5, 5.0, 1.5) / 100
        n_sims      = st.select_slider("Simulations", options=[1000, 5000, 10000, 25000], value=10000)

    st.divider()
    st.markdown("### Comps Settings")
    manual_tickers = st.text_input("Custom Comp Tickers (comma-sep)", placeholder="e.g. MSFT,GOOGL,META")

    st.divider()
    show_raw = st.checkbox("Show raw sensitivity table", value=False)


# ── Gate ─────────────────────────────────────────────────────────────────────
if not run_btn and "last_ticker" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click Run Valuation to begin.")
    st.stop()

if run_btn:
    st.session_state["last_ticker"] = ticker
else:
    ticker = st.session_state.get("last_ticker", "AAPL")


# ── Fetch ─────────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker}..."):
    try:
        info, hist, fin, cf, bs = fetch_company_data(ticker)
        rf = get_risk_free_rate()
        d  = extract_financials(info, fin, cf, bs, ticker)
    except Exception as e:
        st.error(f"Could not fetch data for {ticker}. Verify the ticker and try again. ({e})")
        st.stop()

if not d["price"] or d["price"] == 0:
    st.error(f"No price data found for {ticker}. Please verify the ticker.")
    st.stop()


# ── Derived values ────────────────────────────────────────────────────────────
wacc, cost_eq, cost_dbt, e_wt, d_wt = compute_wacc(d, rf)
if wacc_override > 0:
    wacc = wacc_override / 100

g1       = g1_pct / 100
g2       = g2_pct / 100
terminal_g = min(terminal_g_pct / 100, wacc - 0.005)

hist_growth  = estimate_growth(d["rev_hist"])
current_price = d["price"]

# FCF base — guard against zero
if d["fcf"] and d["fcf"] != 0:
    fcf_base = d["fcf"]
    fcf_note = None
elif d["ebitda"] and d["ebitda"] != 0:
    fcf_base = d["ebitda"] * 0.5
    fcf_note = "FCF not available from cash flow statement — using 50% of EBITDA as proxy."
elif d["net_income"] and d["net_income"] != 0:
    fcf_base = d["net_income"]
    fcf_note = "FCF and EBITDA not available — using Net Income as proxy."
else:
    fcf_base = 1e8
    fcf_note = "Could not derive FCF from available data. Using $100M placeholder — adjust assumptions carefully."

debt   = d["total_debt"] or 0
cash   = d["cash"]       or 0
shares = d["shares"]     or 1


# ── Company header ────────────────────────────────────────────────────────────
st.markdown(
    f"## {d['name']} "
    f"<span style='font-size:1rem;color:#6b7280;'>({ticker})</span>",
    unsafe_allow_html=True
)
st.markdown(
    f"<span class='tag'>{d['sector']}</span> "
    f"<span class='tag'>{d['industry']}</span> "
    f"<span class='tag'>{d['country']}</span>",
    unsafe_allow_html=True
)
if d["description"]:
    with st.expander("Business description"):
        st.write(d["description"][:900] + "...")

if fcf_note:
    st.warning(fcf_note)


# ── Key financials ────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Key Financials</p>', unsafe_allow_html=True)

row1 = st.columns(7)
for col, (label, val) in zip(row1, [
    ("Stock Price",       f"${current_price:,.2f}"),
    ("Market Cap",        fmt_B(d["mkt_cap"])),
    ("Enterprise Value",  fmt_B(d["enterprise_value"])),
    ("Revenue",           fmt_B(d["revenue"])),
    ("EBITDA",            fmt_B(d["ebitda"])),
    ("FCF",               fmt_B(d["fcf"]) if d["fcf"] else "N/A"),
    ("Net Income",        fmt_B(d["net_income"])),
]):
    col.metric(label, val)

row2 = st.columns(7)
for col, (label, val) in zip(row2, [
    ("Beta",        f"{float(d['beta']):.2f}"),
    ("P/E (TTM)",   fmt_x(d["pe"])),
    ("Fwd P/E",     fmt_x(d["fwd_pe"])),
    ("EV/EBITDA",   fmt_x(d["ev_ebitda"])),
    ("EV/Revenue",  fmt_x(d["ev_revenue"])),
    ("ROE",         fmt_pct(d["roe"])),
    ("WACC",        fmt_pct(wacc)),
]):
    col.metric(label, val)


# ── Valuation tabs ────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">Valuation Analysis</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "DCF Analysis", "Comps", "DDM", "Football Field", "Sensitivity"
])


# ═══════════════════════════════════════════
# TAB 1 — DCF
# ═══════════════════════════════════════════
with tab1:
    st.markdown(
        f"**FCF Base:** {fmt_B(fcf_base)} &nbsp;|&nbsp; "
        f"**WACC:** {fmt_pct(wacc)} &nbsp;|&nbsp; "
        f"**Stage 1 Growth:** {fmt_pct(g1)} &nbsp;|&nbsp; "
        f"**Terminal Growth:** {fmt_pct(terminal_g)} &nbsp;|&nbsp; "
        f"**Risk-Free Rate:** {fmt_pct(rf)} (10Y UST) &nbsp;|&nbsp; "
        f"**Beta:** {float(d['beta']):.2f}"
    )

    all_dcf = {}

    # 2-Stage
    if dcf_scenario in ["2-Stage", "All Three"]:
        ps_2s, ev_2s, pvs_2s = dcf_2stage(
            fcf_base, g1, wacc, terminal_g, projection_years, debt, cash, shares
        )
        all_dcf["2-Stage DCF"] = ps_2s

        if dcf_scenario == "2-Stage":
            up = (ps_2s - current_price) / current_price * 100 if current_price else 0
            ca, cb, cc = st.columns(3)
            ca.metric("Implied Share Price", f"${ps_2s:,.2f}")
            cb.metric("Current Price",       f"${current_price:,.2f}")
            cc.metric("Upside / Downside",   f"{up:+.1f}%",
                      delta_color="normal" if up > 0 else "inverse")

            yr_labels = [f"Yr {i+1}" for i in range(projection_years)]
            tv_pv = max((ps_2s * shares) - sum(pvs_2s) + debt - cash, 0)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="PV of FCF", x=yr_labels, y=pvs_2s,
                                 marker_color="#3b82f6"))
            fig.add_trace(go.Bar(name="PV of Terminal Value", x=["Terminal"],
                                 y=[tv_pv], marker_color="#8b5cf6"))
            fig.update_layout(
                title="DCF Value Build — PV of Cash Flows",
                barmode="group", height=350,
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#f0f0f0"),
                yaxis=dict(title="Value ($)", gridcolor="#f0f0f0", tickformat="$,.0f"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # 3-Stage
    if dcf_scenario in ["3-Stage", "All Three"]:
        ps_3s, ev_3s, pvs_3s = dcf_3stage(
            fcf_base, g1, g2, wacc, terminal_g,
            3, max(projection_years - 3, 1), debt, cash, shares
        )
        all_dcf["3-Stage DCF"] = ps_3s

        if dcf_scenario == "3-Stage":
            up = (ps_3s - current_price) / current_price * 100 if current_price else 0
            ca, cb, cc = st.columns(3)
            ca.metric("Implied Share Price", f"${ps_3s:,.2f}")
            cb.metric("Current Price",       f"${current_price:,.2f}")
            cc.metric("Upside / Downside",   f"{up:+.1f}%",
                      delta_color="normal" if up > 0 else "inverse")

    # Monte Carlo
    mc_results = None
    mc_p10 = mc_p50 = mc_p90 = 0.0

    if dcf_scenario in ["Monte Carlo", "All Three"]:
        raw = dcf_monte_carlo(
            fcf_base, g1, mc_g_std, wacc, mc_wacc_std, terminal_g,
            n=n_sims, years=projection_years, debt=debt, cash=cash, shares=shares
        )
        mc_results = raw[np.isfinite(raw)]
        mc_p10  = float(np.percentile(mc_results, 10))
        mc_p50  = float(np.percentile(mc_results, 50))
        mc_p90  = float(np.percentile(mc_results, 90))
        all_dcf["Monte Carlo (P10)"] = mc_p10
        all_dcf["Monte Carlo (P50)"] = mc_p50
        all_dcf["Monte Carlo (P90)"] = mc_p90

        if dcf_scenario == "Monte Carlo":
            ca, cb, cc, cd = st.columns(4)
            ca.metric("P10 — Bear",           f"${mc_p10:,.2f}")
            cb.metric("P50 — Base",           f"${mc_p50:,.2f}")
            cc.metric("P90 — Bull",           f"${mc_p90:,.2f}")
            cd.metric("Probability > Current",
                      f"{(mc_results > current_price).mean()*100:.0f}%")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(
            x=mc_results, nbinsx=80, name="Simulations",
            marker_color="#3b82f6", opacity=0.7
        ))
        fig_mc.add_vline(x=current_price, line_dash="dash", line_color="#dc2626",
                         annotation_text=f"Current: ${current_price:.2f}",
                         annotation_position="top right")
        fig_mc.add_vline(x=mc_p50, line_dash="dot", line_color="#16a34a",
                         annotation_text=f"P50: ${mc_p50:.2f}",
                         annotation_position="top left")
        fig_mc.update_layout(
            title=f"Monte Carlo DCF — {n_sims:,} Simulations",
            xaxis_title="Implied Share Price ($)", yaxis_title="Frequency",
            height=380, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_mc, use_container_width=True)

    # All Three summary
    if dcf_scenario == "All Three":
        st.markdown("#### DCF Model Comparison")
        rows_cmp = []
        for model, price in [
            ("2-Stage DCF",       all_dcf.get("2-Stage DCF")),
            ("3-Stage DCF",       all_dcf.get("3-Stage DCF")),
            ("Monte Carlo P10",   mc_p10),
            ("Monte Carlo P50",   mc_p50),
            ("Monte Carlo P90",   mc_p90),
        ]:
            if price is not None and current_price:
                up_str = f"{(price - current_price)/current_price*100:+.1f}%"
            else:
                up_str = "N/A"
            rows_cmp.append({
                "Model":         model,
                "Implied Price": f"${price:,.2f}" if price else "N/A",
                "Upside (%)":    up_str,
            })
        st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)

        if mc_results is not None:
            fig_all = go.Figure()
            fig_all.add_trace(go.Histogram(
                x=mc_results, nbinsx=80, marker_color="#3b82f6", opacity=0.65,
                name="MC Simulations"
            ))
            fig_all.add_vline(x=current_price, line_dash="dash", line_color="#dc2626",
                              annotation_text=f"Current ${current_price:.2f}",
                              annotation_position="top right")
            if all_dcf.get("2-Stage DCF"):
                fig_all.add_vline(x=all_dcf["2-Stage DCF"], line_dash="solid",
                                  line_color="#7c3aed", annotation_text="2-Stage",
                                  annotation_position="top left")
            if all_dcf.get("3-Stage DCF"):
                fig_all.add_vline(x=all_dcf["3-Stage DCF"], line_dash="solid",
                                  line_color="#0891b2", annotation_text="3-Stage",
                                  annotation_position="top left")
            fig_all.update_layout(
                title="All DCF Models vs Monte Carlo Distribution",
                height=380, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(title="Implied Share Price ($)", gridcolor="#f0f0f0"),
                yaxis=dict(gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig_all, use_container_width=True)

    # WACC breakdown
    with st.expander("WACC Decomposition"):
        wd = pd.DataFrame({
            "Component":    ["Cost of Equity", "After-tax Cost of Debt", "Blended WACC"],
            "Rate":         [fmt_pct(cost_eq), fmt_pct(cost_dbt * 0.79), fmt_pct(wacc)],
            "Weight":       [fmt_pct(e_wt), fmt_pct(d_wt), "100%"],
            "Contribution": [fmt_pct(cost_eq * e_wt), fmt_pct(cost_dbt * 0.79 * d_wt), fmt_pct(wacc)],
        })
        st.dataframe(wd, hide_index=True, use_container_width=True)
        st.caption(
            f"CAPM: Rf {fmt_pct(rf)} + Beta {float(d['beta']):.2f} x ERP 5.5% "
            f"= Cost of Equity {fmt_pct(cost_eq)}"
        )


# ═══════════════════════════════════════════
# TAB 2 — COMPS
# ═══════════════════════════════════════════
with tab2:
    sector     = d["sector"]
    base_comps = SECTOR_COMPS.get(sector, ["AAPL", "MSFT", "GOOGL", "AMZN", "META"])
    if manual_tickers:
        extra      = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()]
        base_comps = list(dict.fromkeys(extra + base_comps))

    with st.spinner("Fetching comparable companies..."):
        comps_df = fetch_comps(tuple(base_comps[:8]), ticker)

    if comps_df.empty:
        st.warning("Could not fetch comp data. Try entering custom tickers in the sidebar.")
    else:
        st.markdown(
            f"**Sector:** {sector} &nbsp;|&nbsp; "
            f"**Comps:** {', '.join(comps_df['Ticker'].tolist())}"
        )

        disp = comps_df[["Ticker","Name","Mkt Cap ($B)","EV/EBITDA","EV/Revenue",
                          "P/E","Fwd P/E","EBITDA Margin","Rev Growth"]].copy()
        for col in ["EV/EBITDA","EV/Revenue","P/E","Fwd P/E"]:
            disp[col] = disp[col].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
        for col in ["EBITDA Margin","Rev Growth"]:
            disp[col] = disp[col].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A"
            )

        ebitda_margin_subj = (
            fmt_pct(d["ebitda"] / d["revenue"]) if d["revenue"] and d["revenue"] != 0 else "N/A"
        )
        subj = pd.DataFrame([{
            "Ticker":       f"[{ticker}]",
            "Name":         d["name"][:22],
            "Mkt Cap ($B)": round((d["mkt_cap"] or 0) / 1e9, 1),
            "EV/EBITDA":    fmt_x(d["ev_ebitda"]),
            "EV/Revenue":   fmt_x(d["ev_revenue"]),
            "P/E":          fmt_x(d["pe"]),
            "Fwd P/E":      fmt_x(d["fwd_pe"]),
            "EBITDA Margin":ebitda_margin_subj,
            "Rev Growth":   fmt_pct(hist_growth),
        }])
        st.dataframe(pd.concat([disp, subj], ignore_index=True),
                     use_container_width=True, hide_index=True)

        # Implied prices from comps
        cp = comps_implied(d, comps_df)
        if cp:
            st.markdown("#### Comps-Implied Share Price")
            cp_cols = st.columns(len(cp))
            for col, (method, price) in zip(cp_cols, cp.items()):
                up = (price - current_price) / current_price * 100
                col.metric(method, f"${price:,.2f}", f"{up:+.1f}%",
                           delta_color="normal" if up > 0 else "inverse")

        # Scatter
        fig_sc = go.Figure()
        for _, row in comps_df.iterrows():
            rg = row["Rev Growth"] * 100 if pd.notna(row["Rev Growth"]) else None
            ev = row["EV/EBITDA"]  if pd.notna(row["EV/EBITDA"])  else None
            sz = row["Mkt Cap ($B)"] ** 0.4 * 4 if row["Mkt Cap ($B)"] else 8
            fig_sc.add_trace(go.Scatter(
                x=[rg], y=[ev], mode="markers+text",
                text=[row["Ticker"]], textposition="top center",
                marker=dict(size=sz, color="#3b82f6", opacity=0.7),
                name=row["Ticker"], showlegend=False,
            ))
        if d["ev_ebitda"]:
            fig_sc.add_trace(go.Scatter(
                x=[hist_growth * 100], y=[d["ev_ebitda"]],
                mode="markers+text", text=[f"[{ticker}]"],
                textposition="top center",
                marker=dict(size=14, color="#dc2626", symbol="diamond"),
                name=ticker, showlegend=False,
            ))
        fig_sc.update_layout(
            title="EV/EBITDA vs Revenue Growth",
            xaxis_title="Revenue Growth (%)", yaxis_title="EV/EBITDA (x)",
            height=420, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_sc, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 3 — DDM
# ═══════════════════════════════════════════
with tab3:
    div_annual = d["dividend"] or 0
    gordon_val = None
    ms_val     = None

    if div_annual == 0:
        st.warning(
            f"{ticker} does not pay a dividend — DDM not applicable. "
            "Showing the implied dividend required to justify the current price."
        )
        implied_div = current_price * (cost_eq - terminal_g) / (1 + terminal_g)
        st.metric(
            "Implied Annual Dividend (for fair value at current price)",
            f"${implied_div:.2f} / share"
        )
        st.info(
            "DDM is most meaningful for dividend-paying stocks: "
            "utilities, banks, consumer staples, and REITs."
        )
    else:
        st.markdown(
            f"**Annual Dividend:** ${div_annual:.2f} &nbsp;|&nbsp; "
            f"**Payout Ratio:** {fmt_pct(d['payout_ratio'])} &nbsp;|&nbsp; "
            f"**Cost of Equity:** {fmt_pct(cost_eq)}"
        )
        gordon_val = ddm_gordon(div_annual, terminal_g, cost_eq)
        ms_val     = ddm_multistage(div_annual, g1, terminal_g, cost_eq, 5)

        ca, cb, cc = st.columns(3)
        if gordon_val:
            up = (gordon_val - current_price) / current_price * 100
            ca.metric("Gordon Growth Model", f"${gordon_val:,.2f}", f"{up:+.1f}%",
                      delta_color="normal" if up > 0 else "inverse")
        if ms_val:
            up2 = (ms_val - current_price) / current_price * 100
            cb.metric("Multi-Stage DDM", f"${ms_val:,.2f}", f"{up2:+.1f}%",
                      delta_color="normal" if up2 > 0 else "inverse")
        cc.metric("Current Price", f"${current_price:,.2f}")

        ke_range  = np.arange(cost_eq - 0.03, cost_eq + 0.035, 0.005)
        ddm_curve = [ddm_gordon(div_annual, terminal_g, ke) or 0 for ke in ke_range]
        fig_ddm = go.Figure()
        fig_ddm.add_trace(go.Scatter(
            x=ke_range * 100, y=ddm_curve, mode="lines+markers",
            line=dict(color="#3b82f6", width=2), name="DDM Value"
        ))
        fig_ddm.add_hline(y=current_price, line_dash="dash", line_color="#dc2626",
                          annotation_text=f"Current: ${current_price:.2f}")
        fig_ddm.update_layout(
            title="DDM Implied Value vs Required Return",
            xaxis_title="Required Return / Cost of Equity (%)",
            yaxis_title="Implied Price ($)",
            height=360, plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        )
        st.plotly_chart(fig_ddm, use_container_width=True)


# ═══════════════════════════════════════════
# TAB 4 — FOOTBALL FIELD
# ═══════════════════════════════════════════
with tab4:
    bars = []

    # DCF bars
    if "2-Stage DCF" in all_dcf:
        ps = all_dcf["2-Stage DCF"]
        bars.append(("2-Stage DCF", ps * 0.80, ps * 1.20, ps))

    if "3-Stage DCF" in all_dcf:
        ps = all_dcf["3-Stage DCF"]
        bars.append(("3-Stage DCF", ps * 0.80, ps * 1.20, ps))

    if mc_results is not None and len(mc_results) > 0:
        bars.append(("Monte Carlo DCF", mc_p10, mc_p90, mc_p50))

    # Comps bars — uses comps_df if it was fetched
    if "comps_df" in dir() and not comps_df.empty:
        ev2eq = lambda ev: (ev - debt + cash) / shares if shares else 0

        ev_eb = comps_df["EV/EBITDA"].dropna()
        if len(ev_eb) >= 2 and d["ebitda"] and d["ebitda"] > 0:
            bars.append((
                "EV/EBITDA Comps",
                ev2eq(d["ebitda"] * ev_eb.quantile(0.25)),
                ev2eq(d["ebitda"] * ev_eb.quantile(0.75)),
                ev2eq(d["ebitda"] * ev_eb.median()),
            ))

        ev_rv = comps_df["EV/Revenue"].dropna()
        if len(ev_rv) >= 2 and d["revenue"] and d["revenue"] > 0:
            bars.append((
                "EV/Revenue Comps",
                ev2eq(d["revenue"] * ev_rv.quantile(0.25)),
                ev2eq(d["revenue"] * ev_rv.quantile(0.75)),
                ev2eq(d["revenue"] * ev_rv.median()),
            ))

        pe_v = comps_df["P/E"].dropna()
        if len(pe_v) >= 2 and d["net_income"] and d["net_income"] > 0:
            bars.append((
                "P/E Comps",
                (d["net_income"] * pe_v.quantile(0.25)) / shares,
                (d["net_income"] * pe_v.quantile(0.75)) / shares,
                (d["net_income"] * pe_v.median())        / shares,
            ))

    # DDM bar
    if gordon_val is not None and gordon_val and gordon_val > 0:
        bars.append(("DDM — Gordon Growth", gordon_val * 0.85, gordon_val * 1.15, gordon_val))

    # Filter nonsensical values
    bars = [
        (name, lo, hi, mid) for name, lo, hi, mid in bars
        if lo > 0 and hi > 0
        and hi < current_price * 15
        and lo < current_price * 15
    ]

    if not bars:
        st.warning("Not enough valuation data to render the football field chart.")
    else:
        colors = ["#3b82f6","#6366f1","#8b5cf6","#0891b2",
                  "#059669","#d97706","#dc2626","#db2777"]

        fig_ff = go.Figure()
        for i, (name, lo, hi, mid) in enumerate(bars):
            c = colors[i % len(colors)]
            fig_ff.add_trace(go.Bar(
                name=name, x=[hi - lo], y=[name], base=[lo],
                orientation="h",
                marker=dict(color=c, opacity=0.7),
                text=f"${lo:,.1f} – ${hi:,.1f}",
                textposition="inside", insidetextanchor="middle",
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Low: ${lo:,.2f}<br>Mid: ${mid:,.2f}<br>High: ${hi:,.2f}"
                    "<extra></extra>"
                ),
            ))
            fig_ff.add_trace(go.Scatter(
                x=[mid], y=[name], mode="markers",
                marker=dict(color="white", size=10, line=dict(color=c, width=2)),
                showlegend=False, hoverinfo="skip",
            ))

        fig_ff.add_vline(
            x=current_price,
            line_dash="solid", line_color="#dc2626", line_width=2,
            annotation_text=f"Current: ${current_price:.2f}",
            annotation_position="top right",
            annotation=dict(font=dict(color="#dc2626", size=12)),
        )

        all_lo = [b[1] for b in bars]
        all_hi = [b[2] for b in bars]
        fig_ff.update_layout(
            title=f"{d['name']} ({ticker}) — Valuation Football Field",
            xaxis=dict(
                title="Implied Share Price ($)",
                range=[max(0, min(all_lo) * 0.85), max(all_hi) * 1.15],
                tickprefix="$", gridcolor="#f0f0f0", zeroline=False,
            ),
            yaxis=dict(autorange="reversed"),
            height=max(350, 80 + len(bars) * 60),
            plot_bgcolor="white", paper_bgcolor="white",
            barmode="overlay", showlegend=False,
            margin=dict(l=170, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_ff, use_container_width=True)

        st.markdown("#### Implied Price Summary")
        summary_rows = []
        for name, lo, hi, mid in bars:
            up = (mid - current_price) / current_price * 100
            summary_rows.append({
                "Method":         name,
                "Bear Case":      f"${lo:,.2f}",
                "Base Case":      f"${mid:,.2f}",
                "Bull Case":      f"${hi:,.2f}",
                "Upside to Base": f"{up:+.1f}%",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
# TAB 5 — SENSITIVITY
# ═══════════════════════════════════════════
with tab5:
    st.markdown("#### DCF Sensitivity: Implied Price by WACC and Growth Rate")
    st.caption("Green = above current price. Red = below current price.")

    sens = sensitivity_table(fcf_base, wacc, g1, terminal_g, debt, cash, shares, projection_years)

    z    = sens.values.astype(float)
    x_lb = [f"{v*100:.1f}%" for v in sens.columns]
    y_lb = [f"{v*100:.1f}%" for v in sens.index]

    fig_h = go.Figure(data=go.Heatmap(
        z=z, x=x_lb, y=y_lb,
        colorscale=[
            [0.0, "#dc2626"], [0.3, "#fca5a5"],
            [0.5, "#fef3c7"],
            [0.7, "#86efac"], [1.0, "#16a34a"],
        ],
        zmid=current_price,
        text=[[f"${v:.0f}" if np.isfinite(v) else "N/A" for v in row] for row in z],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="Growth: %{y}<br>WACC: %{x}<br>Price: %{text}<extra></extra>",
    ))
    fig_h.update_layout(
        xaxis_title="WACC",
        yaxis_title="FCF Growth Rate (Yr 1-5)",
        height=430, margin=dict(l=80, r=20, t=30, b=60),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("#### Terminal Growth Rate Sensitivity")
    tg_range  = np.arange(0.01, 0.04, 0.005)
    wacc_pts  = [wacc - 0.01, wacc, wacc + 0.01]
    fig_tg    = go.Figure()
    for w in wacc_pts:
        pts = []
        for tg in tg_range:
            if w <= tg:
                pts.append(np.nan)
                continue
            p, _, _ = dcf_2stage(fcf_base, g1, w, tg, projection_years, debt, cash, shares)
            pts.append(p)
        fig_tg.add_trace(go.Scatter(
            x=tg_range * 100, y=pts, mode="lines+markers",
            name=f"WACC = {w*100:.1f}%", line=dict(width=2),
        ))
    fig_tg.add_hline(y=current_price, line_dash="dash", line_color="#dc2626",
                     annotation_text=f"Current: ${current_price:.2f}")
    fig_tg.update_layout(
        xaxis_title="Terminal Growth Rate (%)", yaxis_title="Implied Price ($)",
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_tg, use_container_width=True)

    if show_raw:
        with st.expander("Raw Sensitivity Table"):
            raw_df = sens.copy()
            raw_df.columns = [f"WACC {v*100:.1f}%" for v in raw_df.columns]
            raw_df.index   = [f"Growth {v*100:.1f}%" for v in raw_df.index]
            st.dataframe(raw_df.style.format("${:.2f}"), use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    f"Data: Yahoo Finance · FRED (10Y UST: {rf*100:.2f}%) · "
    "For educational and research purposes only. Not investment advice. "
    f"Refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}"
)
