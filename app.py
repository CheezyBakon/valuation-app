import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import io

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Dynamic Valuation Model",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header {font-size:2rem;font-weight:700;color:#1a1a2e;margin-bottom:0;}
  .sub-header  {font-size:0.9rem;color:#6b7280;margin-bottom:1.5rem;}
  .section-title {font-size:1.05rem;font-weight:600;color:#1a1a2e;
                  border-left:3px solid #3b82f6;padding-left:0.75rem;
                  margin:1.5rem 0 0.75rem;}
  .tag {display:inline-block;background:#eff6ff;color:#1d4ed8;border-radius:4px;
        padding:2px 8px;font-size:0.75rem;font-weight:500;margin:2px;}
  .risk-flag {background:#fef2f2;border:1px solid #fca5a5;border-radius:8px;
              padding:10px 14px;color:#991b1b;font-size:0.85rem;margin:8px 0;}
  .scenario-base {border-left:3px solid #3b82f6;}
  .scenario-bull {border-left:3px solid #16a34a;}
  .scenario-bear {border-left:3px solid #dc2626;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600)
def get_risk_free_rate():
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
        df  = pd.read_csv(url, parse_dates=["DATE"])
        df  = df[df["DGS10"] != "."]
        return float(df["DGS10"].iloc[-1]) / 100
    except Exception:
        return 0.043


@st.cache_data(ttl=900)
def fetch_company_data(ticker: str):
    t    = yf.Ticker(ticker)
    info = t.info
    hist = t.history(period="5y")
    fin  = t.financials
    cf   = t.cashflow
    bs   = t.balance_sheet
    return info, hist, fin, cf, bs


def extract_financials(info, fin, cf, bs, ticker_sym):
    d = {}
    d["revenue"]          = info.get("totalRevenue")          or 0
    d["ebitda"]           = info.get("ebitda")                or 0
    d["net_income"]       = info.get("netIncomeToCommon")     or 0
    d["gross_profit"]     = info.get("grossProfits")          or 0
    d["total_debt"]       = info.get("totalDebt")             or 0
    d["cash"]             = info.get("totalCash")             or 0
    d["shares"]           = info.get("sharesOutstanding")     or 1
    d["price"]            = info.get("currentPrice") or info.get("regularMarketPrice") or 0
    d["beta"]             = info.get("beta")                  or 1.0
    d["mkt_cap"]          = info.get("marketCap")             or 0
    d["dividend"]         = info.get("dividendRate")          or 0
    d["payout_ratio"]     = info.get("payoutRatio")           or 0
    d["pe"]               = info.get("trailingPE")
    d["fwd_pe"]           = info.get("forwardPE")
    d["ev_ebitda"]        = info.get("enterpriseToEbitda")
    d["ev_revenue"]       = info.get("enterpriseToRevenue")
    d["roe"]              = info.get("returnOnEquity")        or 0
    d["roic"]             = info.get("returnOnAssets")        or 0
    d["ebitda_margin"]    = info.get("ebitdaMargins")         or 0
    d["gross_margin"]     = info.get("grossMargins")          or 0
    d["fcf_margin"]       = info.get("freeCashflowYield")     or 0
    d["rev_growth"]       = info.get("revenueGrowth")         or 0
    d["earnings_growth"]  = info.get("earningsGrowth")        or 0
    d["sector"]           = info.get("sector")                or "Unknown"
    d["industry"]         = info.get("industry")             or "Unknown"
    d["name"]             = info.get("longName") or info.get("shortName") or ticker_sym
    d["currency"]         = info.get("currency")              or "USD"
    d["country"]          = info.get("country")               or ""
    d["description"]      = info.get("longBusinessSummary")   or ""
    d["enterprise_value"] = info.get("enterpriseValue")       or d["mkt_cap"]
    d["interest_expense"] = info.get("interestExpense")       or 0
    d["minority_interest"]= info.get("minorityInterest")      or 0
    d["preferred_stock"]  = info.get("preferredStock")        or 0
    d["fwd_eps"]          = info.get("forwardEps")            or 0
    d["trailing_eps"]     = info.get("trailingEps")           or 0
    d["analyst_target"]   = info.get("targetMeanPrice")       or 0

    # Historical revenue series (oldest → newest)
    rev_hist = []
    if fin is not None and not fin.empty:
        for idx in fin.index:
            if "revenue" in idx.lower():
                series   = fin.loc[idx].dropna()
                rev_hist = list(series.values[::-1])
                break
    d["rev_hist"] = rev_hist

    # Historical EBITDA margin series
    ebitda_hist = []
    if fin is not None and not fin.empty:
        for idx in fin.index:
            if "ebitda" in idx.lower():
                series      = fin.loc[idx].dropna()
                ebitda_hist = list(series.values[::-1])
                break
    d["ebitda_hist"] = ebitda_hist

    # FCF — direct line first, then OCF + capex
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
            fcf = ocf + capex
    d["fcf"] = fcf

    # Null-guard
    for key in ["revenue","ebitda","net_income","gross_profit","total_debt","cash",
                "mkt_cap","enterprise_value","fcf","dividend","payout_ratio",
                "roe","roic","beta","interest_expense","fwd_eps","trailing_eps"]:
        if d[key] is None:
            d[key] = 0

    return d


# ═══════════════════════════════════════════════════════════════
# FINANCIAL LOGIC
# ═══════════════════════════════════════════════════════════════

def compute_wacc(d, rf, tax_rate=0.21):
    beta      = max(float(d["beta"] or 1.0), 0.3)
    erp       = 0.055
    cost_eq   = rf + beta * erp
    total_dbt = d["total_debt"] or 0
    mkt_cap   = d["mkt_cap"]   or 1
    # Implied cost of debt from interest expense if available
    if d["interest_expense"] and total_dbt > 0:
        cost_dbt = abs(d["interest_expense"]) / total_dbt
        cost_dbt = max(min(cost_dbt, 0.15), 0.03)
    else:
        cost_dbt = max(rf + 0.015, 0.04)
    total_cap = mkt_cap + total_dbt
    ew        = mkt_cap   / total_cap if total_cap > 0 else 1.0
    dw        = total_dbt / total_cap if total_cap > 0 else 0.0
    wacc      = ew * cost_eq + dw * cost_dbt * (1 - tax_rate)
    return round(wacc,4), round(cost_eq,4), round(cost_dbt,4), round(ew,4), round(dw,4)


def estimate_growth(rev_hist):
    if len(rev_hist) >= 2:
        rates = []
        for i in range(1, len(rev_hist)):
            if rev_hist[i-1] and rev_hist[i-1] != 0:
                rates.append((rev_hist[i] - rev_hist[i-1]) / abs(rev_hist[i-1]))
        if rates:
            return float(np.median(rates))
    return 0.08


def credit_risk_flag(d):
    """Return (level, message) or None. level = 'warning' | 'error'."""
    ebitda = d["ebitda"]
    debt   = d["total_debt"]
    if not ebitda or ebitda <= 0 or not debt or debt <= 0:
        return None
    leverage = debt / ebitda
    if leverage > 6:
        return ("error",
                f"High leverage: Net Debt / EBITDA = {leverage:.1f}x. "
                "DCF equity value is extremely sensitive to WACC at this leverage level. "
                "Consider adding a distress discount.")
    if leverage > 4:
        return ("warning",
                f"Elevated leverage: Net Debt / EBITDA = {leverage:.1f}x. "
                "Monitor refinancing risk and coverage ratios.")
    return None


# ── FCF build from margin assumptions ─────────────────────────────────────────
def build_fcf_from_margins(revenue, ebitda_margin, fcf_conversion, g_rates):
    """
    Project revenue with fading growth rates, apply EBITDA margin,
    then FCF conversion rate to get free cash flows year by year.
    """
    fcfs = []
    rev  = revenue
    for g in g_rates:
        rev  = rev * (1 + g)
        ebitda = rev * ebitda_margin
        fcf    = ebitda * fcf_conversion
        fcfs.append(fcf)
    return fcfs


# ── DCF engines ───────────────────────────────────────────────────────────────
def dcf_fading_growth(revenue, ebitda_margin, fcf_conv, g_start, g_end,
                      wacc, tg, years, debt, cash, shares):
    """Full margin-based DCF with automatic growth fade."""
    if shares == 0 or revenue == 0:
        return 0, 0, [], []
    g_rates = list(np.linspace(g_start, g_end, years))
    fcfs    = build_fcf_from_margins(revenue, ebitda_margin, fcf_conv, g_rates)
    pvs     = [f / (1+wacc)**(i+1) for i,f in enumerate(fcfs)]
    tv      = fcfs[-1]*(1+tg)/(wacc-tg) if wacc > tg else 0
    pv_tv   = tv / (1+wacc)**years
    ev      = sum(pvs) + pv_tv
    per_share = (ev - debt + cash) / shares
    return per_share, ev, pvs, fcfs


def dcf_2stage(fcf, g1, wacc, tg, years=5, debt=0, cash=0, shares=1):
    if shares == 0 or fcf == 0:
        return 0, 0, []
    fcfs, pvs = [], []
    f = fcf
    for i in range(1, years+1):
        f = f*(1+g1)
        pvs.append(f/(1+wacc)**i)
        fcfs.append(f)
    tv    = fcfs[-1]*(1+tg)/(wacc-tg) if wacc > tg else 0
    pv_tv = tv/(1+wacc)**years
    ev    = sum(pvs)+pv_tv
    return (ev-debt+cash)/shares, ev, pvs


def dcf_3stage(fcf, g1, g2, wacc, tg, y1=3, y2=4, debt=0, cash=0, shares=1):
    if shares == 0 or fcf == 0:
        return 0, 0, []
    fcfs, pvs = [], []
    f = fcf
    for i in range(1, y1+1):
        f = f*(1+g1); pvs.append(f/(1+wacc)**i); fcfs.append(f)
    for i,gr in enumerate(np.linspace(g1,g2,y2), start=y1+1):
        f = f*(1+gr); pvs.append(f/(1+wacc)**i); fcfs.append(f)
    tv    = fcfs[-1]*(1+tg)/(wacc-tg) if wacc > tg else 0
    pv_tv = tv/(1+wacc)**(y1+y2)
    ev    = sum(pvs)+pv_tv
    return (ev-debt+cash)/shares, ev, pvs


def dcf_monte_carlo(fcf, g_mean, g_std, wacc_mean, wacc_std, tg,
                    n=10000, years=5, debt=0, cash=0, shares=1):
    np.random.seed(42)
    out = []
    for _ in range(n):
        g   = np.random.normal(g_mean, g_std)
        w   = max(np.random.normal(wacc_mean, wacc_std), tg+0.01)
        tgr = min(np.random.normal(tg, 0.005), w-0.01)
        f, pv_sum = fcf, 0
        for i in range(1, years+1):
            f = f*(1+g); pv_sum += f/(1+w)**i
        tv    = f*(1+tgr)/(w-tgr) if w > tgr else 0
        pv_tv = tv/(1+w)**years
        ev    = pv_sum+pv_tv
        out.append((ev-debt+cash)/shares if shares else 0)
    return np.array(out)


# ── Reverse DCF ───────────────────────────────────────────────────────────────
def reverse_dcf(current_price, shares, debt, cash, fcf_base,
                wacc, tg, years=10):
    """
    Solve for the growth rate g that makes DCF implied price = current price.
    Uses bisection search.
    """
    target_eq = current_price * shares
    target_ev = target_eq + debt - cash

    def ev_at_g(g):
        f, pv = fcf_base, 0
        for i in range(1, years+1):
            f = f*(1+g)
            pv += f/(1+wacc)**i
        tv    = f*(1+tg)/(wacc-tg) if wacc > tg else 0
        pv_tv = tv/(1+wacc)**years
        return pv + pv_tv

    lo, hi = -0.30, 0.80
    if ev_at_g(lo) > target_ev:
        return lo
    if ev_at_g(hi) < target_ev:
        return hi
    for _ in range(60):
        mid = (lo+hi)/2
        if ev_at_g(mid) < target_ev:
            lo = mid
        else:
            hi = mid
    return round((lo+hi)/2, 4)


# ── DDM ───────────────────────────────────────────────────────────────────────
def ddm_gordon(div, g, ke):
    if ke <= g or div == 0: return None
    return div*(1+g)/(ke-g)


def ddm_multistage(div, g1, g2, ke, years=5):
    if ke <= g2 or div == 0: return None
    d, pvs = div, []
    for i in range(1, years+1):
        d = d*(1+g1); pvs.append(d/(1+ke)**i)
    tv    = d*(1+g2)/(ke-g2)
    pv_tv = tv/(1+ke)**years
    return sum(pvs)+pv_tv


# ── Comps ─────────────────────────────────────────────────────────────────────
SECTOR_COMPS = {
    "Technology":             ["AAPL","MSFT","GOOGL","META","AMZN","ORCL","CRM"],
    "Consumer Cyclical":      ["AMZN","TSLA","HD","NKE","MCD","BKNG","TGT"],
    "Healthcare":             ["JNJ","UNH","PFE","ABBV","MRK","TMO","ABT"],
    "Financial Services":     ["JPM","BAC","WFC","GS","MS","BLK","SCHW"],
    "Communication Services": ["GOOGL","META","NFLX","DIS","T","VZ","CMCSA"],
    "Industrials":            ["CAT","BA","HON","UPS","GE","RTX","LMT"],
    "Consumer Defensive":     ["PG","KO","PEP","WMT","COST","CL","GIS"],
    "Energy":                 ["XOM","CVX","COP","SLB","EOG","PSX","VLO"],
    "Utilities":              ["NEE","DUK","SO","D","EXC","SRE","AEP"],
    "Real Estate":            ["AMT","PLD","CCI","EQIX","SPG","O","PSA"],
    "Basic Materials":        ["LIN","APD","ECL","NEM","FCX","DOW","DD"],
}


@st.cache_data(ttl=3600)
def fetch_comps(tickers, exclude):
    rows = []
    for t in tickers:
        if t == exclude.upper(): continue
        try:
            info = yf.Ticker(t).info
            rows.append({
                "Ticker":        t,
                "Name":          info.get("shortName", t),
                "EV/EBITDA":     info.get("enterpriseToEbitda"),
                "EV/Revenue":    info.get("enterpriseToRevenue"),
                "P/E (TTM)":     info.get("trailingPE"),
                "Fwd P/E":       info.get("forwardPE"),
                "Mkt Cap ($B)":  round((info.get("marketCap") or 0)/1e9,1),
                "Rev Growth":    info.get("revenueGrowth"),
                "EBITDA Margin": info.get("ebitdaMargins"),
                "Gross Margin":  info.get("grossMargins"),
                "ROE":           info.get("returnOnEquity"),
            })
        except Exception:
            pass
    return pd.DataFrame(rows)


def comps_implied(d, comps_df):
    results = {}
    debt, cash, shares = d["total_debt"], d["cash"], d["shares"]
    def ev2eq(ev): return (ev-debt+cash)/shares if shares else 0
    if d["ebitda"] and d["ebitda"] > 0:
        med = comps_df["EV/EBITDA"].dropna().median()
        if pd.notna(med): results["EV/EBITDA Comps"] = ev2eq(d["ebitda"]*med)
    if d["revenue"] and d["revenue"] > 0:
        med = comps_df["EV/Revenue"].dropna().median()
        if pd.notna(med): results["EV/Revenue Comps"] = ev2eq(d["revenue"]*med)
    if d["net_income"] and d["net_income"] > 0:
        med = comps_df["P/E (TTM)"].dropna().median()
        if pd.notna(med): results["P/E Comps"] = (d["net_income"]*med)/shares if shares else 0
    return results


def comps_regression_implied(d, comps_df):
    """Regression-implied price: fit EV/EBITDA ~ Rev Growth, read off company."""
    sub = comps_df[["EV/EBITDA","Rev Growth"]].dropna()
    if len(sub) < 3 or not d["ebitda"] or d["ebitda"] <= 0:
        return None, None, None
    x = sub["Rev Growth"].values
    y = sub["EV/EBITDA"].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    company_growth = d["rev_growth"] or estimate_growth(d["rev_hist"])
    implied_multiple = intercept + slope * company_growth
    implied_multiple = max(implied_multiple, 1.0)
    debt, cash, shares = d["total_debt"], d["cash"], d["shares"]
    ev2eq = lambda ev: (ev-debt+cash)/shares if shares else 0
    implied_price = ev2eq(d["ebitda"] * implied_multiple)
    return implied_price, implied_multiple, r**2


# ── Sensitivity ───────────────────────────────────────────────────────────────
def sensitivity_table(fcf, base_wacc, base_g, tg, debt, cash, shares, years):
    wacc_range = np.arange(base_wacc-0.02, base_wacc+0.025, 0.005)
    g_range    = np.arange(base_g   -0.04, base_g   +0.045, 0.010)
    rows = {}
    for g in g_range:
        row = {}
        for w in wacc_range:
            if w <= tg:
                row[round(w,3)] = np.nan
            else:
                p,_,_ = dcf_2stage(fcf, g, w, tg, years, debt, cash, shares)
                row[round(w,3)] = round(p,2)
        rows[round(g,3)] = row
    df = pd.DataFrame(rows).T
    df.index.name   = "Growth Rate"
    df.columns.name = "WACC"
    return df


# ── Excel export ──────────────────────────────────────────────────────────────
def build_excel(d, wacc, cost_eq, rf, terminal_g, g1, fcf_base,
                projection_years, all_dcf, comps_df, bars):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1: Company overview
        overview = pd.DataFrame({
            "Metric": ["Company","Ticker","Sector","Industry","Stock Price",
                       "Market Cap","Enterprise Value","Revenue","EBITDA","FCF",
                       "Net Income","Total Debt","Cash","Beta","WACC","Cost of Equity",
                       "P/E TTM","Fwd P/E","EV/EBITDA","EV/Revenue","ROE"],
            "Value":  [d["name"], "", d["sector"], d["industry"],
                       d["price"], d["mkt_cap"], d["enterprise_value"],
                       d["revenue"], d["ebitda"], d["fcf"], d["net_income"],
                       d["total_debt"], d["cash"], d["beta"], wacc, cost_eq,
                       d["pe"], d["fwd_pe"], d["ev_ebitda"], d["ev_revenue"], d["roe"]],
        })
        overview.to_excel(writer, sheet_name="Overview", index=False)

        # Sheet 2: DCF projections
        if fcf_base and g1:
            yrs   = list(range(1, projection_years+1))
            g_rates = list(np.linspace(g1, terminal_g+0.01, projection_years))
            f     = fcf_base
            rows_dcf = []
            for i, g in enumerate(g_rates):
                f = f*(1+g)
                pv = f/(1+wacc)**(i+1)
                rows_dcf.append({"Year": f"Yr {i+1}", "Growth Rate": g,
                                  "FCF": f, "PV of FCF": pv})
            pd.DataFrame(rows_dcf).to_excel(writer, sheet_name="DCF Projections", index=False)

        # Sheet 3: Valuation summary
        if bars:
            summ = [{"Method":n,"Bear":lo,"Base":mid,"Bull":hi,
                     "Upside to Base %":(mid-d["price"])/d["price"]*100}
                    for n,lo,hi,mid in bars]
            pd.DataFrame(summ).to_excel(writer, sheet_name="Valuation Summary", index=False)

        # Sheet 4: Comps
        if comps_df is not None and not comps_df.empty:
            comps_df.to_excel(writer, sheet_name="Comparable Companies", index=False)

        # Sheet 5: Sensitivity
        sens = sensitivity_table(fcf_base, wacc, g1, terminal_g,
                                 d["total_debt"], d["cash"], d["shares"], projection_years)
        sens.to_excel(writer, sheet_name="DCF Sensitivity")

    buf.seek(0)
    return buf.getvalue()


# ── Formatting ────────────────────────────────────────────────────────────────
def fmt_B(v):
    if v is None: return "N/A"
    sign = "-$" if v < 0 else "$"
    av = abs(v)
    if av >= 1e12: return f"{sign}{av/1e12:.2f}T"
    if av >= 1e9:  return f"{sign}{av/1e9:.1f}B"
    if av >= 1e6:  return f"{sign}{av/1e6:.0f}M"
    return f"{sign}{av:,.0f}"

def fmt_pct(v):
    return "N/A" if v is None else f"{v*100:.1f}%"

def fmt_x(v):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))): return "N/A"
    return f"{v:.1f}x"

def kpi_card(label, value, sub=None):
    sub_html = f"<div style='font-size:11px;color:#6b7280;margin-top:3px;'>{sub}</div>" if sub else ""
    return (f"<div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;"
            f"padding:14px 16px;min-width:0;'>"
            f"<div style='font-size:11px;color:#6b7280;text-transform:uppercase;"
            f"letter-spacing:0.05em;margin-bottom:6px;'>{label}</div>"
            f"<div style='font-size:18px;font-weight:700;color:#1a1a2e;"
            f"word-break:break-word;line-height:1.3;'>{value}</div>{sub_html}</div>")

def kpi_row(items):
    cards = "".join(kpi_card(l, v, s if len(i)==3 else None)
                    for i in items for l,v,*s in [i])
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat({len(items)},1fr);'
        f'gap:10px;margin-bottom:10px;">{cards}</div>',
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="main-header">Dynamic Valuation Model</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">DCF · Comps · DDM · Reverse DCF · Scenarios · Football Field · Monte Carlo</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Company")
    ticker  = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. MSFT, NVDA, JPM").upper().strip()
    run_btn = st.button("Run Valuation", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### DCF Model")
    dcf_scenario = st.selectbox("DCF Type", ["2-Stage","3-Stage","Monte Carlo","All Three"])

    st.markdown("**Margin-Based Projections**")
    use_margin_dcf = st.checkbox("Use margin-based FCF build", value=False,
                                  help="Project revenue x EBITDA margin x FCF conversion instead of flat FCF growth")

    st.divider()
    st.markdown("### Base Scenario")
    col1, col2 = st.columns(2)
    with col1:
        g1_pct  = st.number_input("Growth Yr 1 (%)", value=10.0, step=0.5, format="%.1f")
    with col2:
        tg_pct  = st.number_input("Terminal Growth (%)", value=2.5, step=0.1, format="%.1f")

    if "3-Stage" in dcf_scenario or dcf_scenario == "All Three":
        g2_pct = st.number_input("Fade Growth Yr 6-10 (%)", value=5.0, step=0.5, format="%.1f")
    else:
        g2_pct = 5.0

    if use_margin_dcf:
        ebitda_margin_inp = st.slider("EBITDA Margin (%)", 5, 60, 20) / 100
        fcf_conv_inp      = st.slider("FCF / EBITDA Conversion (%)", 30, 90, 60) / 100
    else:
        ebitda_margin_inp = 0.20
        fcf_conv_inp      = 0.60

    wacc_override    = st.number_input("WACC Override (0 = auto)", value=0.0, step=0.1, format="%.1f")
    projection_years = st.slider("Projection Years", 5, 10, 5)
    tax_rate_pct     = st.slider("Tax Rate (%)", 10, 35, 21)

    st.divider()
    st.markdown("### Bull / Bear Scenarios")
    with st.expander("Bull Case"):
        bull_g1 = st.number_input("Bull Growth Yr 1 (%)", value=g1_pct+5, step=0.5, format="%.1f", key="bull_g1")
        bull_tg = st.number_input("Bull Terminal Growth (%)", value=tg_pct+0.5, step=0.1, format="%.1f", key="bull_tg")
        bull_wacc_adj = st.number_input("Bull WACC Adj (pp)", value=-0.5, step=0.1, format="%.1f", key="bull_wacc")
    with st.expander("Bear Case"):
        bear_g1 = st.number_input("Bear Growth Yr 1 (%)", value=max(g1_pct-5,0), step=0.5, format="%.1f", key="bear_g1")
        bear_tg = st.number_input("Bear Terminal Growth (%)", value=max(tg_pct-0.5,0.5), step=0.1, format="%.1f", key="bear_tg")
        bear_wacc_adj = st.number_input("Bear WACC Adj (pp)", value=1.0, step=0.1, format="%.1f", key="bear_wacc")

    # Monte Carlo — always defined
    mc_g_std = 0.05; mc_wacc_std = 0.015; n_sims = 10000
    if "Monte Carlo" in dcf_scenario or dcf_scenario == "All Three":
        st.divider()
        st.markdown("### Monte Carlo Settings")
        mc_g_std    = st.slider("Growth Std Dev (%)", 1.0, 15.0, 5.0) / 100
        mc_wacc_std = st.slider("WACC Std Dev (%)", 0.5, 5.0, 1.5) / 100
        n_sims      = st.select_slider("Simulations", options=[1000,5000,10000,25000], value=10000)

    st.divider()
    st.markdown("### Comps")
    manual_tickers = st.text_input("Custom Comp Tickers (comma-sep)", placeholder="e.g. MSFT,GOOGL,META")

    st.divider()
    show_raw    = st.checkbox("Show raw sensitivity table", value=False)
    show_bridge = st.checkbox("Show EV bridge", value=True)


# ═══════════════════════════════════════════════════════════════
# GATE & FETCH
# ═══════════════════════════════════════════════════════════════
if not run_btn and "last_ticker" not in st.session_state:
    st.info("Enter a ticker in the sidebar and click Run Valuation to begin.")
    st.stop()

if run_btn:
    st.session_state["last_ticker"] = ticker
else:
    ticker = st.session_state.get("last_ticker","AAPL")

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


# ═══════════════════════════════════════════════════════════════
# DERIVED VALUES
# ═══════════════════════════════════════════════════════════════
tax_rate   = tax_rate_pct / 100
wacc, cost_eq, cost_dbt, e_wt, d_wt = compute_wacc(d, rf, tax_rate)
if wacc_override > 0:
    wacc = wacc_override / 100

g1         = g1_pct / 100
g2         = g2_pct / 100
terminal_g = min(tg_pct / 100, wacc - 0.005)

# Scenario WACCs
bull_wacc  = max(wacc + bull_wacc_adj/100, terminal_g + 0.01)
bear_wacc  = wacc + bear_wacc_adj/100
bull_g     = bull_g1 / 100
bear_g     = bear_g1 / 100
bull_tg_v  = min(bull_tg / 100, bull_wacc - 0.005)
bear_tg_v  = min(bear_tg / 100, bear_wacc - 0.005)

hist_growth   = estimate_growth(d["rev_hist"])
current_price = d["price"]
debt          = d["total_debt"] or 0
cash          = d["cash"]       or 0
shares        = d["shares"]     or 1

# FCF base
if d["fcf"] and d["fcf"] != 0:
    fcf_base = d["fcf"]; fcf_note = None
elif d["ebitda"] and d["ebitda"] != 0:
    fcf_base = d["ebitda"]*0.5
    fcf_note = "FCF not available — using 50% of EBITDA as proxy."
elif d["net_income"] and d["net_income"] != 0:
    fcf_base = d["net_income"]
    fcf_note = "FCF and EBITDA not available — using Net Income as proxy."
else:
    fcf_base = 1e8
    fcf_note = "Could not derive FCF from available data. Using $100M placeholder."

# Margin-based FCF override
if use_margin_dcf and d["revenue"] and d["revenue"] > 0:
    fcf_base_margin = d["revenue"] * ebitda_margin_inp * fcf_conv_inp
else:
    fcf_base_margin = fcf_base


# ═══════════════════════════════════════════════════════════════
# COMPANY HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown(
    f"## {d['name']} <span style='font-size:1rem;color:#6b7280;'>({ticker})</span>",
    unsafe_allow_html=True
)
st.markdown(
    f"<span class='tag'>{d['sector']}</span> "
    f"<span class='tag'>{d['industry']}</span> "
    f"<span class='tag'>{d['country']}</span>",
    unsafe_allow_html=True
)
if d["analyst_target"]:
    upside_analyst = (d["analyst_target"] - current_price) / current_price * 100
    st.caption(f"Analyst consensus target: ${d['analyst_target']:.2f}  ({upside_analyst:+.1f}% from current)")

if d["description"]:
    with st.expander("Business description"):
        st.write(d["description"][:900] + "...")

if fcf_note:
    st.warning(fcf_note)

# Credit risk flag
flag = credit_risk_flag(d)
if flag:
    level, msg = flag
    if level == "error":
        st.error(f"Leverage Risk: {msg}")
    else:
        st.warning(f"Leverage Note: {msg}")


# ═══════════════════════════════════════════════════════════════
# KEY FINANCIALS
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Key Financials</p>', unsafe_allow_html=True)

kpi_row([
    ("Stock Price",      f"${current_price:,.2f}"),
    ("Market Cap",       fmt_B(d["mkt_cap"])),
    ("Enterprise Value", fmt_B(d["enterprise_value"])),
    ("Revenue",          fmt_B(d["revenue"])),
    ("EBITDA",           fmt_B(d["ebitda"])),
    ("FCF",              fmt_B(d["fcf"]) if d["fcf"] else "N/A"),
    ("Net Income",       fmt_B(d["net_income"])),
])
kpi_row([
    ("Beta",           f"{float(d['beta']):.2f}"),
    ("P/E (TTM)",      fmt_x(d["pe"])),
    ("Fwd P/E",        fmt_x(d["fwd_pe"])),
    ("EV/EBITDA",      fmt_x(d["ev_ebitda"])),
    ("EV/Revenue",     fmt_x(d["ev_revenue"])),
    ("ROE",            fmt_pct(d["roe"])),
    ("WACC",           fmt_pct(wacc)),
])
kpi_row([
    ("Gross Margin",   fmt_pct(d["gross_margin"])),
    ("EBITDA Margin",  fmt_pct(d["ebitda_margin"])),
    ("Rev Growth",     fmt_pct(d["rev_growth"])),
    ("Hist Rev CAGR",  fmt_pct(hist_growth)),
    ("Net Debt",       fmt_B(debt - cash)),
    ("ND/EBITDA",      fmt_x((debt-cash)/d["ebitda"]) if d["ebitda"] and d["ebitda"]>0 else "N/A"),
    ("Risk-Free Rate", fmt_pct(rf)),
])

# EV Bridge
if show_bridge:
    st.markdown('<p class="section-title">Enterprise Value to Equity Bridge</p>', unsafe_allow_html=True)
    bridge_items = [
        ("Enterprise Value",    d["enterprise_value"],  "#3b82f6"),
        ("(-) Total Debt",      -debt,                  "#dc2626"),
        ("(+) Cash",            cash,                   "#16a34a"),
        ("(-) Minority Int.",   -d["minority_interest"],"#f59e0b"),
        ("(-) Preferred Stock", -d["preferred_stock"],  "#8b5cf6"),
        ("= Equity Value",      d["mkt_cap"],           "#1a1a2e"),
    ]
    fig_bridge = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","relative","relative","relative","total"],
        x=[b[0] for b in bridge_items],
        y=[b[1] for b in bridge_items],
        connector=dict(line=dict(color="#d1d5db", width=1)),
        increasing=dict(marker_color="#16a34a"),
        decreasing=dict(marker_color="#dc2626"),
        totals=dict(marker_color="#3b82f6"),
        text=[fmt_B(b[1]) for b in bridge_items],
        textposition="outside",
    ))
    fig_bridge.update_layout(
        height=340, plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="Value ($)", gridcolor="#f0f0f0"),
        showlegend=False, margin=dict(t=20,b=40),
    )
    st.plotly_chart(fig_bridge, use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# VALUATION TABS
# ═══════════════════════════════════════════════════════════════
st.markdown('<p class="section-title">Valuation Analysis</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "DCF Analysis", "Scenarios", "Reverse DCF",
    "Comps", "DDM", "Football Field", "Sensitivity"
])

all_dcf      = {}
mc_results   = None
mc_p10 = mc_p50 = mc_p90 = 0.0
comps_df     = pd.DataFrame()
gordon_val   = None
div_annual   = d["dividend"] or 0
active_fcf   = fcf_base_margin if use_margin_dcf else fcf_base


# ══════════════════════════
# TAB 1 — DCF
# ══════════════════════════
with tab1:
    st.markdown(
        f"**FCF Base:** {fmt_B(active_fcf)} &nbsp;|&nbsp; "
        f"**WACC:** {fmt_pct(wacc)} &nbsp;|&nbsp; "
        f"**Stage 1 Growth:** {fmt_pct(g1)} &nbsp;|&nbsp; "
        f"**Terminal Growth:** {fmt_pct(terminal_g)} &nbsp;|&nbsp; "
        f"**Rf:** {fmt_pct(rf)} &nbsp;|&nbsp; "
        f"**Beta:** {float(d['beta']):.2f} &nbsp;|&nbsp; "
        f"**Tax Rate:** {tax_rate_pct}%"
    )

    if use_margin_dcf:
        st.caption(
            f"Margin-based build: Revenue {fmt_B(d['revenue'])} x "
            f"EBITDA Margin {ebitda_margin_inp*100:.0f}% x "
            f"FCF Conv {fcf_conv_inp*100:.0f}% = FCF {fmt_B(active_fcf)}"
        )

    # 2-Stage
    if dcf_scenario in ["2-Stage","All Three"]:
        ps_2s, ev_2s, pvs_2s = dcf_2stage(
            active_fcf, g1, wacc, terminal_g, projection_years, debt, cash, shares
        )
        all_dcf["2-Stage DCF"] = ps_2s

        if dcf_scenario == "2-Stage":
            up = (ps_2s - current_price)/current_price*100 if current_price else 0
            ca, cb, cc = st.columns(3)
            ca.metric("Implied Share Price", f"${ps_2s:,.2f}")
            cb.metric("Current Price",       f"${current_price:,.2f}")
            cc.metric("Upside / Downside",   f"{up:+.1f}%",
                      delta_color="normal" if up > 0 else "inverse")

            yr_labels = [f"Yr {i+1}" for i in range(projection_years)]
            tv_pv     = max((ps_2s*shares) - sum(pvs_2s) + debt - cash, 0)
            fig = go.Figure()
            fig.add_trace(go.Bar(name="PV of FCF", x=yr_labels, y=pvs_2s,
                                 marker_color="#3b82f6"))
            fig.add_trace(go.Bar(name="PV of Terminal Value", x=["Terminal"],
                                 y=[tv_pv], marker_color="#8b5cf6"))
            fig.update_layout(title="DCF Value Build — PV of Cash Flows",
                              barmode="group", height=340,
                              plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(gridcolor="#f0f0f0"),
                              yaxis=dict(title="Value ($)", gridcolor="#f0f0f0",
                                         tickformat="$,.0f"))
            st.plotly_chart(fig, use_container_width=True)

    # 3-Stage
    if dcf_scenario in ["3-Stage","All Three"]:
        ps_3s, ev_3s, pvs_3s = dcf_3stage(
            active_fcf, g1, g2, wacc, terminal_g,
            3, max(projection_years-3,1), debt, cash, shares
        )
        all_dcf["3-Stage DCF"] = ps_3s

        if dcf_scenario == "3-Stage":
            up = (ps_3s - current_price)/current_price*100 if current_price else 0
            ca, cb, cc = st.columns(3)
            ca.metric("Implied Share Price", f"${ps_3s:,.2f}")
            cb.metric("Current Price",       f"${current_price:,.2f}")
            cc.metric("Upside / Downside",   f"{up:+.1f}%",
                      delta_color="normal" if up > 0 else "inverse")

    # Monte Carlo
    if dcf_scenario in ["Monte Carlo","All Three"]:
        raw = dcf_monte_carlo(
            active_fcf, g1, mc_g_std, wacc, mc_wacc_std, terminal_g,
            n=n_sims, years=projection_years, debt=debt, cash=cash, shares=shares
        )
        mc_results = raw[np.isfinite(raw)]
        mc_p10 = float(np.percentile(mc_results, 10))
        mc_p50 = float(np.percentile(mc_results, 50))
        mc_p90 = float(np.percentile(mc_results, 90))
        all_dcf["Monte Carlo (P10)"] = mc_p10
        all_dcf["Monte Carlo (P50)"] = mc_p50
        all_dcf["Monte Carlo (P90)"] = mc_p90

        if dcf_scenario == "Monte Carlo":
            ca, cb, cc, cd = st.columns(4)
            ca.metric("P10 — Bear",            f"${mc_p10:,.2f}")
            cb.metric("P50 — Base",            f"${mc_p50:,.2f}")
            cc.metric("P90 — Bull",            f"${mc_p90:,.2f}")
            cd.metric("Probability > Current",
                      f"{(mc_results > current_price).mean()*100:.0f}%")

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Histogram(x=mc_results, nbinsx=80, name="Simulations",
                                       marker_color="#3b82f6", opacity=0.7))
        fig_mc.add_vline(x=current_price, line_dash="dash", line_color="#dc2626",
                         annotation_text=f"Current: ${current_price:.2f}",
                         annotation_position="top right")
        fig_mc.add_vline(x=mc_p50, line_dash="dot", line_color="#16a34a",
                         annotation_text=f"P50: ${mc_p50:.2f}",
                         annotation_position="top left")
        fig_mc.update_layout(title=f"Monte Carlo DCF — {n_sims:,} Simulations",
                              xaxis_title="Implied Share Price ($)", yaxis_title="Frequency",
                              height=380, plot_bgcolor="white", paper_bgcolor="white",
                              xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"))
        st.plotly_chart(fig_mc, use_container_width=True)

    # All Three summary
    if dcf_scenario == "All Three":
        st.markdown("#### DCF Model Comparison")
        rows_cmp = []
        for model, price in [
            ("2-Stage DCF",     all_dcf.get("2-Stage DCF")),
            ("3-Stage DCF",     all_dcf.get("3-Stage DCF")),
            ("Monte Carlo P10", mc_p10),
            ("Monte Carlo P50", mc_p50),
            ("Monte Carlo P90", mc_p90),
        ]:
            up_str = (f"{(price-current_price)/current_price*100:+.1f}%"
                      if price and current_price else "N/A")
            rows_cmp.append({"Model": model,
                              "Implied Price": f"${price:,.2f}" if price else "N/A",
                              "Upside (%)": up_str})
        st.dataframe(pd.DataFrame(rows_cmp), use_container_width=True, hide_index=True)

        if mc_results is not None:
            fig_all = go.Figure()
            fig_all.add_trace(go.Histogram(x=mc_results, nbinsx=80,
                                            marker_color="#3b82f6", opacity=0.65,
                                            name="MC Simulations"))
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
            fig_all.update_layout(title="All DCF Models vs Monte Carlo Distribution",
                                   height=380, plot_bgcolor="white", paper_bgcolor="white",
                                   xaxis=dict(title="Implied Share Price ($)", gridcolor="#f0f0f0"),
                                   yaxis=dict(gridcolor="#f0f0f0"))
            st.plotly_chart(fig_all, use_container_width=True)

    with st.expander("WACC Decomposition"):
        wd = pd.DataFrame({
            "Component":    ["Cost of Equity","After-tax Cost of Debt","Blended WACC"],
            "Rate":         [fmt_pct(cost_eq), fmt_pct(cost_dbt*(1-tax_rate)), fmt_pct(wacc)],
            "Weight":       [fmt_pct(e_wt), fmt_pct(d_wt), "100%"],
            "Contribution": [fmt_pct(cost_eq*e_wt), fmt_pct(cost_dbt*(1-tax_rate)*d_wt), fmt_pct(wacc)],
        })
        st.dataframe(wd, hide_index=True, use_container_width=True)
        st.caption(
            f"CAPM: Rf {fmt_pct(rf)} + Beta {float(d['beta']):.2f} x ERP 5.5% = "
            f"Cost of Equity {fmt_pct(cost_eq)}"
        )


# ══════════════════════════
# TAB 2 — SCENARIOS
# ══════════════════════════
with tab2:
    st.markdown("#### Bull / Base / Bear Scenario Comparison")
    st.caption(
        "Each scenario runs an independent 2-stage DCF with its own growth rate, "
        "terminal growth, and WACC. Adjust inputs in the sidebar."
    )

    scenarios = {
        "Bear Case": (bear_g,    bear_tg_v,  bear_wacc),
        "Base Case": (g1,        terminal_g, wacc),
        "Bull Case": (bull_g,    bull_tg_v,  bull_wacc),
    }
    colors_sc = {"Bear Case": "#dc2626", "Base Case": "#3b82f6", "Bull Case": "#16a34a"}
    sc_results = {}

    for sc_name, (sc_g, sc_tg, sc_w) in scenarios.items():
        ps, ev, pvs = dcf_2stage(active_fcf, sc_g, sc_w, sc_tg,
                                  projection_years, debt, cash, shares)
        sc_results[sc_name] = {"price": ps, "ev": ev, "g": sc_g,
                                "tg": sc_tg, "wacc": sc_w}

    # Scenario KPI cards
    ca, cb, cc = st.columns(3)
    for col, (sc_name, res) in zip([ca, cb, cc], sc_results.items()):
        price = res["price"]
        up    = (price - current_price)/current_price*100 if current_price else 0
        col.metric(sc_name, f"${price:,.2f}", f"{up:+.1f}%",
                   delta_color="normal" if up > 0 else "inverse")

    # Scenario assumptions table
    sc_table = []
    for sc_name, res in sc_results.items():
        sc_table.append({
            "Scenario":        sc_name,
            "FCF Growth Yr 1": fmt_pct(res["g"]),
            "Terminal Growth": fmt_pct(res["tg"]),
            "WACC":            fmt_pct(res["wacc"]),
            "Implied Price":   f"${res['price']:,.2f}",
            "Upside":          f"{(res['price']-current_price)/current_price*100:+.1f}%",
            "Implied EV":      fmt_B(res["ev"]),
        })
    st.dataframe(pd.DataFrame(sc_table), use_container_width=True, hide_index=True)

    # Scenario bar chart
    fig_sc = go.Figure()
    for sc_name, res in sc_results.items():
        fig_sc.add_trace(go.Bar(
            name=sc_name, x=[sc_name], y=[res["price"]],
            marker_color=colors_sc[sc_name], opacity=0.8,
            text=f"${res['price']:,.2f}", textposition="outside",
        ))
    fig_sc.add_hline(y=current_price, line_dash="dash", line_color="#6b7280",
                     annotation_text=f"Current: ${current_price:.2f}")
    fig_sc.update_layout(
        title="Scenario Implied Share Price vs Current",
        yaxis=dict(title="Implied Price ($)", gridcolor="#f0f0f0", tickprefix="$"),
        height=380, plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False, margin=dict(t=50,b=40),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Tornado chart: show sensitivity of each assumption
    st.markdown("#### Assumption Sensitivity (Tornado)")
    base_price = sc_results["Base Case"]["price"]
    tornado = []
    # WACC swing
    p_lo,_,_ = dcf_2stage(active_fcf, g1, wacc+0.02, terminal_g, projection_years, debt, cash, shares)
    p_hi,_,_ = dcf_2stage(active_fcf, g1, wacc-0.02, terminal_g, projection_years, debt, cash, shares)
    tornado.append(("WACC +/-2pp", p_lo-base_price, p_hi-base_price))
    # Growth swing
    p_lo,_,_ = dcf_2stage(active_fcf, max(g1-0.05,0), wacc, terminal_g, projection_years, debt, cash, shares)
    p_hi,_,_ = dcf_2stage(active_fcf, g1+0.05, wacc, terminal_g, projection_years, debt, cash, shares)
    tornado.append(("FCF Growth +/-5pp", p_lo-base_price, p_hi-base_price))
    # Terminal growth swing
    p_lo,_,_ = dcf_2stage(active_fcf, g1, wacc, max(terminal_g-0.01,0.005), projection_years, debt, cash, shares)
    p_hi,_,_ = dcf_2stage(active_fcf, g1, wacc, min(terminal_g+0.01,wacc-0.005), projection_years, debt, cash, shares)
    tornado.append(("Terminal Growth +/-1pp", p_lo-base_price, p_hi-base_price))
    # FCF swing
    p_lo,_,_ = dcf_2stage(active_fcf*0.8, g1, wacc, terminal_g, projection_years, debt, cash, shares)
    p_hi,_,_ = dcf_2stage(active_fcf*1.2, g1, wacc, terminal_g, projection_years, debt, cash, shares)
    tornado.append(("FCF Base +/-20%", p_lo-base_price, p_hi-base_price))

    fig_tor = go.Figure()
    for label, lo_delta, hi_delta in tornado:
        fig_tor.add_trace(go.Bar(x=[lo_delta], y=[label], orientation="h",
                                  marker_color="#dc2626", opacity=0.75, showlegend=False))
        fig_tor.add_trace(go.Bar(x=[hi_delta], y=[label], orientation="h",
                                  marker_color="#16a34a", opacity=0.75, showlegend=False))
    fig_tor.add_vline(x=0, line_color="#1a1a2e", line_width=1.5)
    fig_tor.update_layout(
        title=f"Impact on Base Implied Price (${base_price:,.2f})",
        xaxis=dict(title="Change in Implied Price ($)", gridcolor="#f0f0f0", tickprefix="$"),
        yaxis=dict(autorange="reversed"),
        barmode="overlay", height=320,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=180),
    )
    st.plotly_chart(fig_tor, use_container_width=True)


# ══════════════════════════
# TAB 3 — REVERSE DCF
# ══════════════════════════
with tab3:
    st.markdown("#### Reverse DCF — What Growth Rate Does the Market Imply?")
    st.markdown(
        "Instead of projecting forward, we solve for the FCF growth rate embedded in "
        "the current stock price. This tells you what you need to **believe** to justify "
        "buying at today's price."
    )

    implied_g = reverse_dcf(current_price, shares, debt, cash, active_fcf,
                            wacc, terminal_g, projection_years)

    ca, cb, cc, cd = st.columns(4)
    ca.metric("Market-Implied Growth Rate", fmt_pct(implied_g))
    cb.metric("Your Base Case Growth",      fmt_pct(g1))
    cc.metric("Historical Revenue CAGR",    fmt_pct(hist_growth))
    cd.metric("Analyst Est. Growth",        fmt_pct(d["earnings_growth"]))

    # Interpretation
    gap = implied_g - g1
    if gap > 0.05:
        interp = (f"The market is pricing in {fmt_pct(implied_g)} growth — "
                  f"{fmt_pct(gap)} above your base case of {fmt_pct(g1)}. "
                  "The stock appears expensive relative to your assumptions.")
        interp_color = "#991b1b"
    elif gap < -0.05:
        interp = (f"The market is pricing in only {fmt_pct(implied_g)} growth — "
                  f"{fmt_pct(abs(gap))} below your base case of {fmt_pct(g1)}. "
                  "The stock may offer value relative to your assumptions.")
        interp_color = "#14532d"
    else:
        interp = (f"The market-implied growth of {fmt_pct(implied_g)} is broadly in "
                  f"line with your base case of {fmt_pct(g1)}. "
                  "The stock appears fairly valued on these assumptions.")
        interp_color = "#1e3a5f"

    st.markdown(
        f"<div style='background:#f8fafc;border-left:4px solid {interp_color};"
        f"padding:12px 16px;border-radius:4px;margin:12px 0;color:{interp_color};"
        f"font-size:0.9rem;'>{interp}</div>",
        unsafe_allow_html=True
    )

    # Chart: sweep price vs implied growth
    price_range = np.linspace(current_price*0.5, current_price*2.0, 60)
    implied_gs  = [
        reverse_dcf(p, shares, debt, cash, active_fcf, wacc, terminal_g, projection_years)
        for p in price_range
    ]
    fig_rdcf = go.Figure()
    fig_rdcf.add_trace(go.Scatter(
        x=implied_gs, y=price_range, mode="lines",
        line=dict(color="#3b82f6", width=2), name="Implied Growth"
    ))
    fig_rdcf.add_hline(y=current_price, line_dash="dash", line_color="#dc2626",
                       annotation_text=f"Current: ${current_price:.2f}")
    fig_rdcf.add_vline(x=g1, line_dash="dot", line_color="#16a34a",
                       annotation_text=f"Base: {g1*100:.1f}%")
    fig_rdcf.add_vline(x=hist_growth, line_dash="dot", line_color="#f59e0b",
                       annotation_text=f"Hist: {hist_growth*100:.1f}%")
    fig_rdcf.update_layout(
        title="Stock Price vs Market-Implied FCF Growth Rate",
        xaxis=dict(title="Implied FCF Growth Rate", tickformat=".1%", gridcolor="#f0f0f0"),
        yaxis=dict(title="Stock Price ($)", tickprefix="$", gridcolor="#f0f0f0"),
        height=400, plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_rdcf, use_container_width=True)

    # Sensitivity: implied growth vs WACC
    st.markdown("#### Implied Growth at Different WACC Levels")
    wacc_sweep = np.arange(wacc-0.03, wacc+0.035, 0.005)
    ig_sweep   = [
        reverse_dcf(current_price, shares, debt, cash, active_fcf,
                    w, terminal_g, projection_years)
        for w in wacc_sweep
    ]
    ig_df = pd.DataFrame({
        "WACC": [f"{w*100:.1f}%" for w in wacc_sweep],
        "Implied FCF Growth": [f"{g*100:.1f}%" for g in ig_sweep],
        "vs Base Case": [f"{(g-g1)*100:+.1f}pp" for g in ig_sweep],
    })
    st.dataframe(ig_df, use_container_width=True, hide_index=True)


# ══════════════════════════
# TAB 4 — COMPS
# ══════════════════════════
with tab4:
    sector     = d["sector"]
    base_comps = SECTOR_COMPS.get(sector, ["AAPL","MSFT","GOOGL","AMZN","META"])
    if manual_tickers:
        extra      = [t.strip().upper() for t in manual_tickers.split(",") if t.strip()]
        base_comps = list(dict.fromkeys(extra + base_comps))

    with st.spinner("Fetching comparable companies..."):
        comps_df = fetch_comps(tuple(base_comps[:8]), ticker)

    if comps_df.empty:
        st.warning("Could not fetch comp data. Try entering custom tickers in the sidebar.")
    else:
        st.markdown(f"**Sector:** {sector} &nbsp;|&nbsp; **Comps:** {', '.join(comps_df['Ticker'].tolist())}")

        # Display table
        disp = comps_df[["Ticker","Name","Mkt Cap ($B)","EV/EBITDA","EV/Revenue",
                          "P/E (TTM)","Fwd P/E","EBITDA Margin","Gross Margin",
                          "Rev Growth","ROE"]].copy()
        for col in ["EV/EBITDA","EV/Revenue","P/E (TTM)","Fwd P/E"]:
            disp[col] = disp[col].apply(lambda x: f"{x:.1f}x" if pd.notna(x) else "N/A")
        for col in ["EBITDA Margin","Gross Margin","Rev Growth","ROE"]:
            disp[col] = disp[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")

        em_subj = fmt_pct(d["ebitda"]/d["revenue"]) if d["revenue"] and d["revenue"]!=0 else "N/A"
        gm_subj = fmt_pct(d["gross_margin"]) if d["gross_margin"] else "N/A"
        subj    = pd.DataFrame([{
            "Ticker": f"[{ticker}]", "Name": d["name"][:22],
            "Mkt Cap ($B)": round((d["mkt_cap"] or 0)/1e9, 1),
            "EV/EBITDA":  fmt_x(d["ev_ebitda"]),  "EV/Revenue": fmt_x(d["ev_revenue"]),
            "P/E (TTM)":  fmt_x(d["pe"]),          "Fwd P/E":   fmt_x(d["fwd_pe"]),
            "EBITDA Margin": em_subj, "Gross Margin": gm_subj,
            "Rev Growth": fmt_pct(d["rev_growth"]), "ROE": fmt_pct(d["roe"]),
        }])
        st.dataframe(pd.concat([disp, subj], ignore_index=True),
                     use_container_width=True, hide_index=True)

        # Median multiples box
        st.markdown("#### Sector Median Multiples")
        med_cols = st.columns(4)
        for col, (metric, label) in zip(med_cols, [
            ("EV/EBITDA","EV/EBITDA"), ("EV/Revenue","EV/Revenue"),
            ("P/E (TTM)","P/E TTM"),   ("Fwd P/E","Fwd P/E"),
        ]):
            med = comps_df[metric].dropna().median()
            col.metric(f"Sector Median {label}", fmt_x(med) if pd.notna(med) else "N/A")

        # Implied prices
        cp = comps_implied(d, comps_df)
        if cp:
            st.markdown("#### Median-Multiple Implied Price")
            cp_cols = st.columns(len(cp))
            for col, (method, price) in zip(cp_cols, cp.items()):
                up = (price - current_price)/current_price*100
                col.metric(method, f"${price:,.2f}", f"{up:+.1f}%",
                           delta_color="normal" if up > 0 else "inverse")

        # Regression-implied price
        reg_price, reg_mult, r2 = comps_regression_implied(d, comps_df)
        if reg_price is not None:
            st.markdown("#### Regression-Implied Price (EV/EBITDA ~ Rev Growth)")
            ra, rb, rc = st.columns(3)
            ra.metric("Regression Implied Price", f"${reg_price:,.2f}",
                      f"{(reg_price-current_price)/current_price*100:+.1f}%",
                      delta_color="normal" if reg_price > current_price else "inverse")
            rb.metric("Regression-Implied Multiple", fmt_x(reg_mult))
            rc.metric("R-squared", f"{r2:.2f}")
            st.caption(
                "Fits a regression of EV/EBITDA on Revenue Growth across the comp set. "
                "Reads off the implied multiple for this company's growth rate. "
                "Low R-squared means the relationship is weak in this sector."
            )

        # Scatter: EV/EBITDA vs growth with regression line
        sub_sc = comps_df[["EV/EBITDA","Rev Growth","Ticker","Mkt Cap ($B)"]].dropna()
        if len(sub_sc) >= 3:
            x_sc = sub_sc["Rev Growth"].values
            y_sc = sub_sc["EV/EBITDA"].values
            sl, ic, _, _, _ = stats.linregress(x_sc, y_sc)
            x_fit = np.linspace(x_sc.min(), x_sc.max(), 50)
            y_fit = ic + sl * x_fit

            fig_sc2 = go.Figure()
            fig_sc2.add_trace(go.Scatter(
                x=x_fit*100, y=y_fit, mode="lines",
                line=dict(color="#6b7280", dash="dot", width=1.5),
                name="Regression", showlegend=True,
            ))
            for _, row in sub_sc.iterrows():
                sz = row["Mkt Cap ($B)"]**0.4*4 if row["Mkt Cap ($B)"] else 8
                fig_sc2.add_trace(go.Scatter(
                    x=[row["Rev Growth"]*100], y=[row["EV/EBITDA"]],
                    mode="markers+text", text=[row["Ticker"]],
                    textposition="top center",
                    marker=dict(size=sz, color="#3b82f6", opacity=0.7),
                    showlegend=False,
                ))
            if d["ev_ebitda"] and d["rev_growth"]:
                fig_sc2.add_trace(go.Scatter(
                    x=[d["rev_growth"]*100], y=[d["ev_ebitda"]],
                    mode="markers+text", text=[f"[{ticker}]"],
                    textposition="top center",
                    marker=dict(size=14, color="#dc2626", symbol="diamond"),
                    showlegend=False,
                ))
            fig_sc2.update_layout(
                title="EV/EBITDA vs Revenue Growth — Comps + Regression",
                xaxis_title="Revenue Growth (%)", yaxis_title="EV/EBITDA (x)",
                height=420, plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
            )
            st.plotly_chart(fig_sc2, use_container_width=True)


# ══════════════════════════
# TAB 5 — DDM
# ══════════════════════════
with tab5:
    if div_annual == 0:
        st.warning(
            f"{ticker} does not pay a dividend — DDM not applicable. "
            "Showing the implied dividend required to justify the current price."
        )
        implied_div = current_price*(cost_eq - terminal_g)/(1+terminal_g)
        st.metric("Implied Annual Dividend (for fair value at current price)",
                  f"${implied_div:.2f} / share")
        st.info("DDM is most meaningful for utilities, banks, consumer staples, and REITs.")
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
            up = (gordon_val-current_price)/current_price*100
            ca.metric("Gordon Growth Model", f"${gordon_val:,.2f}", f"{up:+.1f}%",
                      delta_color="normal" if up > 0 else "inverse")
        if ms_val:
            up2 = (ms_val-current_price)/current_price*100
            cb.metric("Multi-Stage DDM", f"${ms_val:,.2f}", f"{up2:+.1f}%",
                      delta_color="normal" if up2 > 0 else "inverse")
        cc.metric("Current Price", f"${current_price:,.2f}")

        ke_range  = np.arange(cost_eq-0.03, cost_eq+0.035, 0.005)
        ddm_curve = [ddm_gordon(div_annual, terminal_g, ke) or 0 for ke in ke_range]
        fig_ddm   = go.Figure()
        fig_ddm.add_trace(go.Scatter(x=ke_range*100, y=ddm_curve, mode="lines+markers",
                                      line=dict(color="#3b82f6", width=2), name="DDM Value"))
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


# ══════════════════════════
# TAB 6 — FOOTBALL FIELD
# ══════════════════════════
with tab6:
    bars = []

    if "2-Stage DCF" in all_dcf:
        ps = all_dcf["2-Stage DCF"]
        bars.append(("2-Stage DCF", ps*0.80, ps*1.20, ps))

    if "3-Stage DCF" in all_dcf:
        ps = all_dcf["3-Stage DCF"]
        bars.append(("3-Stage DCF", ps*0.80, ps*1.20, ps))

    if mc_results is not None and len(mc_results) > 0:
        bars.append(("Monte Carlo DCF", mc_p10, mc_p90, mc_p50))

    # Scenario bars
    if "sc_results" in dir():
        bars.append(("Scenario Range",
                     sc_results["Bear Case"]["price"],
                     sc_results["Bull Case"]["price"],
                     sc_results["Base Case"]["price"]))

    if not comps_df.empty:
        ev2eq = lambda ev: (ev-debt+cash)/shares if shares else 0
        ev_eb = comps_df["EV/EBITDA"].dropna()
        if len(ev_eb) >= 2 and d["ebitda"] and d["ebitda"] > 0:
            bars.append(("EV/EBITDA Comps",
                         ev2eq(d["ebitda"]*ev_eb.quantile(0.25)),
                         ev2eq(d["ebitda"]*ev_eb.quantile(0.75)),
                         ev2eq(d["ebitda"]*ev_eb.median())))
        ev_rv = comps_df["EV/Revenue"].dropna()
        if len(ev_rv) >= 2 and d["revenue"] and d["revenue"] > 0:
            bars.append(("EV/Revenue Comps",
                         ev2eq(d["revenue"]*ev_rv.quantile(0.25)),
                         ev2eq(d["revenue"]*ev_rv.quantile(0.75)),
                         ev2eq(d["revenue"]*ev_rv.median())))
        pe_v = comps_df["P/E (TTM)"].dropna()
        if len(pe_v) >= 2 and d["net_income"] and d["net_income"] > 0:
            bars.append(("P/E Comps",
                         (d["net_income"]*pe_v.quantile(0.25))/shares,
                         (d["net_income"]*pe_v.quantile(0.75))/shares,
                         (d["net_income"]*pe_v.median())/shares))
        if reg_price is not None:
            bars.append(("Regression Comps",
                         reg_price*0.85, reg_price*1.15, reg_price))

    if gordon_val and gordon_val > 0:
        bars.append(("DDM — Gordon Growth", gordon_val*0.85, gordon_val*1.15, gordon_val))

    if d["analyst_target"] and d["analyst_target"] > 0:
        at = d["analyst_target"]
        bars.append(("Analyst Consensus", at*0.85, at*1.15, at))

    bars = [
        (name, lo, hi, mid) for name, lo, hi, mid in bars
        if lo > 0 and hi > 0
        and hi < current_price*15 and lo < current_price*15
    ]

    if not bars:
        st.warning("Not enough valuation data to render the football field chart.")
    else:
        colors = ["#3b82f6","#6366f1","#8b5cf6","#0891b2",
                  "#059669","#d97706","#dc2626","#db2777","#0f766e"]

        fig_ff = go.Figure()
        for i, (name, lo, hi, mid) in enumerate(bars):
            c = colors[i % len(colors)]
            fig_ff.add_trace(go.Bar(
                name=name, x=[hi-lo], y=[name], base=[lo], orientation="h",
                marker=dict(color=c, opacity=0.7),
                text=f"${lo:,.1f} – ${hi:,.1f}",
                textposition="inside", insidetextanchor="middle",
                hovertemplate=f"<b>{name}</b><br>Low: ${lo:,.2f}<br>Mid: ${mid:,.2f}<br>High: ${hi:,.2f}<extra></extra>",
            ))
            fig_ff.add_trace(go.Scatter(
                x=[mid], y=[name], mode="markers",
                marker=dict(color="white", size=10, line=dict(color=c, width=2)),
                showlegend=False, hoverinfo="skip",
            ))

        fig_ff.add_vline(x=current_price, line_dash="solid", line_color="#dc2626", line_width=2,
                         annotation_text=f"Current: ${current_price:.2f}",
                         annotation_position="top right",
                         annotation=dict(font=dict(color="#dc2626", size=12)))

        all_lo = [b[1] for b in bars]; all_hi = [b[2] for b in bars]
        fig_ff.update_layout(
            title=f"{d['name']} ({ticker}) — Valuation Football Field",
            xaxis=dict(title="Implied Share Price ($)",
                       range=[max(0, min(all_lo)*0.85), max(all_hi)*1.15],
                       tickprefix="$", gridcolor="#f0f0f0", zeroline=False),
            yaxis=dict(autorange="reversed"),
            height=max(380, 80+len(bars)*60),
            plot_bgcolor="white", paper_bgcolor="white",
            barmode="overlay", showlegend=False,
            margin=dict(l=180, r=40, t=60, b=60),
        )
        st.plotly_chart(fig_ff, use_container_width=True)

        st.markdown("#### Implied Price Summary")
        summ_rows = []
        for name, lo, hi, mid in bars:
            up = (mid-current_price)/current_price*100
            summ_rows.append({"Method": name, "Bear Case": f"${lo:,.2f}",
                               "Base Case": f"${mid:,.2f}", "Bull Case": f"${hi:,.2f}",
                               "Upside to Base": f"{up:+.1f}%"})
        st.dataframe(pd.DataFrame(summ_rows), use_container_width=True, hide_index=True)

    # Excel export
    st.divider()
    st.markdown("#### Export")
    try:
        excel_bytes = build_excel(
            d, wacc, cost_eq, rf, terminal_g, g1, active_fcf,
            projection_years, all_dcf, comps_df if not comps_df.empty else None, bars
        )
        st.download_button(
            label="Download Full Model as Excel",
            data=excel_bytes,
            file_name=f"{ticker}_valuation_model.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as ex:
        st.caption(f"Excel export unavailable: {ex}")


# ══════════════════════════
# TAB 7 — SENSITIVITY
# ══════════════════════════
with tab7:
    st.markdown("#### DCF Sensitivity: Implied Price by WACC and Growth Rate")
    st.caption("Green = above current price. Red = below current price.")

    sens = sensitivity_table(active_fcf, wacc, g1, terminal_g, debt, cash, shares, projection_years)
    z    = sens.values.astype(float)
    x_lb = [f"{v*100:.1f}%" for v in sens.columns]
    y_lb = [f"{v*100:.1f}%" for v in sens.index]

    fig_h = go.Figure(data=go.Heatmap(
        z=z, x=x_lb, y=y_lb,
        colorscale=[[0.0,"#dc2626"],[0.3,"#fca5a5"],[0.5,"#fef3c7"],
                    [0.7,"#86efac"],[1.0,"#16a34a"]],
        zmid=current_price,
        text=[[f"${v:.0f}" if np.isfinite(v) else "N/A" for v in row] for row in z],
        texttemplate="%{text}", textfont=dict(size=10),
        hovertemplate="Growth: %{y}<br>WACC: %{x}<br>Price: %{text}<extra></extra>",
    ))
    fig_h.update_layout(
        xaxis_title="WACC", yaxis_title="FCF Growth Rate (Yr 1)",
        height=430, margin=dict(l=80, r=20, t=30, b=60),
    )
    st.plotly_chart(fig_h, use_container_width=True)

    st.markdown("#### Terminal Growth Rate Sensitivity")
    tg_range = np.arange(0.01, 0.04, 0.005)
    wacc_pts = [wacc-0.01, wacc, wacc+0.01]
    fig_tg   = go.Figure()
    for w in wacc_pts:
        pts = []
        for tg in tg_range:
            if w <= tg: pts.append(np.nan); continue
            p,_,_ = dcf_2stage(active_fcf, g1, w, tg, projection_years, debt, cash, shares)
            pts.append(p)
        fig_tg.add_trace(go.Scatter(x=tg_range*100, y=pts, mode="lines+markers",
                                     name=f"WACC = {w*100:.1f}%", line=dict(width=2)))
    fig_tg.add_hline(y=current_price, line_dash="dash", line_color="#dc2626",
                     annotation_text=f"Current: ${current_price:.2f}")
    fig_tg.update_layout(
        xaxis_title="Terminal Growth Rate (%)", yaxis_title="Implied Price ($)",
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_tg, use_container_width=True)

    st.markdown("#### EBITDA Margin Sensitivity (Margin-Based Build)")
    margin_range = np.arange(0.05, 0.55, 0.05)
    wacc_pts2    = [wacc-0.01, wacc, wacc+0.01]
    fig_mg       = go.Figure()
    for w in wacc_pts2:
        prices_mg = []
        for m in margin_range:
            fcf_m  = d["revenue"] * m * fcf_conv_inp if d["revenue"] else 0
            if fcf_m == 0: prices_mg.append(np.nan); continue
            p,_,_ = dcf_2stage(fcf_m, g1, w, terminal_g, projection_years, debt, cash, shares)
            prices_mg.append(p)
        fig_mg.add_trace(go.Scatter(x=margin_range*100, y=prices_mg, mode="lines+markers",
                                     name=f"WACC = {w*100:.1f}%", line=dict(width=2)))
    fig_mg.add_hline(y=current_price, line_dash="dash", line_color="#dc2626",
                     annotation_text=f"Current: ${current_price:.2f}")
    fig_mg.update_layout(
        title="Implied Price vs EBITDA Margin Assumption",
        xaxis_title="EBITDA Margin (%)", yaxis_title="Implied Price ($)",
        height=360, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(gridcolor="#f0f0f0"), yaxis=dict(gridcolor="#f0f0f0"),
        legend=dict(x=0.02, y=0.98),
    )
    st.plotly_chart(fig_mg, use_container_width=True)

    if show_raw:
        with st.expander("Raw Sensitivity Table"):
            raw_df = sens.copy()
            raw_df.columns = [f"WACC {v*100:.1f}%" for v in raw_df.columns]
            raw_df.index   = [f"Growth {v*100:.1f}%" for v in raw_df.index]
            st.dataframe(raw_df.style.format("${:.2f}"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.divider()
st.caption(
    f"Data: Yahoo Finance · FRED (10Y UST: {rf*100:.2f}%) · "
    "For educational and research purposes only. Not investment advice. "
    f"Refreshed: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}"
)
