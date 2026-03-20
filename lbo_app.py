# ═══════════════════════════════════════════════════════════════════════════
#  LBO Model Engine v2  |  Samaksh Sha  |  FLAME University
#  Full explanations on every input + output · Case summary tab
# ═══════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="LBO Model Engine · Samaksh Sha",
    page_icon="⚡", layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"]        { background:#0c0c0c; border-right:1px solid #1e1e1e; }
[data-testid="stSidebar"] *      { color:#e2e8f0 !important; }
.metric-card { background:#111; border:1px solid #1e1e1e; border-radius:8px;
               padding:14px 16px; text-align:center; }
.m-label { font-size:10px; color:#64748b; text-transform:uppercase;
            letter-spacing:.08em; margin-bottom:4px; }
.m-value { font-size:20px; font-weight:700; }
.m-sub   { font-size:11px; color:#475569; margin-top:2px; }
.sec     { font-size:10px; font-weight:600; text-transform:uppercase;
            letter-spacing:.09em; color:#475569; border-bottom:1px solid #1e1e1e;
            padding-bottom:5px; margin:18px 0 10px; }
.thesis  { background:#111; border-left:3px solid #3b82f6;
            border-radius:0 6px 6px 0; padding:10px 14px;
            font-size:12px; color:#94a3b8; margin:8px 0; }
.tag     { display:inline-block; font-size:10px; border-radius:4px;
            padding:2px 7px; margin:2px; }
.explain-box { background:#0d1117; border:1px solid #21262d; border-radius:8px;
               padding:14px 16px; margin:8px 0 14px; font-size:12px;
               color:#8b949e; line-height:1.7; }
.explain-box b { color:#58a6ff; font-weight:600; }
.explain-box .calc { background:#161b22; border-radius:4px; padding:4px 8px;
                     font-family:monospace; color:#79c0ff; font-size:11px;
                     display:inline-block; margin:3px 0; }
.explain-box ul  { margin:6px 0 6px 16px; padding:0; }
.explain-box li  { margin:3px 0; }
.verdict-good { background:#0f2318; border:1px solid #238636;
                border-radius:8px; padding:16px 20px; }
.verdict-warn { background:#1c1407; border:1px solid #9e6a03;
                border-radius:8px; padding:16px 20px; }
.verdict-bad  { background:#1a0909; border:1px solid #da3633;
                border-radius:8px; padding:16px 20px; }
.chart-explain { background:#0d1117; border-left:3px solid #388bfd;
                 border-radius:0 6px 6px 0; padding:10px 14px;
                 font-size:12px; color:#8b949e; margin:4px 0 14px; line-height:1.65; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ── EXPLANATION HELPERS ───────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def explain(title: str, what: str, why: str, how: str, benchmark: str = ""):
    """Render a collapsible explanation box beneath an input."""
    bench_html = (f"<li><b>Benchmarks:</b> {benchmark}</li>" if benchmark else "")
    st.markdown(f"""
    <div class="explain-box">
      <b>What is {title}?</b><br>{what}<br><br>
      <b>Why does it matter?</b><br>{why}<br><br>
      <b>How to calculate:</b><br>
      <span class="calc">{how}</span>
      <ul>{bench_html}</ul>
    </div>""", unsafe_allow_html=True)


INPUT_EXPLANATIONS = {
    "ltm_revenue": dict(
        title="LTM Revenue",
        what="Last Twelve Months Revenue — the total sales a company generated over the most recent 12-month period. It is always trailing (backward-looking), not forecast.",
        why="It is the starting point for every projection. PE firms use LTM (not annual) because it reflects the most current run-rate of the business — a company with June year-end acquired in October needs the 12 months ending October.",
        how="Sum the last 4 quarters of revenue from BSE/NSE filings. Or: Latest FY revenue + H1 current year revenue − H1 prior year revenue.",
        benchmark="Indian mid-cap LBO targets: ₹500–5,000 Cr. Large-cap take-privates (rare): ₹10,000 Cr+.",
    ),
    "ltm_ebitda_margin": dict(
        title="LTM EBITDA Margin",
        what="EBITDA (Earnings Before Interest, Taxes, Depreciation & Amortisation) as a percentage of Revenue. It is the most widely used proxy for operating cash generation in M&A and LBO analysis.",
        why="Debt in an LBO is sized as a multiple of EBITDA — not revenue or net income. So EBITDA margin directly determines how much leverage (and therefore financial engineering) is possible. A 5% margin company at 10x revenue = same EBITDA as a 20% margin company at 2.5x revenue.",
        how="EBITDA ÷ Revenue × 100. EBITDA = Operating Profit (EBIT) + D&A. From P&L: Revenue − COGS − SGA (exclude D&A and interest).",
        benchmark="IT services: 22–28%. QSR: 14–20%. Auto dealership: 4–7%. Consumer brands: 12–22%.",
    ),
    "entry_ev_ebitda": dict(
        title="Entry EV/EBITDA Multiple",
        what="Enterprise Value (the total price paid for the business including debt) divided by LTM EBITDA. This is the acquisition valuation multiple — analogous to a P/E ratio but capital-structure neutral.",
        why="This single number determines the total price. The higher the entry multiple, the more equity must be used (since debt capacity is fixed at ~5–6x EBITDA), and the harder it is to generate a 20%+ IRR. It is the #1 sensitivity driver.",
        how="Entry EV = LTM EBITDA × Entry Multiple. Derived from: comparable transaction multiples (comps) + DCF + premium-to-market.",
        benchmark="Indian IT: 18–25x. QSR: 16–22x. FMCG: 30–50x. Auto: 8–14x. Ideal LBO entry: 6–12x.",
    ),
    "revenue_growth": dict(
        title="Revenue CAGR",
        what="Compound Annual Growth Rate of revenue over the hold period. This is the single most important operating assumption — it drives every downstream number.",
        why="Revenue growth directly expands EBITDA (assuming stable margins), which lifts the exit EV. A 2pp change in CAGR over 5 years changes the exit equity value by 15–25% in a typical LBO. It is also how PE firms underwrite their investment thesis.",
        how="CAGR = (Exit Revenue ÷ Entry Revenue)^(1/Years) − 1. For projections: apply consistently each year or use a phased ramp.",
        benchmark="IT services India: 8–15%. QSR: 12–22%. Consumer: 10–18%. Be conservative — PE firms stress-test at entry CAGR − 3–4pp.",
    ),
    "ltm_ebitda_margin_exit": dict(
        title="Exit EBITDA Margin",
        what="The EBITDA margin in the final year of the hold period. If this is higher than the entry margin, it represents margin expansion — a key value creation lever.",
        why="Every 1pp of margin expansion on a large revenue base generates significant incremental EBITDA, which multiplies by the exit EV/EBITDA multiple to compound returns. PE firms specifically target operational improvements (cost rationalisation, pricing power, mix shift) to drive this.",
        how="No single formula. Built bottom-up: Entry margin + SGA efficiency gains + scale leverage + pricing + mix. Or simply assume linear expansion over hold period.",
        benchmark="Typical PE value-add: +2–5pp over 5 years. More than +5pp requires a strong operational thesis.",
    ),
    "da_pct": dict(
        title="D&A as % of Revenue",
        what="Depreciation & Amortisation divided by revenue. D&A is a non-cash charge — it reduces reported profit but not actual cash. Adding it back converts net income to operating cash flow.",
        why="D&A is the bridge between EBITDA and EBIT (EBITDA − D&A = EBIT). In FCF calculation, D&A is added back to net income because it was deducted for tax purposes but no cash actually left the business.",
        how="D&A = PPE depreciation + intangible amortisation. From cash flow statement. D&A % = Total D&A ÷ Revenue × 100.",
        benchmark="Asset-light (IT, QSR): 2–6%. Capital-intensive (manufacturing, auto): 6–12%. Higher D&A = more tax shield.",
    ),
    "capex_pct": dict(
        title="CapEx as % of Revenue",
        what="Capital Expenditure — cash spent on buying or maintaining physical assets (property, plant, equipment, technology infrastructure). Split into maintenance CapEx (keeping assets running) and growth CapEx (expanding capacity).",
        why="CapEx is a direct cash outflow that reduces Free Cash Flow. High-CapEx businesses generate less FCF per ₹ of EBITDA, which reduces debt repayment capacity and lowers LBO returns. Asset-light businesses (IT, QSR franchises) are more LBO-friendly precisely because CapEx is low.",
        how="CapEx from Cash Flow Statement (Investing Activities). CapEx % = CapEx ÷ Revenue × 100. Maintenance CapEx ≈ D&A for stable businesses.",
        benchmark="IT services: 2–4%. QSR franchisee: 6–9%. Manufacturing: 8–15%. Below 4% = very LBO-friendly.",
    ),
    "nwc_pct": dict(
        title="NWC as % of Revenue Change",
        what="Net Working Capital is the cash trapped in the day-to-day operations of the business: Receivables + Inventory − Payables. As revenue grows, more cash gets locked up in working capital, reducing FCF.",
        why="NWC investment is often ignored by students but matters significantly. A company growing ₹1,000 Cr in revenue with 5% NWC intensity uses ₹50 Cr of cash just to fund that growth — cash that cannot repay debt. Negative NWC businesses (like QSR — customers pay before suppliers are paid) generate cash from growth.",
        how="NWC % = ΔNWC ÷ ΔRevenue × 100. ΔNWC = (Current Assets ex-cash) − (Current Liabilities ex-debt). From balance sheet.",
        benchmark="IT services: 5–10% (high receivables). QSR: −5–2% (cash business). Retail: 3–8%.",
    ),
    "tax_rate": dict(
        title="Effective Tax Rate",
        what="The percentage of pre-tax profit paid as corporate income tax. In India, the base corporate tax rate is 22% + 10% surcharge + 4% cess = ~25.17% for domestic companies (Section 115BAA).",
        why="Tax reduces net income directly, and therefore reduces FCF. In an LBO, interest payments are tax-deductible (the 'interest tax shield') — meaning the government effectively subsidises part of the debt cost. Higher tax rates = bigger tax shield = more benefit from leverage.",
        how="Effective tax rate = Income Tax ÷ PBT × 100. Use 25.17% for most Indian companies under the new regime (Section 115BAA). Old regime: ~34.94%.",
        benchmark="India new regime: 25.17%. India old regime: 34.94%. Use actuals from company's P&L if available.",
    ),
    "senior_tla_turns": dict(
        title="TLA — Leverage Turns",
        what="Term Loan A is the most senior, cheapest, and first-to-be-repaid tranche of debt in an LBO. 'Turns' means multiples of EBITDA. 2.0x turns = TLA balance equals 2× the company's annual EBITDA.",
        why="TLA is the anchor debt — banks are comfortable lending up to 2–3x EBITDA at this seniority. It carries mandatory amortisation (scheduled repayments), so it gets paid down fastest. The more TLA you can raise, the cheaper the overall blended interest rate.",
        how="TLA Amount = LTM EBITDA × TLA Turns. Lender sizing based on: interest coverage (EBITDA ÷ Total interest ≥ 2.0x) and leverage (Total Debt ÷ EBITDA ≤ 5.5–6x for Indian market).",
        benchmark="Indian PE deals: TLA typically 1.5–2.5x EBITDA. Rate: 9–11% (MCLR-linked). Amortises over 5–7 years.",
    ),
    "senior_tlb_turns": dict(
        title="TLB — Leverage Turns",
        what="Term Loan B is junior to TLA — it gets repaid after TLA in a default. It carries a higher interest rate as compensation. In Indian PE deals, TLB is often structured as an NCDs (Non-Convertible Debentures) or subordinated bank debt.",
        why="TLB expands total debt capacity beyond TLA limits. Because it is bullet (minimal amortisation — typically 1% per year), it preserves more FCF for growth or TLA repayment. The tradeoff is a higher coupon rate.",
        how="TLB Amount = LTM EBITDA × TLB Turns. Total senior secured debt (TLA + TLB) typically limited to 3.5–4.5x EBITDA in India.",
        benchmark="TLB rate: 10–12.5%. Amort: 1% p.a. bullet. Combined TLA+TLB: 3–4.5x EBITDA.",
    ),
    "mezz_turns": dict(
        title="Mezzanine — Leverage Turns",
        what="Mezzanine is the most junior, highest-risk debt layer. It sits below TLA and TLB in the repayment waterfall. It can be structured as Cash Pay (interest paid every year) or PIK (Payment-In-Kind — interest accrues and is added to principal, paid at exit).",
        why="Mezz fills the gap between senior debt capacity and equity. It lets the sponsor use less equity (improving IRR) at the cost of a very high interest rate. PIK mezz is powerful because it does not reduce FCF during the hold — but it compounds, making the balance at exit much larger.",
        how="Mezz Amount = LTM EBITDA × Mezz Turns. Total leverage including mezz typically capped at 5–6x EBITDA. Mezz providers: NBFCs, PE credit funds, family offices.",
        benchmark="Mezz rate: 14–18%. PIK means no cash interest — balance grows each year at the rate. Total leverage: 3.5–5.5x typical.",
    ),
    "mezz_pik": dict(
        title="PIK Toggle (Payment-In-Kind)",
        what="When PIK is ON, the mezz interest is NOT paid in cash each year. Instead, it is added ('accrued') to the outstanding principal. So a ₹100 Cr mezz at 14% PIK becomes ₹114 Cr after Year 1, ₹129.96 Cr after Year 2, and so on — compounding.",
        why="PIK preserves free cash flow during the hold period (no cash interest paid), allowing more cash to sweep senior debt or fund growth. The catch: the mezz balance balloons significantly by exit, reducing equity proceeds.",
        how="Mezz closing balance = Opening balance × (1 + PIK rate). Cash impact in year: ₹0. Impact at exit: full compounded balance must be repaid from exit proceeds.",
        benchmark="PIK mezz common in highly-levered deals, growth-stage companies, or when FCF is insufficient for full cash interest. Watch: does PIK balance at exit eat your equity?",
    ),
    "cash_sweep": dict(
        title="Cash Sweep",
        what="A contractual provision that requires the company to use a defined percentage of excess free cash flow (after mandatory debt service) to prepay the TLA ahead of schedule.",
        why="The cash sweep accelerates debt paydown, reducing the leverage ratio faster. This improves the credit profile, reduces interest expense in later years (compounding the FCF improvement), and de-risks the deal. Most institutional LBO credit agreements include a 50–75% cash sweep.",
        how="Sweep amount = max(0, FCF − Mandatory Amort) × Sweep %. Applied first to TLA until fully repaid, then optionally to TLB.",
        benchmark="75% sweep is institutional standard. 50% for growth companies that need reinvestment capital. 100% in stressed situations.",
    ),
    "transaction_fees_pct": dict(
        title="Transaction Fees",
        what="The total one-time cost of executing the deal: investment bank advisory fees, legal fees, due diligence (accounting, commercial, technology), regulatory filings, and financing arrangement fees.",
        why="Transaction fees are a 'use of funds' — they are funded alongside the purchase price from the same debt+equity pool. They increase total equity invested without increasing enterprise value, directly diluting returns.",
        how="Transaction fees % = Total fees ÷ Entry EV × 100. Typical components: M&A advisory (0.5–1.0%), legal (0.3–0.5%), DD (0.2–0.4%), financing fees (0.5–1.0%).",
        benchmark="1.5–2.5% of EV for mid-market deals. 1–2% for large-cap. Higher on complex cross-border or contested auctions.",
    ),
    "exit_ev_ebitda": dict(
        title="Exit EV/EBITDA Multiple",
        what="The valuation multiple at which the PE firm sells the company at the end of the hold period. Combined with Year 5 EBITDA, this sets the exit enterprise value — the number from which debt is subtracted to get equity proceeds.",
        why="The exit multiple is the second most sensitive input after the entry multiple. Every 1x change in exit EV/EBITDA changes equity proceeds by Exit EBITDA × 1x. For a large EBITDA business, this is enormous. Conservative modelling: exit = entry multiple. Bull case: exit > entry (multiple expansion).",
        how="Exit EV = Y5 EBITDA × Exit Multiple. Exit multiple estimated from: comparable public company trading multiples at exit + expected market conditions + sector growth premium.",
        benchmark="Assume entry = exit as base case. Bull case: +1–2x. Bear case: −1–2x. Multiple expansion is a bonus, not a thesis.",
    ),
    "hold_period": dict(
        title="Hold Period",
        what="The number of years the PE firm owns the company between acquisition and exit. Most PE funds have a 10-year fund life, with investments typically held 3–7 years.",
        why="Hold period directly impacts IRR. A 2.5x MOIC over 3 years = 35% IRR. The same 2.5x over 7 years = 14% IRR. Shorter holds inflate IRR; longer holds require more absolute value creation. PE firms are incentivised to exit within 5–6 years.",
        how="Chosen based on: time needed for operational improvements, market conditions, fund lifecycle, and realistic exit path (IPO, strategic sale, secondary buyout).",
        benchmark="Median PE hold globally: 4.5–5.5 years. India PE: 4–7 years. Distributions: 3yr (quick flip), 5yr (standard), 7yr (complex turnaround).",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# ── LBO MATH ENGINE (unchanged) ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def run_lbo(p: dict) -> dict | None:
    yrs = int(p["hold_period"])
    ltm_ebitda  = p["ltm_revenue"] * p["ltm_ebitda_margin"] / 100
    entry_ev    = ltm_ebitda * p["entry_ev_ebitda"]
    tx_fees     = entry_ev  * p["transaction_fees_pct"] / 100
    total_uses  = entry_ev  + tx_fees

    tla_0   = ltm_ebitda * p["senior_tla_turns"]
    tlb_0   = ltm_ebitda * p["senior_tlb_turns"]
    mezz_0  = ltm_ebitda * p["mezz_turns"]
    debt_0  = tla_0 + tlb_0 + mezz_0
    equity_in = total_uses - debt_0
    if equity_in <= 0:
        return None
    equity_pct = equity_in / total_uses * 100

    m_start = p["ltm_ebitda_margin"]
    m_end   = p["ltm_ebitda_margin_exit"]
    rev_arr, ebitda_arr, da_arr, ebit_arr = [], [], [], []
    for yr in range(1, yrs + 1):
        rev    = p["ltm_revenue"] * (1 + p["revenue_growth"] / 100) ** yr
        margin = m_start + (m_end - m_start) * yr / yrs
        ebitda = rev * margin / 100
        da     = rev * p["da_pct"] / 100
        rev_arr.append(rev); ebitda_arr.append(ebitda)
        da_arr.append(da);   ebit_arr.append(ebitda - da)

    tla_bal  = tla_0
    tlb_bal  = tlb_0
    mezz_bal = mezz_0
    tla_annual_mand = tla_0 / max(p.get("tla_amort_years", yrs), 1)
    tlb_annual_mand = tlb_0 * 0.01

    D = {k: [] for k in [
        "tla_open","tla_int","tla_amort","tla_close",
        "tlb_open","tlb_int","tlb_amort","tlb_close",
        "mezz_open","mezz_int_cash","mezz_pik_accr","mezz_close",
        "total_debt","leverage","ebt","tax","ni","capex","nwc","fcf",
    ]}
    prev_rev = p["ltm_revenue"]

    for yr in range(yrs):
        D["tla_open"].append(tla_bal);  D["tlb_open"].append(tlb_bal)
        D["mezz_open"].append(mezz_bal)
        i_tla  = tla_bal  * p["tla_rate"]  / 100
        i_tlb  = tlb_bal  * p["tlb_rate"]  / 100
        i_mezz = mezz_bal * p["mezz_rate"] / 100
        D["tla_int"].append(i_tla);  D["tlb_int"].append(i_tlb)
        cash_mezz = 0.0 if p["mezz_pik"] else i_mezz
        pik_accr  = i_mezz if p["mezz_pik"] else 0.0
        D["mezz_int_cash"].append(cash_mezz); D["mezz_pik_accr"].append(pik_accr)

        total_int = i_tla + i_tlb + cash_mezz
        ebt  = ebit_arr[yr] - total_int
        tax  = max(0.0, ebt * p["tax_rate"] / 100)
        ni   = ebt - tax
        capex  = rev_arr[yr] * p["capex_pct"] / 100
        nwc_d  = (rev_arr[yr] - prev_rev) * p["nwc_pct"] / 100
        fcf    = ni + da_arr[yr] - capex - nwc_d
        D["ebt"].append(ebt); D["tax"].append(tax); D["ni"].append(ni)
        D["capex"].append(capex); D["nwc"].append(nwc_d); D["fcf"].append(fcf)

        tla_mand  = min(tla_annual_mand, tla_bal)
        tlb_mand  = min(tlb_annual_mand, tlb_bal)
        tla_sweep = 0.0
        cash_after = fcf - tla_mand - tlb_mand
        if p["cash_sweep"] and cash_after > 0:
            tla_sweep = min(cash_after * p["sweep_pct"] / 100, max(0, tla_bal - tla_mand))

        tla_tot = tla_mand + tla_sweep
        D["tla_amort"].append(tla_tot); D["tlb_amort"].append(tlb_mand)
        tla_bal  = max(0.0, tla_bal  - tla_tot)
        tlb_bal  = max(0.0, tlb_bal  - tlb_mand)
        mezz_bal = mezz_bal + pik_accr
        D["tla_close"].append(tla_bal); D["tlb_close"].append(tlb_bal)
        D["mezz_close"].append(mezz_bal)
        D["total_debt"].append(tla_bal + tlb_bal + mezz_bal)
        D["leverage"].append((tla_bal + tlb_bal + mezz_bal) / ebitda_arr[yr])
        prev_rev = rev_arr[yr]

    exit_ebitda  = ebitda_arr[-1]
    exit_ev      = exit_ebitda * p["exit_ev_ebitda"]
    debt_at_exit = tla_bal + tlb_bal + mezz_bal
    exit_equity  = max(0.0, exit_ev - debt_at_exit)
    moic = exit_equity / equity_in
    irr  = moic ** (1.0 / yrs) - 1

    return dict(
        ltm_ebitda=ltm_ebitda, entry_ev=entry_ev, tx_fees=tx_fees,
        total_uses=total_uses, tla_0=tla_0, tlb_0=tlb_0, mezz_0=mezz_0,
        debt_0=debt_0, equity_in=equity_in, equity_pct=equity_pct,
        entry_leverage=debt_0 / ltm_ebitda,
        rev=rev_arr, ebitda=ebitda_arr, da=da_arr, ebit=ebit_arr,
        D=D, yrs=yrs,
        exit_ebitda=exit_ebitda, exit_ev=exit_ev,
        debt_at_exit=debt_at_exit, exit_equity=exit_equity,
        exit_leverage=debt_at_exit / exit_ebitda if exit_ebitda else None,
        debt_repaid=debt_0 - debt_at_exit, moic=moic, irr=irr,
    )


def sensitivity_grid(p: dict):
    base_exit = p["exit_ev_ebitda"]; base_cagr = p["revenue_growth"]
    exits = [round(base_exit + d, 1) for d in (-1.5, -0.75, 0, 0.75, 1.5)]
    cagrs = [round(base_cagr + d, 1) for d in (-4, -2, 0, 2, 4)]
    irr_data, moic_data = {}, {}
    for cagr in cagrs:
        irrs, moics = [], []
        for em in exits:
            pp  = {**p, "exit_ev_ebitda": em, "revenue_growth": max(0.0, cagr)}
            res = run_lbo(pp)
            irrs.append(res["irr"]*100 if res else float("nan"))
            moics.append(res["moic"]   if res else float("nan"))
        col = f"{cagr:.0f}% CAGR"
        irr_data[col] = irrs; moic_data[col] = moics
    idx = [f"{e:.1f}x exit" for e in exits]
    return pd.DataFrame(irr_data, index=idx), pd.DataFrame(moic_data, index=idx)


# ═══════════════════════════════════════════════════════════════════════════
# ── COMPANIES DATABASE ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

COMPANIES = {
    "Devyani International": {
        "ticker":"DEVYANI.NS","flag":"🍕","sector":"QSR Franchisee",
        "desc":"Master franchisee of KFC, Pizza Hut & Costa Coffee in India. 700+ stores.",
        "thesis":"Scale rollout across Tier 2/3 cities, franchise unit economics improving, same-store-sales recovery post-COVID, SSSG 8–10% visible.",
        "tags":["Consumer","Asset-light","Recurring revenue"],
        "ltm_revenue":3200.0,"ltm_ebitda_margin":17.5,"entry_ev_ebitda":22.0,
        "revenue_growth":18.0,"ltm_ebitda_margin_exit":20.0,"da_pct":5.5,
        "capex_pct":7.5,"nwc_pct":1.0,"tax_rate":25.17,"senior_tla_turns":2.5,
        "tla_rate":9.5,"tla_amort_years":5,"senior_tlb_turns":1.0,"tlb_rate":10.5,
        "mezz_turns":0.5,"mezz_rate":14.0,"mezz_pik":True,"cash_sweep":True,
        "sweep_pct":75.0,"transaction_fees_pct":2.0,"hold_period":5,"exit_ev_ebitda":22.0,
    },
    "Sapphire Foods": {
        "ticker":"SAPPHIRFDS.NS","flag":"🍗","sector":"QSR Franchisee",
        "desc":"Largest KFC & Pizza Hut franchisee in South & West India + Sri Lanka.",
        "thesis":"Underpenetrated South India market, SSSG recovery, international optionality, digital ordering mix driving margin.",
        "tags":["Consumer","Franchise","South India moat"],
        "ltm_revenue":2800.0,"ltm_ebitda_margin":16.0,"entry_ev_ebitda":20.0,
        "revenue_growth":15.0,"ltm_ebitda_margin_exit":18.5,"da_pct":5.0,
        "capex_pct":7.0,"nwc_pct":1.0,"tax_rate":25.17,"senior_tla_turns":2.0,
        "tla_rate":9.5,"tla_amort_years":5,"senior_tlb_turns":1.0,"tlb_rate":10.5,
        "mezz_turns":0.5,"mezz_rate":14.0,"mezz_pik":True,"cash_sweep":True,
        "sweep_pct":75.0,"transaction_fees_pct":2.0,"hold_period":5,"exit_ev_ebitda":20.0,
    },
    "Barbeque Nation": {
        "ticker":"BARBEQUE.NS","flag":"🔥","sector":"Casual Dining",
        "desc":"India's largest casual dining chain. Live-grill-at-table concept. 200+ restaurants.",
        "thesis":"Unique dining format with high loyalty, premiumisation tailwind, international expansion, delivery channel untapped.",
        "tags":["Consumer","Differentiated format","International"],
        "ltm_revenue":1400.0,"ltm_ebitda_margin":14.5,"entry_ev_ebitda":16.0,
        "revenue_growth":12.0,"ltm_ebitda_margin_exit":17.0,"da_pct":6.0,
        "capex_pct":6.5,"nwc_pct":0.5,"tax_rate":25.17,"senior_tla_turns":2.0,
        "tla_rate":9.5,"tla_amort_years":5,"senior_tlb_turns":0.75,"tlb_rate":10.5,
        "mezz_turns":0.25,"mezz_rate":14.0,"mezz_pik":True,"cash_sweep":True,
        "sweep_pct":75.0,"transaction_fees_pct":2.0,"hold_period":5,"exit_ev_ebitda":16.0,
    },
    "Campus Activewear": {
        "ticker":"CAMPUS.NS","flag":"👟","sector":"Consumer Footwear",
        "desc":"India's largest sports & athleisure footwear brand. 30k+ MBO points + D2C.",
        "thesis":"Premiumisation of Indian footwear, D2C channel expanding margin, Tier 2/3 underpenetration, Nike/Adidas substitution.",
        "tags":["Consumer","D2C","Brand asset"],
        "ltm_revenue":1850.0,"ltm_ebitda_margin":13.0,"entry_ev_ebitda":18.0,
        "revenue_growth":14.0,"ltm_ebitda_margin_exit":16.0,"da_pct":3.0,
        "capex_pct":5.0,"nwc_pct":3.0,"tax_rate":25.17,"senior_tla_turns":1.75,
        "tla_rate":9.5,"tla_amort_years":5,"senior_tlb_turns":0.75,"tlb_rate":10.5,
        "mezz_turns":0.25,"mezz_rate":14.0,"mezz_pik":False,"cash_sweep":True,
        "sweep_pct":50.0,"transaction_fees_pct":2.0,"hold_period":5,"exit_ev_ebitda":18.0,
    },
    "Landmark Cars": {
        "ticker":"LANDMARK.NS","flag":"🚗","sector":"Premium Auto Dealership",
        "desc":"India's largest premium automotive retail group. Mercedes, Honda, Jeep, VW + more.",
        "thesis":"Rising premium car penetration, EV transition driving service revenue, authorised dealer relationships are defensible moats.",
        "tags":["Auto","Premium","Recurring service revenue"],
        "ltm_revenue":5200.0,"ltm_ebitda_margin":5.5,"entry_ev_ebitda":12.0,
        "revenue_growth":10.0,"ltm_ebitda_margin_exit":6.5,"da_pct":1.5,
        "capex_pct":2.0,"nwc_pct":2.0,"tax_rate":25.17,"senior_tla_turns":1.5,
        "tla_rate":9.5,"tla_amort_years":5,"senior_tlb_turns":0.5,"tlb_rate":10.5,
        "mezz_turns":0.25,"mezz_rate":14.0,"mezz_pik":False,"cash_sweep":True,
        "sweep_pct":75.0,"transaction_fees_pct":2.0,"hold_period":5,"exit_ev_ebitda":12.0,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# ── YFINANCE ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price(ticker):
    try:
        info  = yf.Ticker(ticker).info
        price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        mktcap = (info.get("marketCap") or 0) / 1e7
        return {"price": price, "mktcap": round(mktcap, 0)}
    except Exception:
        return {}

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_financials(ticker):
    try:
        t   = yf.Ticker(ticker)
        fin = t.financials
        if fin is None or fin.empty:
            return {}
        rev    = fin.loc["Total Revenue"].iloc[0] / 1e7 if "Total Revenue" in fin.index else None
        ebitda = fin.loc["EBITDA"].iloc[0] / 1e7        if "EBITDA"        in fin.index else None
        margin = ebitda / rev * 100 if (rev and ebitda and rev > 0) else None
        return {
            "revenue": round(rev, 1)    if rev    else None,
            "ebitda":  round(ebitda, 1) if ebitda else None,
            "margin":  round(margin, 1) if margin else None,
        }
    except Exception:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# ── CHART FACTORIES ───────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

_DARK = dict(plot_bgcolor="#0a0a0a", paper_bgcolor="#0a0a0a",
             font=dict(color="#94a3b8", size=11))
_GRID = dict(gridcolor="#1a1a1a", zerolinecolor="#222")
_MARG = dict(l=8, r=8, t=40, b=8)

def fig_debt_paydown(r):
    yrs = r["yrs"]; labels = ["Entry"] + [f"Y{i+1}" for i in range(yrs)]
    fig = go.Figure()
    for name, data, color in [
        ("TLA",  [r["tla_0"]]  + r["D"]["tla_close"],  "#3b82f6"),
        ("TLB",  [r["tlb_0"]]  + r["D"]["tlb_close"],  "#8b5cf6"),
        ("Mezz", [r["mezz_0"]] + r["D"]["mezz_close"], "#f59e0b"),
    ]:
        fig.add_trace(go.Bar(name=name, x=labels, y=data,
                             marker_color=color, marker_line_width=0))
    fig.update_layout(barmode="stack", title="Debt paydown (₹ Cr)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      legend=dict(bgcolor="#111", bordercolor="#222"),
                      margin=_MARG, height=300, **_DARK)
    return fig

def fig_ebitda_rev(r):
    yrs = r["yrs"]; labels = [f"Y{i+1}" for i in range(yrs)]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="Revenue", x=labels, y=r["rev"],
                         marker_color="#1e3a5f", marker_line_width=0), secondary_y=False)
    fig.add_trace(go.Bar(name="EBITDA",  x=labels, y=r["ebitda"],
                         marker_color="#22c55e", marker_line_width=0), secondary_y=False)
    margins = [e/rv*100 for e,rv in zip(r["ebitda"], r["rev"])]
    fig.add_trace(go.Scatter(name="EBITDA %", x=labels, y=margins,
                             mode="lines+markers", line=dict(color="#f59e0b", width=2),
                             marker=dict(size=6)), secondary_y=True)
    fig.update_layout(barmode="group", title="Revenue & EBITDA (₹ Cr)",
                      xaxis=dict(**_GRID),
                      yaxis=dict(title="₹ Cr", **_GRID),
                      yaxis2=dict(title="Margin %", **_GRID),
                      legend=dict(bgcolor="#111", bordercolor="#222"),
                      margin=_MARG, height=300, **_DARK)
    return fig

def fig_fcf(r):
    yrs = r["yrs"]; labels = [f"Y{i+1}" for i in range(yrs)]
    fcfs = r["D"]["fcf"]
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in fcfs]
    fig = go.Figure(go.Bar(x=labels, y=fcfs, marker_color=colors, marker_line_width=0,
                           text=[f"₹{v:,.0f}" for v in fcfs],
                           textposition="outside", textfont_size=10))
    fig.update_layout(title="Free cash flow (₹ Cr) — cash available after all obligations",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      margin=_MARG, height=280, **_DARK)
    return fig

def fig_waterfall(r):
    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute","relative","relative","absolute"],
        x=["Equity invested","EBITDA expansion","Debt repaid","Equity received"],
        y=[r["equity_in"], r["exit_ev"]-r["entry_ev"], r["debt_repaid"], None],
        text=[f"₹{r['equity_in']:,.0f}",None,None,f"₹{r['exit_equity']:,.0f}"],
        totals=dict(marker_color="#22c55e"),
        increasing=dict(marker_color="#3b82f6"),
        decreasing=dict(marker_color="#ef4444"),
        connector=dict(line=dict(color="#222", width=1)),
    ))
    fig.update_layout(title="Value creation waterfall (₹ Cr)",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      margin=_MARG, height=300, **_DARK)
    return fig

def fig_leverage(r):
    yrs = r["yrs"]; labels = [f"Y{i+1}" for i in range(yrs)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=r["D"]["leverage"],
                             mode="lines+markers+text",
                             line=dict(color="#f59e0b", width=2), marker=dict(size=8),
                             text=[f"{v:.1f}x" for v in r["D"]["leverage"]],
                             textposition="top center", textfont_size=10))
    fig.add_hline(y=5.0, line_dash="dash", line_color="rgba(239, 68, 68, 0.3125)",
                  annotation_text="5x comfort threshold", annotation_font_size=10)
    fig.add_hline(y=3.0, line_dash="dash", line_color="rgba(34, 197, 94, 0.3125)",
                  annotation_text="3x target", annotation_font_size=10)
    fig.update_layout(title="Net leverage (x EBITDA) — the deleveraging story",
                      xaxis=dict(**_GRID), yaxis=dict(**_GRID),
                      margin=_MARG, height=280, **_DARK)
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# ── SENSITIVITY RENDERER ─────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _irr_color(v):
    if np.isnan(v): return "background:#111;color:#444"
    if v >= 25:     return "background:#14532d;color:#86efac"
    if v >= 20:     return "background:#166534;color:#bbf7d0"
    if v >= 15:     return "background:#713f12;color:#fde68a"
    return              "background:#450a0a;color:#fca5a5"

def _moic_color(v):
    if np.isnan(v): return "background:#111;color:#444"
    if v >= 3.0:    return "background:#14532d;color:#86efac"
    if v >= 2.5:    return "background:#166534;color:#bbf7d0"
    if v >= 2.0:    return "background:#713f12;color:#fde68a"
    return              "background:#450a0a;color:#fca5a5"

def render_sensitivity(irr_df, moic_df, p):
    st.markdown("""
    <div class="chart-explain">
    <b>What is a sensitivity table?</b><br>
    A sensitivity table (also called a "two-way data table") shows how your <b>key output metric</b>
    (IRR or MOIC) changes when you simultaneously vary two input assumptions.
    Here the two variables are <b>Exit EV/EBITDA multiple</b> (rows) and <b>Revenue CAGR</b> (columns).
    Every cell is a complete, independent LBO run with those two parameters changed and everything
    else held constant.<br><br>
    <b>How to read it:</b> Find the cell where your base-case exit multiple and base-case CAGR intersect —
    that's your central scenario. Move right/up = bull case. Move left/down = bear case.
    PE investment committees require that at least 60–70% of the table stays green (≥20% IRR)
    before approving a deal.<br><br>
    <b>Colour legend:</b>
    🟢 IRR ≥25% — strong (fund's target return)  ·
    🟡 IRR ≥20% — acceptable  ·
    🟠 IRR ≥15% — borderline  ·
    🔴 IRR &lt;15% — fund would not proceed
    </div>
    """, unsafe_allow_html=True)

    def _html_table(df, color_fn, fmt):
        cols = list(df.columns)
        header = ("<tr><th style='color:#475569;padding:4px 10px;text-align:left'>"
                  "Exit / CAGR →</th>")
        for c in cols:
            header += f"<th style='color:#475569;padding:4px 10px'>{c}</th>"
        header += "</tr>"
        rows_html = ""
        for idx_label, row in df.iterrows():
            cells = f"<td style='color:#64748b;font-size:11px;padding:4px 10px'>{idx_label}</td>"
            for c in cols:
                v = row[c]; sty = color_fn(v)
                txt = f"{v:{fmt}}" if not np.isnan(v) else "—"
                cells += f"<td style='{sty};padding:5px 12px;border-radius:4px'>{txt}</td>"
            rows_html += f"<tr>{cells}</tr>"
        return (f"<table style='border-collapse:separate;border-spacing:3px;width:100%'>"
                f"<thead>{header}</thead><tbody>{rows_html}</tbody></table>")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**IRR (%)** — return rate to the PE sponsor")
        st.markdown(_html_table(irr_df, _irr_color, ".1f"), unsafe_allow_html=True)
    with c2:
        st.markdown("**MOIC (x)** — total money multiple on invested equity")
        st.markdown(_html_table(moic_df, _moic_color, ".2f"), unsafe_allow_html=True)

    st.caption(f"Base case: entry {p['entry_ev_ebitda']:.1f}x · exit {p['exit_ev_ebitda']:.1f}x "
               f"· CAGR {p['revenue_growth']:.1f}% · hold {int(p['hold_period'])}yr")


# ═══════════════════════════════════════════════════════════════════════════
# ── ASSUMPTIONS PANEL WITH INLINE EXPLANATIONS ───────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def assumptions_panel(d: dict, kp: str) -> dict:
    show_exp = st.toggle("Show detailed explanations for every input", value=False,
                         key=f"{kp}_exp_toggle")

    def _exp(key):
        if show_exp and key in INPUT_EXPLANATIONS:
            e = INPUT_EXPLANATIONS[key]
            explain(e["title"], e["what"], e["why"], e["how"], e.get("benchmark",""))

    with st.expander("⚙️  Deal assumptions", expanded=True):

        # ── Entry ─────────────────────────────────────────────────────────
        st.markdown('<p class="sec">Entry valuation</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        entry_mult = c1.number_input("Entry EV/EBITDA (x)", value=float(d["entry_ev_ebitda"]),
                                     min_value=2.0, max_value=60.0, step=0.5, key=f"{kp}_em")
        hold       = c2.selectbox("Hold period (yr)", [3,4,5,6,7],
                                  index=[3,4,5,6,7].index(int(d["hold_period"])), key=f"{kp}_hp")
        tx_fees    = c3.number_input("Transaction fees (%)", value=float(d["transaction_fees_pct"]),
                                     min_value=0.0, max_value=5.0, step=0.25, key=f"{kp}_tf")
        _exp("entry_ev_ebitda"); _exp("hold_period"); _exp("transaction_fees_pct")

        # ── Operating ─────────────────────────────────────────────────────
        st.markdown('<p class="sec">Operating assumptions</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        rev_g  = c1.number_input("Revenue CAGR (%)", value=float(d["revenue_growth"]),
                                  min_value=0.0, max_value=60.0, step=0.5, key=f"{kp}_rg")
        exit_m = c2.number_input("Exit EBITDA margin (%)", value=float(d["ltm_ebitda_margin_exit"]),
                                  min_value=1.0, max_value=60.0, step=0.5, key=f"{kp}_em2")
        tax    = c3.number_input("Tax rate (%)", value=float(d["tax_rate"]),
                                  min_value=0.0, max_value=40.0, step=0.5, key=f"{kp}_tax")
        _exp("revenue_growth"); _exp("ltm_ebitda_margin_exit"); _exp("tax_rate")

        c1, c2, c3 = st.columns(3)
        da_pct    = c1.number_input("D&A (% rev)",   value=float(d["da_pct"]),
                                    min_value=0.0, max_value=20.0, step=0.5, key=f"{kp}_da")
        capex_pct = c2.number_input("CapEx (% rev)", value=float(d["capex_pct"]),
                                    min_value=0.0, max_value=30.0, step=0.5, key=f"{kp}_cx")
        nwc_pct   = c3.number_input("NWC (% Δrev)",  value=float(d["nwc_pct"]),
                                    min_value=0.0, max_value=20.0, step=0.5, key=f"{kp}_nwc")
        _exp("da_pct"); _exp("capex_pct"); _exp("nwc_pct")

        # ── Capital structure ─────────────────────────────────────────────
        st.markdown('<p class="sec">Capital structure (debt sizing)</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        tla_turns = c1.number_input("TLA turns (x EBITDA)", value=float(d["senior_tla_turns"]),
                                    min_value=0.0, max_value=6.0, step=0.25, key=f"{kp}_tla_t")
        tlb_turns = c2.number_input("TLB turns (x EBITDA)", value=float(d["senior_tlb_turns"]),
                                    min_value=0.0, max_value=4.0, step=0.25, key=f"{kp}_tlb_t")
        mz_turns  = c3.number_input("Mezz turns (x EBITDA)", value=float(d["mezz_turns"]),
                                    min_value=0.0, max_value=3.0, step=0.25, key=f"{kp}_mz_t")
        _exp("senior_tla_turns"); _exp("senior_tlb_turns"); _exp("mezz_turns")

        c1, c2, c3 = st.columns(3)
        tla_r = c1.number_input("TLA rate (%)",  value=float(d["tla_rate"]),
                                 min_value=4.0, max_value=20.0, step=0.25, key=f"{kp}_tla_r")
        tlb_r = c2.number_input("TLB rate (%)",  value=float(d["tlb_rate"]),
                                 min_value=4.0, max_value=20.0, step=0.25, key=f"{kp}_tlb_r")
        mz_r  = c3.number_input("Mezz rate (%)", value=float(d["mezz_rate"]),
                                 min_value=4.0, max_value=25.0, step=0.25, key=f"{kp}_mz_r")

        c1, c2, c3 = st.columns(3)
        pik   = c1.checkbox("Mezz is PIK",  value=bool(d["mezz_pik"]),   key=f"{kp}_pik")
        sweep = c2.checkbox("Cash sweep",   value=bool(d["cash_sweep"]), key=f"{kp}_sw")
        sw_pct= c3.number_input("Sweep %",  value=float(d["sweep_pct"]),
                                 min_value=0.0, max_value=100.0, step=5.0,
                                 key=f"{kp}_swp", disabled=not sweep)
        _exp("mezz_pik"); _exp("cash_sweep")

        # ── Exit ──────────────────────────────────────────────────────────
        st.markdown('<p class="sec">Exit assumptions</p>', unsafe_allow_html=True)
        exit_mult = st.number_input("Exit EV/EBITDA (x)", value=float(d["exit_ev_ebitda"]),
                                    min_value=2.0, max_value=60.0, step=0.5, key=f"{kp}_exit")
        _exp("exit_ev_ebitda")

    return dict(
        entry_ev_ebitda=entry_mult, hold_period=hold, transaction_fees_pct=tx_fees,
        revenue_growth=rev_g, ltm_ebitda_margin_exit=exit_m, tax_rate=tax,
        da_pct=da_pct, capex_pct=capex_pct, nwc_pct=nwc_pct,
        senior_tla_turns=tla_turns, tla_rate=tla_r, tla_amort_years=hold,
        senior_tlb_turns=tlb_turns, tlb_rate=tlb_r,
        mezz_turns=mz_turns, mezz_rate=mz_r, mezz_pik=pik,
        cash_sweep=sweep, sweep_pct=sw_pct, exit_ev_ebitda=exit_mult,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ── METRIC CARD HELPER ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def _mc(col, label, value, sub="", color="#f1f5f9"):
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="m-label">{label}</div>'
        f'<div class="m-value" style="color:{color}">{value}</div>'
        f'<div class="m-sub">{sub}</div>'
        f'</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# ── CASE SUMMARY TAB ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def render_case_summary(r, p, name):
    yrs      = r["yrs"]
    irr_pct  = r["irr"] * 100
    moic     = r["moic"]
    ev_entry = r["entry_ev"]
    ev_exit  = r["exit_ev"]
    lev_in   = r["entry_leverage"]
    lev_out  = r["exit_leverage"] or 0

    # ── Verdict logic ─────────────────────────────────────────────────────
    if irr_pct >= 22 and moic >= 2.5:
        verdict_class = "verdict-good"
        verdict_icon  = "✅"
        verdict_text  = "INVEST — returns meet fund threshold"
        verdict_color = "#22c55e"
    elif irr_pct >= 17:
        verdict_class = "verdict-warn"
        verdict_icon  = "⚠️"
        verdict_text  = "BORDERLINE — requires negotiation or re-structuring"
        verdict_color = "#f59e0b"
    else:
        verdict_class = "verdict-bad"
        verdict_icon  = "❌"
        verdict_text  = "DO NOT INVEST — sub-threshold returns"
        verdict_color = "#ef4444"

    multiple_exp = p["exit_ev_ebitda"] - p["entry_ev_ebitda"]
    margin_exp   = p["ltm_ebitda_margin_exit"] - p["ltm_ebitda_margin"]
    debt_contrib = r["debt_repaid"] / (ev_exit - ev_entry + r["debt_repaid"]) * 100 if (ev_exit - ev_entry + r["debt_repaid"]) > 0 else 0
    ebitda_contrib = (ev_exit - ev_entry) / (ev_exit - ev_entry + r["debt_repaid"]) * 100 if (ev_exit - ev_entry + r["debt_repaid"]) > 0 else 0

    # ── FCF quality ───────────────────────────────────────────────────────
    avg_fcf = np.mean(r["D"]["fcf"])
    min_fcf = min(r["D"]["fcf"])
    fcf_positive = all(v > 0 for v in r["D"]["fcf"])

    # ── Interest coverage ─────────────────────────────────────────────────
    coverage_y1 = r["ebitda"][0] / (r["D"]["tla_int"][0] + r["D"]["tlb_int"][0]
                                    + r["D"]["mezz_int_cash"][0] + 0.001)
    coverage_y5 = r["ebitda"][-1] / (r["D"]["tla_int"][-1] + r["D"]["tlb_int"][-1]
                                     + r["D"]["mezz_int_cash"][-1] + 0.001)

    # ── Render ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="{verdict_class}" style="margin-bottom:18px">
      <div style="font-size:18px;font-weight:700;color:{verdict_color}">
        {verdict_icon} &nbsp; Investment verdict: {verdict_text}
      </div>
      <div style="font-size:12px;color:#94a3b8;margin-top:6px">
        IRR {irr_pct:.1f}% &nbsp;·&nbsp; MOIC {moic:.2f}x &nbsp;·&nbsp;
        Hold {yrs}yr &nbsp;·&nbsp; Entry {p['entry_ev_ebitda']:.1f}x →
        Exit {p['exit_ev_ebitda']:.1f}x
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── 1. Deal overview ──────────────────────────────────────────────────
    st.markdown("### Deal overview")
    st.markdown(f"""
    <div class="explain-box">
    This model analyses a <b>Leveraged Buyout of {name}</b> by a private equity sponsor.
    The transaction involves acquiring the company at <b>{p['entry_ev_ebitda']:.1f}x LTM EBITDA</b>,
    implying a total enterprise value of <b>₹{ev_entry:,.0f} Cr</b>. Including transaction fees of
    {p['transaction_fees_pct']:.1f}% (₹{r['tx_fees']:,.0f} Cr), total uses of funds
    amount to <b>₹{r['total_uses']:,.0f} Cr</b>.<br><br>

    The acquisition is funded with <b>₹{r['debt_0']:,.0f} Cr of debt</b>
    ({r['entry_leverage']:.1f}x EBITDA) structured across three tranches:
    Term Loan A (₹{r['tla_0']:,.0f} Cr at {p['tla_rate']:.1f}%),
    Term Loan B (₹{r['tlb_0']:,.0f} Cr at {p['tlb_rate']:.1f}%), and
    Mezzanine (₹{r['mezz_0']:,.0f} Cr at {p['mezz_rate']:.1f}%
    {'PIK' if p['mezz_pik'] else 'cash pay'}).
    The sponsor contributes <b>₹{r['equity_in']:,.0f} Cr of equity</b>
    ({r['equity_pct']:.1f}% of total uses).<br><br>

    Over a <b>{yrs}-year hold period</b>, the model projects revenue growing at
    {p['revenue_growth']:.1f}% CAGR from ₹{p['ltm_revenue']:,.0f} Cr to
    ₹{r['rev'][-1]:,.0f} Cr, with EBITDA margins expanding
    {margin_exp:+.1f}pp from {p['ltm_ebitda_margin']:.1f}% to
    {p['ltm_ebitda_margin_exit']:.1f}%. At exit, the company is sold at
    {p['exit_ev_ebitda']:.1f}x EBITDA, generating an exit EV of
    <b>₹{ev_exit:,.0f} Cr</b> and equity proceeds of
    <b>₹{r['exit_equity']:,.0f} Cr</b>.
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Returns analysis ───────────────────────────────────────────────
    st.markdown("### Returns analysis")
    c1, c2, c3, c4 = st.columns(4)
    _mc(c1, "IRR",    f"{irr_pct:.1f}%",  "Fund target ≥20%",   verdict_color)
    _mc(c2, "MOIC",   f"{moic:.2f}x",     "Target ≥2.5x",       verdict_color)
    _mc(c3, "Equity in",  f"₹{r['equity_in']:,.0f} Cr",  f"{r['equity_pct']:.0f}% of deal")
    _mc(c4, "Equity out", f"₹{r['exit_equity']:,.0f} Cr", f"after ₹{r['debt_at_exit']:,.0f} Cr debt")

    st.markdown(f"""
    <div class="explain-box" style="margin-top:12px">
    <b>What drives the {irr_pct:.1f}% IRR?</b><br>
    Returns in an LBO come from three levers — this deal is primarily driven by
    {'<b>EBITDA growth</b>' if ebitda_contrib >= debt_contrib else '<b>debt repayment</b>'}:<br><br>
    <ul>
    <li><b>EBITDA expansion:</b> Exit EBITDA of ₹{r['exit_ebitda']:,.0f} Cr vs entry
    ₹{r['ltm_ebitda']:,.0f} Cr — a {r['exit_ebitda']/r['ltm_ebitda']:.1f}x absolute
    increase. This is the product of {p['revenue_growth']:.1f}% revenue CAGR and
    {margin_exp:+.1f}pp margin expansion.</li>
    <li><b>Debt repayment:</b> Net debt reduced from ₹{r['debt_0']:,.0f} Cr to
    ₹{r['debt_at_exit']:,.0f} Cr — ₹{r['debt_repaid']:,.0f} Cr repaid, directly
    accreting to equity.
    {'Note: Mezz PIK grew during the hold, partially offsetting TLA/TLB repayments.' if p['mezz_pik'] else ''}</li>
    <li><b>Multiple {'expansion' if multiple_exp > 0 else 'contraction' if multiple_exp < 0 else 'neutral'}:</b>
    Entry {p['entry_ev_ebitda']:.1f}x vs exit {p['exit_ev_ebitda']:.1f}x
    ({multiple_exp:+.1f}x).
    {'This adds meaningful value — but conservative models assume no expansion.' if multiple_exp > 0
     else 'No multiple expansion assumed — a conservative and credible stance.' if multiple_exp == 0
     else 'Multiple contraction reduces returns — stress scenario.'}</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ── 3. Credit & risk analysis ─────────────────────────────────────────
    st.markdown("### Credit quality & risk assessment")

    risk_items = []

    # Leverage
    if lev_in > 5.5:
        risk_items.append(("🔴 High entry leverage", f"{lev_in:.1f}x EBITDA at entry — above 5.5x comfort threshold. Any EBITDA miss triggers covenant breach."))
    elif lev_in > 4.5:
        risk_items.append(("🟡 Moderate entry leverage", f"{lev_in:.1f}x EBITDA — within acceptable range for most Indian lenders."))
    else:
        risk_items.append(("🟢 Conservative leverage", f"{lev_in:.1f}x EBITDA — well within lender comfort zone. Strong defensive position."))

    # FCF
    if not fcf_positive:
        risk_items.append(("🔴 Negative FCF in projection", f"Year {[i+1 for i,v in enumerate(r['D']['fcf']) if v < 0]} shows negative FCF — company cannot service debt from operations."))
    else:
        risk_items.append(("🟢 Consistently positive FCF", f"Min FCF ₹{min_fcf:,.0f} Cr (Year 1) — company comfortably services all debt obligations throughout the hold."))

    # Interest coverage
    if coverage_y1 < 1.5:
        risk_items.append(("🔴 Thin interest coverage Y1", f"{coverage_y1:.1f}x EBITDA/interest — dangerously close to 1.0x. High default risk in Year 1."))
    elif coverage_y1 < 2.5:
        risk_items.append(("🟡 Adequate coverage Y1", f"{coverage_y1:.1f}x EBITDA/interest — above 1.5x minimum but limited buffer."))
    else:
        risk_items.append(("🟢 Strong interest coverage", f"Y1: {coverage_y1:.1f}x → Y5: {coverage_y5:.1f}x — ample headroom above 2.0x lender covenant."))

    # Equity cushion
    if r["equity_pct"] > 70:
        risk_items.append(("🟡 High equity contribution", f"{r['equity_pct']:.0f}% equity — minimal financial leverage. IRR is driven primarily by business growth, not financial engineering. More typical of a growth equity deal than a classic LBO."))
    elif r["equity_pct"] > 50:
        risk_items.append(("🟡 Moderate equity contribution", f"{r['equity_pct']:.0f}% equity — lower leverage than ideal LBO. Consider if more debt capacity exists."))
    else:
        risk_items.append(("🟢 Optimal equity contribution", f"{r['equity_pct']:.0f}% equity — classic LBO capital structure. Leverage is maximising returns."))

    # Exit leverage
    if lev_out > 4.0:
        risk_items.append(("🟡 Elevated exit leverage", f"{lev_out:.1f}x at exit — may limit buyer universe (strategic buyers prefer <3x). Could compress exit multiple."))
    else:
        risk_items.append(("🟢 Clean exit leverage", f"{lev_out:.1f}x at exit — attractive to all buyer types. No deleveraging premium required from buyer."))

    for icon_title, desc in risk_items:
        st.markdown(f"""
        <div style="background:#0d1117;border:1px solid #21262d;border-radius:6px;
                    padding:10px 14px;margin:5px 0;font-size:12px;color:#8b949e">
          <b style="color:#e6edf3">{icon_title}</b><br>{desc}
        </div>
        """, unsafe_allow_html=True)

    # ── 4. Value creation attribution ────────────────────────────────────
    st.markdown("### Value creation attribution")
    total_gain  = r["exit_equity"] - r["equity_in"]
    ebitda_gain = r["exit_ev"] - r["entry_ev"]
    debt_gain   = r["debt_repaid"]
    if total_gain > 0:
        st.markdown(f"""
        <div class="explain-box">
        Of the total equity value created (<b>₹{total_gain:,.0f} Cr</b> = exit equity − equity invested),
        approximately:
        <ul>
        <li><b>₹{ebitda_gain:,.0f} Cr ({ebitda_gain/total_gain*100:.0f}%)</b> came from
        EBITDA growth {'and multiple expansion' if multiple_exp > 0 else ''} —
        the operating performance of the business</li>
        <li><b>₹{debt_gain:,.0f} Cr ({debt_gain/total_gain*100:.0f}%)</b> came from
        net debt repayment — the financial engineering lever</li>
        </ul>
        {"⚠️ Note: The Mezz PIK accrual partially offset debt repayment. The gross TLA+TLB repayment was higher than the net figure above." if p['mezz_pik'] else ""}
        </div>
        """, unsafe_allow_html=True)

    # ── 5. Suggestions ────────────────────────────────────────────────────
    st.markdown("### Structuring suggestions & what-if scenarios")

    suggestions = []

    if irr_pct < 20:
        suggestions.append({
            "title": "Negotiate a lower entry multiple",
            "detail": f"The current {p['entry_ev_ebitda']:.1f}x entry is the dominant drag on returns. Dropping entry by 1x to {p['entry_ev_ebitda']-1:.1f}x reduces equity invested by ₹{r['ltm_ebitda']:,.0f} Cr, improving IRR by approximately 2–3pp.",
        })
    if r["equity_pct"] > 50:
        suggestions.append({
            "title": "Explore additional debt capacity",
            "detail": f"Equity contribution of {r['equity_pct']:.0f}% is high for an LBO. If lenders allow an additional 0.5–1.0x of mezz, equity drops by ₹{r['ltm_ebitda']*0.75:,.0f}–{r['ltm_ebitda']:,.0f} Cr, potentially adding 2–4pp to IRR. Requires strong interest coverage buffer.",
        })
    if p["exit_ev_ebitda"] <= p["entry_ev_ebitda"]:
        suggestions.append({
            "title": "Build a multiple expansion thesis",
            "detail": f"The model assumes no multiple expansion. If the business achieves a re-rating to {p['entry_ev_ebitda']+1.5:.1f}x at exit (driven by margin improvement, scale, ESG credentials), IRR improves by ~2–4pp. This requires a documented operating thesis.",
        })
    if margin_exp < 2.0:
        suggestions.append({
            "title": "Identify operational margin levers",
            "detail": f"Only {margin_exp:.1f}pp of margin expansion is assumed. Common levers in Indian consumer companies: vendor consolidation (−50–100bps COGS), tech-enabled SGA reduction (−30–60bps), delivery mix optimisation. Each additional 1pp of margin at exit adds ₹{r['rev'][-1]*0.01*p['exit_ev_ebitda']:,.0f} Cr to exit EV.",
        })
    if p["hold_period"] >= 6:
        suggestions.append({
            "title": "Consider a shorter hold period",
            "detail": f"A {yrs}-year hold compresses IRR significantly. If the same MOIC is achieved in {yrs-1} years, IRR improves from {irr_pct:.1f}% to {(moic**(1/(yrs-1))-1)*100:.1f}%. Explore whether an earlier exit (strategic sale at Y3–4) is achievable.",
        })

    suggestions.append({
        "title": "Stress test a downside scenario",
        "detail": f"Model a scenario where revenue CAGR falls to {max(0,p['revenue_growth']-4):.0f}% and exit multiple compresses to {p['exit_ev_ebitda']-1.5:.1f}x. If the resulting IRR is still above 12–15%, the deal has adequate downside protection. See sensitivity table for this cell.",
    })

    for s in suggestions:
        st.markdown(f"""
        <div style="background:#0d1117;border-left:3px solid #388bfd;
                    border-radius:0 6px 6px 0;padding:10px 14px;margin:6px 0;
                    font-size:12px;color:#8b949e;line-height:1.6">
          <b style="color:#58a6ff">{s['title']}</b><br>{s['detail']}
        </div>
        """, unsafe_allow_html=True)

    # ── 6. Key metrics summary table ──────────────────────────────────────
    st.markdown("### Key metrics at a glance")
    summary = pd.DataFrame({
        "Metric": [
            "Entry EV", "Entry EV/EBITDA", "Entry leverage",
            "Equity contributed", "Equity % of deal",
            "Revenue Y5", "EBITDA Y5", "Exit EV",
            "Debt at exit", "Exit leverage",
            "Equity at exit", "MOIC", "IRR",
        ],
        "Value": [
            f"₹{r['entry_ev']:,.0f} Cr",     f"{p['entry_ev_ebitda']:.1f}x",
            f"{r['entry_leverage']:.1f}x",
            f"₹{r['equity_in']:,.0f} Cr",    f"{r['equity_pct']:.1f}%",
            f"₹{r['rev'][-1]:,.0f} Cr",      f"₹{r['exit_ebitda']:,.0f} Cr",
            f"₹{r['exit_ev']:,.0f} Cr",
            f"₹{r['debt_at_exit']:,.0f} Cr", f"{lev_out:.1f}x",
            f"₹{r['exit_equity']:,.0f} Cr",  f"{moic:.2f}x",
            f"{irr_pct:.1f}%",
        ],
        "Flag": [
            "","",
            "🔴" if lev_in>5.5 else "🟡" if lev_in>4.5 else "🟢",
            "","🟡" if r['equity_pct']>55 else "🟢",
            "","","","","",
            "",
            "🟢" if moic>=2.5 else "🟡" if moic>=2.0 else "🔴",
            "🟢" if irr_pct>=20 else "🟡" if irr_pct>=15 else "🔴",
        ],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# ── RESULTS RENDERER ─────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def render_results(r, p, name):
    yrs   = r["yrs"]
    ylabs = [f"Y{i+1}" for i in range(yrs)]

    irr_col  = "#22c55e" if r["irr"] >= .20 else ("#f59e0b" if r["irr"] >= .15 else "#ef4444")
    moic_col = "#22c55e" if r["moic"] >= 2.5  else ("#f59e0b" if r["moic"] >= 2.0  else "#ef4444")

    st.markdown("---")
    st.markdown(f"## Results — {name}")

    cols = st.columns(6)
    _mc(cols[0], "IRR",           f"{r['irr']*100:.1f}%",     "Target ≥20%",    irr_col)
    _mc(cols[1], "MOIC",          f"{r['moic']:.2f}x",        "Target ≥2.5x",   moic_col)
    _mc(cols[2], "Entry EV",      f"₹{r['entry_ev']:,.0f} Cr", f"{p['entry_ev_ebitda']:.1f}x")
    _mc(cols[3], "Exit EV",       f"₹{r['exit_ev']:,.0f} Cr",  f"{p['exit_ev_ebitda']:.1f}x")
    _mc(cols[4], "Equity in",     f"₹{r['equity_in']:,.0f} Cr",f"{r['equity_pct']:.0f}% of uses")
    _mc(cols[5], "Equity out",    f"₹{r['exit_equity']:,.0f} Cr",f"Debt ₹{r['debt_at_exit']:,.0f} Cr")

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs(["📋 Sources & Uses", "📈 P&L", "🏦 Debt Schedule",
                    "📊 Charts", "🎯 Sensitivity", "📖 Case Summary"])

    # ── Tab 1: Sources & Uses ─────────────────────────────────────────────
    with tabs[0]:
        st.markdown("""
        <div class="chart-explain">
        <b>What is the Sources & Uses table?</b><br>
        This is the financial blueprint of the entire transaction. <b>Uses</b> = where the money
        goes (purchase price + fees). <b>Sources</b> = where the money comes from (debt tranches
        + equity). They must always balance to zero — Sources = Uses. This is the very first thing
        every PE analyst builds when a deal is proposed. The balance check is your first sanity
        check that the model is correct.
        </div>
        """, unsafe_allow_html=True)

        cu, cs = st.columns(2)
        with cu:
            st.markdown('<p class="sec">Uses of funds</p>', unsafe_allow_html=True)
            u = pd.DataFrame({
                "Item":    ["Purchase price (EV)", "Transaction fees", "Total uses"],
                "₹ Cr":   [r["entry_ev"], r["tx_fees"], r["total_uses"]],
                "% total": [r["entry_ev"]/r["total_uses"]*100,
                            r["tx_fees"]/r["total_uses"]*100, 100.0],
                "Explanation": [
                    "LTM EBITDA × entry multiple — what the seller receives",
                    "Banker/legal/DD fees — funded alongside purchase price",
                    "Must equal Total Sources",
                ],
            })
            u["₹ Cr"]    = u["₹ Cr"].map("₹{:,.1f}".format)
            u["% total"] = u["% total"].map("{:.1f}%".format)
            st.dataframe(u, hide_index=True, use_container_width=True)

        with cs:
            st.markdown('<p class="sec">Sources of funds</p>', unsafe_allow_html=True)
            s = pd.DataFrame({
                "Tranche": ["Term Loan A (TLA)", "Term Loan B (TLB)",
                            "Mezzanine", "Sponsor equity", "Total sources"],
                "₹ Cr":   [r["tla_0"], r["tlb_0"], r["mezz_0"], r["equity_in"], r["total_uses"]],
                "% total":[r["tla_0"]/r["total_uses"]*100, r["tlb_0"]/r["total_uses"]*100,
                            r["mezz_0"]/r["total_uses"]*100, r["equity_in"]/r["total_uses"]*100, 100.0],
                "Cost":   [f"{p['tla_rate']:.1f}%", f"{p['tlb_rate']:.1f}%",
                            f"{p['mezz_rate']:.1f}% {'PIK' if p['mezz_pik'] else 'Cash'}",
                            "Equity (residual)", ""],
                "Turns":  [f"{p['senior_tla_turns']:.2f}x", f"{p['senior_tlb_turns']:.2f}x",
                            f"{p['mezz_turns']:.2f}x", "—",
                            f"{r['entry_leverage']:.2f}x total"],
            })
            s["₹ Cr"]    = s["₹ Cr"].map("₹{:,.1f}".format)
            s["% total"] = s["% total"].map("{:.1f}%".format)
            st.dataframe(s, hide_index=True, use_container_width=True)

        st.markdown(f"""
        <div class="explain-box">
        <b>How to read this:</b> The blended cost of debt is approximately
        <span class="calc">
        ({p['tla_rate']:.1f}% × {r['tla_0']/r['debt_0']*100:.0f}%) +
        ({p['tlb_rate']:.1f}% × {r['tlb_0']/r['debt_0']*100:.0f}%) +
        ({p['mezz_rate']:.1f}% × {r['mezz_0']/r['debt_0']*100:.0f}%)
        = {(p['tla_rate']*r['tla_0']+p['tlb_rate']*r['tlb_0']+p['mezz_rate']*r['mezz_0'])/r['debt_0']:.1f}%
        </span><br><br>
        Entry leverage of <b>{r['entry_leverage']:.1f}x EBITDA</b> means:
        {'🔴 Above 5x — lenders will demand strong covenants and may require additional equity.' if r['entry_leverage'] > 5 else
         '🟡 4–5x — moderate leverage, acceptable for quality Indian businesses.' if r['entry_leverage'] > 4 else
         '🟢 Below 4x — conservative. Room to add more leverage if returns need improvement.'}
        Equity of {r['equity_pct']:.0f}% is
        {'🔴 very high for an LBO (typical: 35–45%). This is more growth equity than buyout.' if r['equity_pct'] > 65 else
         '🟡 slightly high — consider if more debt is available.' if r['equity_pct'] > 50 else
         '🟢 in the typical LBO range of 35–50%.'}
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 2: P&L ────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("""
        <div class="chart-explain">
        <b>How to read the LBO P&L:</b><br>
        This is a simplified income statement projected annually over the hold period.
        Each line tells a different story about the deal:<br><br>
        <b>Revenue → EBITDA:</b> The operating thesis — is the business growing and improving margins?<br>
        <b>EBITDA → EBIT:</b> Subtract D&A (non-cash). EBIT is used to calculate taxable income.<br>
        <b>EBIT → EBT:</b> Subtract interest — this is where leverage costs show up. Heavy debt = large interest charge = low EBT, especially in early years.<br>
        <b>EBT → Net Income:</b> After tax. This is accounting profit.<br>
        <b>Net Income → FCF:</b> Add back D&A (non-cash), subtract CapEx and working capital investment. FCF is what actually repays debt. This is the most important line for the credit story.
        </div>
        """, unsafe_allow_html=True)

        D  = r["D"]
        pl = {
            "Revenue (₹ Cr)":            [p["ltm_revenue"]] + r["rev"],
            "  Rev growth":              ["LTM"] + [
                f"{(r['rev'][i]/(r['rev'][i-1] if i>0 else p['ltm_revenue'])-1)*100:.1f}%"
                for i in range(yrs)],
            "EBITDA (₹ Cr)":             [r["ltm_ebitda"]] + r["ebitda"],
            "  EBITDA margin %":         [f"{p['ltm_ebitda_margin']:.1f}%"] + [
                f"{e/rv*100:.1f}%" for e,rv in zip(r["ebitda"],r["rev"])],
            "D&A (₹ Cr)":               ["—"] + [-v for v in r["da"]],
            "EBIT (₹ Cr)":              ["—"] + r["ebit"],
            "Interest — TLA (₹ Cr)":    ["—"] + [-v for v in D["tla_int"]],
            "Interest — TLB (₹ Cr)":    ["—"] + [-v for v in D["tlb_int"]],
            "Interest — Mezz (₹ Cr)":   ["—"] + [-v for v in D["mezz_int_cash"]],
            "  Mezz PIK accrual":        ["—"] + [-v for v in D["mezz_pik_accr"]],
            "EBT (₹ Cr)":               ["—"] + D["ebt"],
            "Tax @ 25.17% (₹ Cr)":      ["—"] + [-v for v in D["tax"]],
            "Net Income (₹ Cr)":        ["—"] + D["ni"],
            "— D&A add-back":           ["—"] + r["da"],
            "— CapEx (₹ Cr)":           ["—"] + [-v for v in D["capex"]],
            "— NWC change (₹ Cr)":      ["—"] + [-v for v in D["nwc"]],
            "Free Cash Flow (₹ Cr)":    ["—"] + D["fcf"],
        }
        df_pl = pd.DataFrame(pl, index=["LTM"] + ylabs).T
        st.dataframe(
            df_pl.map(lambda v: f"₹{v:,.1f}" if isinstance(v, float) else str(v)),
            use_container_width=True,
        )

        st.markdown(f"""
        <div class="explain-box" style="margin-top:10px">
        <b>What to look for:</b>
        <ul>
        <li><b>Interest burden:</b> In Y1, total cash interest = ₹{D['tla_int'][0]+D['tlb_int'][0]+D['mezz_int_cash'][0]:,.0f} Cr.
        EBITDA interest coverage = {r['ebitda'][0]/(D['tla_int'][0]+D['tlb_int'][0]+D['mezz_int_cash'][0]+0.001):.1f}x.
        Rule of thumb: lenders want ≥2.0x. Below 1.5x is danger territory.</li>
        <li><b>FCF conversion:</b> FCF ÷ EBITDA in Y5 =
        {D['fcf'][-1]/r['ebitda'][-1]*100:.0f}%. Higher = more cash available to repay debt.
        IT companies: 60–75%. QSR: 30–50%. Capital-heavy manufacturing: 15–30%.</li>
        <li><b>Mezz PIK non-cash charge:</b> ₹{D['mezz_pik_accr'][0]:,.0f} Cr per year
        accrues to principal but does NOT reduce FCF. This is why PIK mezz is attractive
        to the sponsor — but it silently grows your debt balance.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: Debt Schedule ──────────────────────────────────────────────
    with tabs[2]:
        st.markdown("""
        <div class="chart-explain">
        <b>What is the Debt Schedule?</b><br>
        The debt schedule is the most mechanical but critically important part of the LBO.
        It tracks every debt tranche year by year: opening balance → interest charged →
        repayment → closing balance.<br><br>
        <b>Three things to watch:</b><br>
        (1) <b>TLA amortisation:</b> How fast is the cheapest debt being paid down?
        Mandatory + sweep should retire it well before exit.<br>
        (2) <b>Mezz PIK compounding:</b> If PIK is on, the mezz balance grows every year.
        Watch for the exit mezz balance — it directly reduces equity proceeds.<br>
        (3) <b>Leverage trajectory:</b> Total debt ÷ EBITDA should fall materially each year.
        A deal that goes from 5x at entry to 2x at exit is a strong credit story.
        Flat leverage signals FCF is too thin or EBITDA growth isn't covering interest.
        </div>
        """, unsafe_allow_html=True)

        D = r["D"]
        ds = {
            "TLA — opening (₹ Cr)":       D["tla_open"],
            "TLA — interest (₹ Cr)":      [-v for v in D["tla_int"]],
            "TLA — amortisation (₹ Cr)":  [-v for v in D["tla_amort"]],
            "TLA — closing (₹ Cr)":       D["tla_close"],
            "TLB — opening (₹ Cr)":       D["tlb_open"],
            "TLB — interest (₹ Cr)":      [-v for v in D["tlb_int"]],
            "TLB — amortisation (₹ Cr)":  [-v for v in D["tlb_amort"]],
            "TLB — closing (₹ Cr)":       D["tlb_close"],
            "Mezz — opening (₹ Cr)":      D["mezz_open"],
            "Mezz — cash interest (₹ Cr)":[-v for v in D["mezz_int_cash"]],
            "Mezz — PIK accrual (₹ Cr)":  D["mezz_pik_accr"],
            "Mezz — closing (₹ Cr)":      D["mezz_close"],
            "Total debt (₹ Cr)":          D["total_debt"],
            "Net leverage (x EBITDA)":    D["leverage"],
            "FCF (₹ Cr)":                 D["fcf"],
        }
        df_ds = pd.DataFrame(ds, index=ylabs).T
        st.dataframe(
            df_ds.map(lambda v: f"{v:.2f}x" if (isinstance(v, float) and "leverage" in str(df_ds.index[df_ds.eq(v).any(axis=1)].tolist()))
                      else (f"₹{v:,.1f}" if isinstance(v, float) else str(v))),
            use_container_width=True,
        )

        lev_change = r["entry_leverage"] - (r["D"]["leverage"][-1])
        st.markdown(f"""
        <div class="explain-box" style="margin-top:10px">
        <b>Leverage journey:</b> Entry {r['entry_leverage']:.1f}x → Exit {r['D']['leverage'][-1]:.1f}x
        (Δ {lev_change:+.1f}x). {'🟢 Strong deleveraging story — good credit narrative.' if lev_change > 2 else '🟡 Moderate deleveraging.' if lev_change > 1 else '🔴 Minimal deleveraging — debt barely moved. Investigate why FCF is thin.'}<br><br>

        <b>PIK accrual effect:</b> Mezz opened at ₹{r['mezz_0']:,.0f} Cr and closes at
        ₹{D['mezz_close'][-1]:,.0f} Cr — {'grew by ₹' + f"{D['mezz_close'][-1]-r['mezz_0']:,.0f}" + ' Cr due to PIK compounding. This reduces equity at exit.' if p['mezz_pik'] else 'no PIK — mezz balance stayed flat (only 1% mandatory amort).'}<br><br>

        <b>TLA sweep:</b> In Y1, ₹{D['tla_amort'][0]:,.0f} Cr was repaid (mandatory +
        {p['sweep_pct']:.0f}% sweep of excess FCF).
        TLA {'was fully repaid by Y' + str(next((i+1 for i,v in enumerate(D['tla_close']) if v == 0), yrs)) if 0 in D['tla_close'] else 'was not fully repaid during the hold period'}.
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Charts ─────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("""
        <div class="chart-explain">
        Five charts — each tells one chapter of the LBO story.
        Read them in order for the full narrative.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.plotly_chart(fig_debt_paydown(r), use_container_width=True)
            st.markdown("""
            <div class="chart-explain">
            <b>Debt paydown chart:</b> Shows total debt (stacked by tranche) at entry and
            end of each year. A shrinking stack = deleveraging = value accruing to equity.
            Watch for the TLA bar (blue) disappearing — that's the cheap debt fully repaid.
            If the Mezz bar (yellow) grows, that's PIK compounding. Ideally total debt
            falls meaningfully by exit.
            </div>""", unsafe_allow_html=True)

        with c2:
            st.plotly_chart(fig_ebitda_rev(r), use_container_width=True)
            st.markdown("""
            <div class="chart-explain">
            <b>Revenue & EBITDA chart:</b> Blue bars = revenue. Green bars = EBITDA.
            Yellow line = EBITDA margin % (right axis). The operating thesis lives here.
            You want to see: (1) both bars growing consistently, (2) the margin line rising —
            that is the operational improvement story that justifies the deal.
            </div>""", unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            st.plotly_chart(fig_waterfall(r), use_container_width=True)
            st.markdown("""
            <div class="chart-explain">
            <b>Value creation waterfall:</b> Starts with equity invested, shows how each
            lever contributed, ends with equity received. The two middle bars are the two
            value creation engines: EBITDA growth (exit EV − entry EV) and debt repayment.
            If the EBITDA bar is much taller than the debt bar, returns are driven by
            business performance. If debt repayment dominates, it's financial engineering.
            </div>""", unsafe_allow_html=True)

        with c4:
            st.plotly_chart(fig_fcf(r), use_container_width=True)
            st.markdown("""
            <div class="chart-explain">
            <b>Free cash flow:</b> Cash generated after all operational obligations,
            CapEx and working capital. This is the fuel for debt repayment.
            Green = healthy. Red = the company cannot internally fund its debt service
            in that year — a credit red flag requiring covenant waiver or equity cure.
            FCF should grow each year as interest expense falls (debt being repaid).
            </div>""", unsafe_allow_html=True)

        st.plotly_chart(fig_leverage(r), use_container_width=True)
        st.markdown("""
        <div class="chart-explain">
        <b>Leverage trajectory:</b> This is the "credit story" in one chart.
        Net leverage = Total Debt ÷ EBITDA each year. It must decline to show that
        the company is becoming less risky over time. The red dashed line at 5x is the
        typical lender comfort ceiling — if leverage ever pierces this upward, you
        have a covenant breach. The green dashed line at 3x is a target for clean exit.
        A steeply falling curve signals strong FCF, fast debt repayment, and expanding
        EBITDA simultaneously — the hallmark of a successful LBO.
        </div>""", unsafe_allow_html=True)

    # ── Tab 5: Sensitivity ────────────────────────────────────────────────
    with tabs[4]:
        with st.spinner("Computing sensitivity grid…"):
            irr_df, moic_df = sensitivity_grid(p)
        render_sensitivity(irr_df, moic_df, p)

    # ── Tab 6: Case Summary ───────────────────────────────────────────────
    with tabs[5]:
        render_case_summary(r, p, name)


# ═══════════════════════════════════════════════════════════════════════════
# ── CUSTOM LBO WIZARD ────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def custom_lbo():
    st.markdown("## 🔧 Custom LBO — build your own deal")

    st.markdown("""
    <div class="explain-box">
    <b>How to use the custom LBO builder:</b><br>
    Enter a company name and optionally an NSE ticker (format: TICKER.NS, e.g. JUBLFOOD.NS).
    The model will attempt to auto-fetch LTM revenue and EBITDA margin from Yahoo Finance.
    If the auto-fetch fails or is inaccurate, override manually — the most important inputs
    are revenue and EBITDA margin; get these right and the rest follows logically.<br><br>
    All monetary values are in <b>₹ Crores (INR Cr)</b>. 1 Lakh Crore = ₹1 Trillion.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Step 1 — Company & LTM financials")
    c1, c2 = st.columns(2)
    ticker  = c1.text_input("NSE ticker (optional)", placeholder="e.g. JUBLFOOD.NS",
                             key="cust_ticker")
    co_name = c2.text_input("Company name", value="My Target Co", key="cust_name")

    fetched = {}
    if ticker:
        with st.spinner(f"Fetching {ticker} from Yahoo Finance…"):
            fetched = fetch_financials(ticker)
        if fetched.get("revenue"):
            st.success(
                f"✓ Auto-fetched: Revenue ₹{fetched['revenue']:,} Cr · "
                f"EBITDA Margin {fetched.get('margin','—')}% — override below if needed"
            )
        else:
            st.warning("Could not fetch — enter manually below.")

    c1, c2, c3 = st.columns(3)
    ltm_rev    = c1.number_input("LTM Revenue (₹ Cr)", value=float(fetched.get("revenue", 1000.0)),
                                  min_value=1.0, step=50.0, key="cust_rev")
    ltm_margin = c2.number_input("LTM EBITDA Margin (%)", value=float(fetched.get("margin", 15.0)),
                                  min_value=0.1, max_value=80.0, step=0.5, key="cust_mar")
    _          = c3.text_input("Sector / notes", value="Consumer", key="cust_sec")

    if c1.button("?  What is LTM Revenue?", key="exp_ltm_rev"):
        e = INPUT_EXPLANATIONS["ltm_revenue"]
        explain(e["title"], e["what"], e["why"], e["how"], e["benchmark"])
    if c2.button("?  What is EBITDA Margin?", key="exp_margin"):
        e = INPUT_EXPLANATIONS["ltm_ebitda_margin"]
        explain(e["title"], e["what"], e["why"], e["how"], e["benchmark"])

    st.markdown("### Step 2 — Deal structure")
    d_custom = dict(
        ltm_revenue=ltm_rev, ltm_ebitda_margin=ltm_margin,
        entry_ev_ebitda=10.0, revenue_growth=12.0,
        ltm_ebitda_margin_exit=ltm_margin + 2.0,
        da_pct=4.0, capex_pct=5.0, nwc_pct=1.5, tax_rate=25.17,
        senior_tla_turns=2.0, tla_rate=9.5, tla_amort_years=5,
        senior_tlb_turns=1.0, tlb_rate=10.5, mezz_turns=0.5,
        mezz_rate=14.0, mezz_pik=True, cash_sweep=True, sweep_pct=75.0,
        transaction_fees_pct=2.0, hold_period=5, exit_ev_ebitda=10.0,
    )
    ov = assumptions_panel(d_custom, kp="cust")
    full_p = {**d_custom, **ov, "ltm_revenue": ltm_rev, "ltm_ebitda_margin": ltm_margin}

    st.markdown("### Step 3 — Run")
    if st.button("⚡ Run custom LBO", type="primary", use_container_width=True, key="run_cust"):
        with st.spinner("Computing…"):
            res = run_lbo(full_p)
        if res is None:
            st.error("Over-levered — total debt exceeds enterprise value. "
                     "Reduce leverage turns or raise the entry multiple.")
        else:
            st.session_state["cust_res"] = res
            st.session_state["cust_par"] = full_p
            st.session_state["cust_nm"]  = co_name

    if "cust_res" in st.session_state:
        render_results(st.session_state["cust_res"],
                       st.session_state["cust_par"],
                       st.session_state["cust_nm"])


# ═══════════════════════════════════════════════════════════════════════════
# ── MAIN ─────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style="background:#0c0c0c;border:1px solid #1a1a1a;border-radius:10px;
                padding:18px 22px;margin-bottom:18px">
      <div style="font-size:10px;color:#475569;letter-spacing:.1em;text-transform:uppercase">
        Portfolio Project · Samaksh Sha · Finance & Economics, FLAME University
      </div>
      <div style="font-size:24px;font-weight:700;color:#f1f5f9;margin-top:3px">
        ⚡ LBO Model Engine v2
      </div>
      <div style="font-size:12px;color:#64748b;margin-top:3px">
        Institutional-grade Leveraged Buyout model · Full explanations on every input & output ·
        5 pre-fitted Indian companies + custom deal builder
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("### Select target")
        options = list(COMPANIES.keys()) + ["─────────────", "🔧 Custom LBO"]
        choice  = st.radio("", options,
                           format_func=lambda x: f"{COMPANIES[x]['flag']}  {x}"
                                                  if x in COMPANIES else x,
                           label_visibility="collapsed", key="co_sel")

    if choice in ("─────────────", ""):
        st.info("Select a company from the sidebar to begin.")
        return

    if choice == "🔧 Custom LBO":
        custom_lbo()
        return

    co = COMPANIES[choice]

    st.markdown(f"## {co['flag']}  {choice}")
    col_desc, col_live = st.columns([3, 1])

    with col_desc:
        tags_html = "".join(
            f'<span class="tag" style="background:#1e3a5f;color:#93c5fd">{t}</span>'
            for t in co["tags"]
        )
        st.markdown(
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:8px;padding:12px 15px">'
            f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px">{co["sector"]}</div>'
            f'<div style="font-size:13px;color:#e2e8f0">{co["desc"]}</div>'
            f'<div class="thesis"><b style="color:#93c5fd;font-size:10px">Investment thesis:</b><br>{co["thesis"]}</div>'
            f'{tags_html}</div>', unsafe_allow_html=True,
        )

    with col_live:
        live = fetch_price(co["ticker"])
        price_s  = f"₹{live['price']:,.2f}"     if live.get("price")  else "—"
        mktcap_s = f"₹{live['mktcap']:,.0f} Cr" if live.get("mktcap") else "—"
        st.markdown(
            f'<div style="background:#0f0f0f;border:1px solid #1e1e1e;border-radius:8px;padding:12px 15px">'
            f'<div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.07em">{co["ticker"]} · live</div>'
            f'<div style="font-size:22px;font-weight:700;color:#f1f5f9;margin-top:8px">{price_s}</div>'
            f'<div style="font-size:11px;color:#64748b;margin-top:4px">Mkt cap: {mktcap_s}</div>'
            f'<div style="font-size:11px;color:#475569;margin-top:6px">LTM Rev: ₹{co["ltm_revenue"]:,} Cr<br>EBITDA margin: {co["ltm_ebitda_margin"]}%</div>'
            f'</div>', unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="explain-box" style="margin-top:12px">
    <b>LTM EBITDA</b> (the anchor of the entire model):
    """, unsafe_allow_html=True)
    ltm_ebitda_calc = co["ltm_revenue"] * co["ltm_ebitda_margin"] / 100
    st.markdown(f"""
    <div class="explain-box">
    <b>What is LTM EBITDA?</b> Earnings Before Interest, Taxes, Depreciation & Amortisation —
    the most important single number in M&A and LBO analysis. It is the best proxy for
    operating cash generation, stripped of financing decisions (interest), accounting choices
    (D&A), and tax jurisdictions.<br><br>
    <span class="calc">LTM EBITDA = ₹{co['ltm_revenue']:,} Cr × {co['ltm_ebitda_margin']}% = ₹{ltm_ebitda_calc:,.1f} Cr</span><br><br>
    This single number sets: the <b>purchase price</b> (EBITDA × entry multiple),
    the <b>debt quantum</b> (EBITDA × leverage turns), and the
    <b>exit value</b> (exit EBITDA × exit multiple). Get this wrong and the entire model is wrong.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    ov = assumptions_panel(co, kp=choice.replace(" ","_"))
    full_p = {**co, **ov}

    key_res = f"res_{choice}"; key_par = f"par_{choice}"

    if st.button(f"⚡  Run LBO — {choice}", type="primary",
                 use_container_width=True, key=f"run_{choice}"):
        with st.spinner("Computing LBO model…"):
            res = run_lbo(full_p)
        if res is None:
            st.error("Over-levered — reduce debt turns or raise entry multiple.")
        else:
            st.session_state[key_res] = res
            st.session_state[key_par] = full_p

    if key_res in st.session_state:
        render_results(st.session_state[key_res], st.session_state[key_par], choice)


if __name__ == "__main__":
    main()