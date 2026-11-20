"""
IMPORTANT NOTICE:
Before running this script, change the output file path at line 1090
to match your local directory, otherwise the program will try to save
the generated files to the author's original path.
"""


from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps

try:
    from statsmodels.tsa.stattools import adfuller, coint, kpss
    from statsmodels.stats.diagnostic import breaks_cusumolsresid
    HAS_SM = True
except ImportError:                                           # pragma: no cover
    HAS_SM = False

warnings.filterwarnings("ignore")
YEARS = (2000, 2010, 2015, 2018, 2020, 2022, 2024, 2025)


# ============================================================
# CONFIGURATION
# ============================================================
@dataclass(frozen=True)
class Config:
    start: str = "2000-01-01"

    # --- signals ---
    window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.0
    stop_z: float = 4.0
    max_hold: int = 30
    delay: int = 1                 # 0 = execution at the signal close; >=1 = open of s+delay
    beta_min: float = 0.5
    beta_max: float = 1.5
    hedge_method: str = "ols"      # "ols" (rolling) | "kalman"
    kalman_q: float = 1e-6         # state transition variance
    kalman_r: float = 1e-3         # observation variance

    # --- contracts and sizing ---
    capital: float = 100_000.0
    n_brent: int = 2               # used if risk_frac == 0
    max_brent: int = 10
    risk_frac: float = 0.0         # >0: Brent contracts = risk_frac*equity / risk up to the stop
    multiplier: float = 1000.0
    tick_size: float = 0.01

    # --- costs ---
    slippage_ticks: float = 1.0
    dynamic_slippage: bool = True  # slippage * (vol20 / historical median), limited to [0.5, 3]
    commission: float = 2.5
    roll_every: int = 21           # roll cost MODEL (0 = off)

    # --- margins and risk ---
    margin_brent: float = 7000.0
    margin_wti: float = 6500.0
    dynamic_margin: bool = True    # margin * clip(vol_ratio, 0.8, 2.0)
    maint_frac: float = 0.8
    max_leverage: float = 6.0      # notional / equity
    daily_loss_limit: float = 0.03 # closes if the day's loss exceeds 3% of equity
    max_dd_halt: float = 0.30      # stops trading if drawdown > 30% (0 = off)


# ============================================================
# DATA
# ============================================================

def _clean(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()

    # Drop observations without close prices
    d = d.dropna(subset=["bz", "wti"])

    # Drop invalid close prices
    valid_close = (d[["bz", "wti"]] > 0).all(axis=1)
    d = d.loc[valid_close].copy()

    # Do not artificially modify the Opens.
    # Missing or invalid Opens are flagged as NaN.
    for o in ("bz_o", "wti_o"):
        if o in d.columns:
            d[o] = pd.to_numeric(d[o], errors="coerce")
            d.loc[d[o] <= 0, o] = np.nan

    # Flag extreme gaps without correcting them automatically
    for c, o in (("bz", "bz_o"), ("wti", "wti_o")):
        if o in d.columns:
            previous_close = d[c].shift(1)
            gap = d[o] / previous_close - 1
            d[f"{o}_gap_flag"] = gap.abs() > 0.25

    return d


def load_yahoo(start, end=None):
    import yfinance as yf
    df = yf.download(["BZ=F", "CL=F"], start=start, end=end, auto_adjust=True, progress=False)
    d = pd.DataFrame({"bz": df["Close"]["BZ=F"], "wti": df["Close"]["CL=F"],
                      "bz_o": df["Open"]["BZ=F"], "wti_o": df["Open"]["CL=F"]})
    d.index = pd.to_datetime(d.index).tz_localize(None)
    return _clean(d)


def load_fred(start, end=None):
    def get(sid):
        df = pd.read_csv(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")
        s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                      index=pd.to_datetime(df.iloc[:, 0]))
        return s
    d = pd.concat([get("DCOILBRENTEU"), get("DCOILWTICO")], axis=1, keys=["bz", "wti"]).dropna()
    d["bz_o"], d["wti_o"] = d["bz"], d["wti"]                 # close only: execute at the next close
    d = _clean(d).loc[start:end]
    return d


def load_sources(start):
    src = {}
    try:
        src["yahoo"] = load_yahoo(start)
    except Exception as e:                                    # pragma: no cover
        print(f" ! Yahoo not available: {e}")
    try:
        src["fred"] = load_fred(start)
    except Exception as e:                                    # pragma: no cover
        print(f" ! FRED not available (pre-2007 spot excluded): {e}")
    if not src:
        raise RuntimeError("no data source available")
    return src


def pick_source(sources, year):
    """First source that covers the requested year; otherwise the one that starts earliest."""
    for name, d in sources.items():
        if d.index[0] <= pd.Timestamp(year, 3, 1):
            return name
    return min(sources, key=lambda k: sources[k].index[0])


def to_px(d: pd.DataFrame):
    return {k: d[k].to_numpy(float) for k in ("bz", "wti", "bz_o", "wti_o")}


# ============================================================
# SIGNALS
# ============================================================
def _kalman(y, x, q, r):
    """State theta=[beta, alpha] random walk; y_t = beta*x_t + alpha + v. Returns the filtered beta
    and the innovation (prediction residual, computed BEFORE updating with y_t)."""
    n = len(y)
    xc = x - x[0]
    th = np.array([1.0, y[0]])
    P, Q = np.eye(2) * 0.1, np.eye(2) * q
    beta, e = np.empty(n), np.empty(n)
    for t in range(n):
        H = np.array([xc[t], 1.0])
        P = P + Q
        e[t] = y[t] - H @ th
        PH = P @ H
        S = H @ PH + r
        K = PH / S
        th = th + K * e[t]
        P = P - np.outer(K, PH)
        beta[t] = th[0]
    return beta, e


def _zscore(spread: pd.Series, w):
    mu = spread.rolling(w).mean().shift(1)
    sd = spread.rolling(w).std().shift(1)
    return ((spread - mu) / sd).to_numpy(), sd.to_numpy()


def make_signals(pb, pw, cfg):
    """beta known at t (for the number of contracts), out-of-sample residual, z-score on past data only."""
    y, x = pd.Series(np.log(pb)), pd.Series(np.log(pw))
    w = cfg.window
    if cfg.hedge_method == "kalman":
        b, e = _kalman(y.to_numpy(), x.to_numpy(), cfg.kalman_q, cfg.kalman_r)
        beta = pd.Series(b).clip(cfg.beta_min, cfg.beta_max)
        spread = pd.Series(e)
        spread.iloc[:w] = np.nan
    else:
        beta = (y.rolling(w).cov(x) / x.rolling(w).var()).clip(cfg.beta_min, cfg.beta_max)
        alpha = y.rolling(w).mean() - beta * x.rolling(w).mean()
        spread = y - (beta.shift(1) * x + alpha.shift(1))     # parameters known at t-1
    z, sd = _zscore(spread, w)
    return dict(beta=beta.to_numpy(), spread=spread.to_numpy(), z=z, sd=sd)


def desired_position(z, entry, exit_, stop, max_hold):
    """+1 = long spread (long Brent / short WTI), -1 = short spread."""
    n = len(z)
    pos = np.zeros(n, dtype=np.int8)
    held = 0
    for i in range(n):
        zi = z[i]
        if np.isnan(zi):
            continue
        prev = pos[i - 1] if i > 0 else 0
        if prev == 0:
            held = 0
            if zi <= -entry:
                pos[i] = 1
            elif zi >= entry:
                pos[i] = -1
        else:
            held += 1
            out = (zi >= -exit_ or zi <= -stop) if prev == 1 else (zi <= exit_ or zi >= stop)
            pos[i] = 0 if (out or held >= max_hold) else prev
    return pos


def vol_ratio(pb):
    """20-day volatility / expanding historical median (past only), in [0.5, 3]."""
    v = pd.Series(np.log(pb)).diff().rolling(20).std()
    ref = v.expanding(min_periods=60).median()
    return (v / ref).clip(0.5, 3.0).fillna(1.0).to_numpy()


# ============================================================
# BACKTEST
# ============================================================
def side_cost(n_contracts, cfg, slip_ticks):
    return n_contracts * (slip_ticks * cfg.tick_size * cfg.multiplier + cfg.commission)


def margin_required(nB, nW, vr, cfg):
    k = float(np.clip(vr, 0.8, 2.0)) if cfg.dynamic_margin else 1.0
    return k * (nB * cfg.margin_brent + nW * cfg.margin_wti)


def run_backtest(px, cfg, signals=None, eval_from=0):
    """eval_from: index before which nothing is opened (warm-up for the signals)."""
    pb, pw, ob, ow = px["bz"], px["wti"], px["bz_o"], px["wti_o"]
    sig = signals if signals is not None else make_signals(pb, pw, cfg)
    beta, z, sd = sig["beta"], sig["z"], sig["sd"]
    n, M, d = len(pb), cfg.multiplier, cfg.delay

    z_eff = z.copy()
    z_eff[:eval_from] = np.nan
    want = desired_position(z_eff, cfg.entry_z, cfg.exit_z, cfg.stop_z, cfg.max_hold)
    vr = vol_ratio(pb)

    gross, costs = np.zeros(n), np.zeros(n)
    margin_used, notional = np.zeros(n), np.zeros(n)
    trades = []
    equity = peak = cfg.capital
    cur = nB = nW = since = 0
    t_entry, t_gross, t_cost = 0, 0.0, 0.0
    locked = halted = False
    forced = risk_exits = 0

    def record(i, forced_flag):
        trades.append(dict(entry=t_entry, exit=i, side=cur, n_brent=nB, n_wti=nW, gross=t_gross,
                           cost=t_cost, net=t_gross - t_cost, days=i - t_entry, forced=forced_flag))

    for i in range(1, n):
        s = max(i - d, 0)
        sig_t = int(want[i - d]) if i - d >= 0 else 0
        slip = cfg.slippage_ticks * (vr[s] if cfg.dynamic_slippage else 1.0)
        eb, ew = (ob[i], ow[i]) if d >= 1 else (pb[i], pw[i])

        if locked and sig_t == 0:
            locked = False
        target = 0 if (locked or halted) else sig_t
        day_g = day_c = 0.0
        closed = False

        if cur != 0:
            exiting = target != cur
            rb, rw = (eb, ew) if exiting else (pb[i], pw[i])             # exit at the open, otherwise at the close
            g = cur * M * (nB * (rb - pb[i - 1]) - nW * (rw - pw[i - 1]))
            day_g += g
            t_gross += g
            forced_flag = False
            if not exiting:
                since += 1
                if cfg.roll_every and since % cfg.roll_every == 0:         # roll cost (model)
                    c = 2 * side_cost(nB + nW, cfg, slip)
                    day_c += c
                    t_cost += c
                if equity + day_g - day_c < cfg.maint_frac * margin_required(nB, nW, vr[i], cfg):
                    exiting, forced_flag = True, True
                    forced += 1
                elif day_g < -cfg.daily_loss_limit * equity:
                    exiting, forced_flag = True, True
                    risk_exits += 1
                if forced_flag:
                    locked = True
            if exiting:
                c = side_cost(nB + nW, cfg, slip)
                day_c += c
                t_cost += c
                record(i, forced_flag)
                cur, closed = 0, True

        if cur == 0 and not closed and target != 0 and not np.isnan(beta[s]):
            if cfg.risk_frac > 0 and not np.isnan(sd[s]):
                stop_eff = min(cfg.stop_z, cfg.entry_z + 2.0)
                unit_risk = pb[s] * M * (stop_eff - cfg.entry_z) * sd[s]    # $ per pair up to the stop
                nB_new = int(np.clip(np.floor(cfg.risk_frac * equity / max(unit_risk, 1e-9)), 1, cfg.max_brent))
            else:
                nB_new = cfg.n_brent
            hedge = lambda nb: max(1, int(round(beta[s] * nb * pb[s] / pw[s])))   # info at close(s)
            while nB_new > 1 and (nB_new * eb + hedge(nB_new) * ew) * M > cfg.max_leverage * equity:
                nB_new -= 1
            nW_new = hedge(nB_new)
            avail = equity + day_g - day_c
            if avail >= margin_required(nB_new, nW_new, vr[s], cfg) and \
               (nB_new * eb + nW_new * ew) * M <= cfg.max_leverage * avail:
                cur, nB, nW, since = target, nB_new, nW_new, 0
                c = side_cost(nB + nW, cfg, slip)
                g0 = cur * M * (nB * (pb[i] - eb) - nW * (pw[i] - ew))      # from execution to the close
                day_g += g0
                day_c += c
                t_entry, t_gross, t_cost = i, g0, c
            else:
                locked = True

        gross[i], costs[i] = day_g, day_c
        equity += day_g - day_c
        peak = max(peak, equity)
        if cfg.max_dd_halt and equity < peak * (1 - cfg.max_dd_halt):
            halted = True
        if cur != 0:
            margin_used[i] = margin_required(nB, nW, vr[i], cfg)
            notional[i] = (nB * pb[i] + nW * pw[i]) * M

    if cur != 0:                                                   # final liquidation
        c = side_cost(nB + nW, cfg, cfg.slippage_ticks)
        costs[-1] += c
        t_cost += c
        record(n - 1, False)

    return dict(gross=gross, costs=costs, net=gross - costs, trades=trades, margin_used=margin_used,
                notional=notional, forced=forced, risk_exits=risk_exits)


# ============================================================
# METRICS
# ============================================================

def perf_metrics(net, capital):
    net = np.asarray(net, dtype=float)
    n = len(net)
    nan = np.nan

    if n < 2:
        return dict(
            net_pnl=0.0,
            return_pct=0.0,
            ann_return_pct=nan,
            ann_vol_pct=nan,
            sharpe=nan,
            sortino=nan,
            calmar=nan,
            psr=nan,
            t_stat=nan,
            maxdd=0.0,
            maxdd_pct=0.0
        )

    # Daily equity
    eq = capital + np.cumsum(net)

    # Previous equity, including the initial capital
    previous_eq = np.concatenate(([capital], eq[:-1]))

    # Actual daily returns
    r = np.divide(
        net,
        previous_eq,
        out=np.full_like(net, np.nan),
        where=previous_eq > 0
    )

    r = r[np.isfinite(r)]

    if len(r) < 2:
        raise ValueError("Invalid equity: cannot compute returns.")

    # Total return
    total_return = (eq[-1] / capital) - 1

    # CAGR: compound annualized return
    years = n / 252
    cagr = (eq[-1] / capital) ** (1 / years) - 1 \
        if eq[-1] > 0 else -1.0

    # Volatility and Sharpe
    sd = r.std(ddof=1)
    mean_r = r.mean()

    sharpe = (
        np.sqrt(252) * mean_r / sd
        if sd > 0 else nan
    )

    # Sortino
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))

    sortino = (
        np.sqrt(252) * mean_r / downside
        if downside > 0 else nan
    )

    # Drawdown
    equity_with_initial = np.concatenate(([capital], eq))
    peaks = np.maximum.accumulate(equity_with_initial)

    drawdowns = equity_with_initial - peaks
    drawdowns_pct = equity_with_initial / peaks - 1

    maxdd = float(drawdowns.min())
    maxdd_pct = float(drawdowns_pct.min() * 100)

    # Probabilistic Sharpe Ratio
    psr = nan

    if sd > 0 and len(r) > 3:
        sr = mean_r / sd

        den = (
            1
            - sps.skew(r) * sr
            + (sps.kurtosis(r, fisher=False) - 1)
            / 4 * sr ** 2
        )

        if den > 0:
            psr = float(
                sps.norm.cdf(
                    sr * np.sqrt(len(r) - 1) / np.sqrt(den)
                )
            )

    # Descriptive t-stat
    t_stat = (
        mean_r / (sd / np.sqrt(len(r)))
        if sd > 0 else nan
    )

    return dict(
        net_pnl=float(net.sum()),
        return_pct=float(total_return * 100),
        ann_return_pct=float(cagr * 100),
        ann_vol_pct=float(sd * np.sqrt(252) * 100),
        sharpe=float(sharpe),
        sortino=float(sortino),
        calmar=float(cagr * 100 / abs(maxdd_pct))
            if maxdd_pct < 0 else nan,
        psr=psr,
        t_stat=float(t_stat),
        maxdd=maxdd,
        maxdd_pct=maxdd_pct
    )


def stats(res, cfg, first=0):
    first = max(first, 0)
    m = perf_metrics(res["net"][first:], cfg.capital)
    tn = np.array([t["net"] for t in res["trades"] if t["entry"] >= first])
    td = [t["days"] for t in res["trades"] if t["entry"] >= first]
    wins, losses = tn[tn > 0].sum(), -tn[tn < 0].sum()
    m.update(
        gross_pnl=float(res["gross"][first:].sum()), costs=float(res["costs"][first:].sum()),
        n_trades=len(tn), win_rate=float((tn > 0).mean() * 100) if len(tn) else np.nan,
        profit_factor=float(wins / losses) if losses > 0 else np.nan,
        avg_trade_pnl=float(tn.mean()) if len(tn) else np.nan,
        avg_days=float(np.mean(td)) if td else np.nan,
        time_in_mkt_pct=float((res["notional"][first:] > 0).mean() * 100),
        max_margin_pct=float(res["margin_used"][first:].max() / cfg.capital * 100),
        max_lev=float(res["notional"][first:].max() / cfg.capital),
        margin_call=res["forced"], daily_loss_stops=res["risk_exits"])
    return m


# ============================================================
# STATISTICAL DIAGNOSTICS
# ============================================================
def diagnostics(pb, pw):
    """Cointegration relationship and mean reversion over the given period (log-prices)."""
    y, x = np.log(pb), np.log(pw)
    out = {}
    if len(y) < 80:
        return out
    b, a = np.polyfit(x, y, 1)
    s = y - (b * x + a)
    phi = np.polyfit(s[:-1], s[1:], 1)[0]
    out.update(beta_ols=b, phi=phi, half_life=-np.log(2) / np.log(phi) if 0 < phi < 1 else np.inf,
               sd_spread_pct=s.std() * 100)
    d = np.diff(np.log(pb / pw))
    mad = np.median(np.abs(d - np.median(d))) * 1.4826
    out["outlier_spread"] = int((np.abs(d - np.median(d)) > 6 * mad).sum()) if mad > 0 else 0
    if HAS_SM:
        try:
            out["adf_p_bz"] = adfuller(y, autolag="AIC")[1]
            out["adf_p_wti"] = adfuller(x, autolag="AIC")[1]
            out["adf_p_spread"] = adfuller(s, autolag="AIC")[1]       # indicative ADF critical values
            out["coint_p"] = coint(y, x, trend="c", autolag="aic")[1]  # Engle-Granger (corrected values)
            out["kpss_p"] = kpss(s, regression="c", nlags="auto")[1]   # H0 = stationary
            out["cusum_p"] = breaks_cusumolsresid(s - s.mean(), ddof=2)[1]  # H0 = stable relationship
        except Exception:                                               # pragma: no cover
            pass
    return out


def run_period(prices, cfg, start, end=None):
    """Runs the strategy on [start, end) with signal warm-up before start."""
    idx = prices.index
    i0 = idx.searchsorted(pd.Timestamp(start))
    i1 = idx.searchsorted(pd.Timestamp(end)) if end is not None else len(idx)
    if i1 - i0 < cfg.window + 30:
        return None
    w0 = max(0, i0 - cfg.window - 5)
    sub = prices.iloc[w0:i1]
    px = to_px(sub)
    sig = make_signals(px["bz"], px["wti"], cfg)
    ef = i0 - w0
    res = run_backtest(px, cfg, sig, ef)
    valid = np.where(~np.isnan(sig["z"]))[0]
    first = max(ef, int(valid[0])) if len(valid) else ef
    return sub, px, sig, res, first


def period_report(prices, cfg, start, end, source):
    out = run_period(prices, cfg, start, end)
    if out is None:
        return dict(source=source, note="insufficient data"), None
    sub, px, sig, res, first = out
    st = stats(res, cfg, first)
    dg = diagnostics(px["bz"][first:], px["wti"][first:])
    b = sig["beta"][first:]
    rep = dict(source=source, from_date=str(sub.index[first].date()), to_date=str(sub.index[-1].date()),
               days=len(sub) - first)
    rep.update(st)
    rep.update(dg)
    rep["beta_cv_pct"] = float(np.nanstd(b) / np.nanmean(b) * 100) if len(b) else np.nan
    return rep, out


def years_tables(sources, cfg, years):
    rows_from, rows_cal = {}, {}
    for y in years:
        src = pick_source(sources, y)
        rows_from[y], _ = period_report(sources[src], cfg, f"{y}-01-01", None, src)
        rows_cal[y], _ = period_report(sources[src], cfg, f"{y}-01-01", f"{y + 1}-01-01", src)
    return pd.DataFrame(rows_from).T, pd.DataFrame(rows_cal).T


# ============================================================
# COST AND PARAMETER ANALYSIS
# ============================================================
def _first(sig, window):
    v = np.where(~np.isnan(sig["z"]))[0]
    return int(v[0]) if len(v) else window


def cost_scenarios(px, cfg, sig, first):
    scen = {
        "Ideal (no costs, delay 0)": replace(cfg, slippage_ticks=0, commission=0, delay=0, roll_every=0),
        "Base": cfg,
        "Conservative (slip 2, comm 5)": replace(cfg, slippage_ticks=2, commission=5),
        "Stress (slip 5, comm 10, delay 2)": replace(cfg, slippage_ticks=5, commission=10, delay=2),
    }
    return pd.DataFrame({k: stats(run_backtest(px, c, sig), c, first) for k, c in scen.items()}).T


def cost_grid(px, cfg, sig, first, slips, comms):
    pnl, sh = np.zeros((len(comms), len(slips))), np.zeros((len(comms), len(slips)))
    for i, cm in enumerate(comms):
        for j, sl in enumerate(slips):
            c = replace(cfg, slippage_ticks=sl, commission=cm)
            s = stats(run_backtest(px, c, sig), c, first)
            pnl[i, j], sh[i, j] = s["net_pnl"], s["sharpe"]
    return pnl, sh


def breakeven_slippage(px, cfg, sig, first, max_ticks=30):
    ticks = np.arange(0, max_ticks + 1)
    pnl = np.array([stats(run_backtest(px, c, sig), c, first)["net_pnl"]
                    for c in (replace(cfg, slippage_ticks=t) for t in ticks)])
    be = np.nan
    for k in range(1, len(ticks)):
        if pnl[k - 1] > 0 >= pnl[k]:
            be = ticks[k - 1] + pnl[k - 1] / (pnl[k - 1] - pnl[k])
            break
    return ticks, pnl, be


def param_grid(px, cfg, entries, windows):
    pnl = np.zeros((len(entries), len(windows)))
    for j, w in enumerate(windows):
        cw = replace(cfg, window=w)
        sig = make_signals(px["bz"], px["wti"], cw)
        f = _first(sig, w)
        for i, e in enumerate(entries):
            c = replace(cw, entry_z=e)
            pnl[i, j] = stats(run_backtest(px, c, sig), c, f)["net_pnl"]
    return pnl


def stress_test(cfg, pb, pw, beta, nB=None):
    """Instantaneous P&L of a long Brent / short WTI pair for simultaneous shocks on the two legs."""
    nB = nB or cfg.n_brent
    nW = max(1, int(round(beta * nB * pb / pw)))
    shocks = [(-0.10, 0.10), (-0.05, 0.05), (0.05, -0.05), (0.10, -0.10), (-0.15, -0.15), (0.15, 0.15)]
    rows = {f"Brent {a:+.0%} / WTI {b:+.0%}": cfg.multiplier * (nB * pb * a - nW * pw * b) for a, b in shocks}
    return pd.Series(rows, name=f"P&L $ long spread ({nB} BZ, {nW} CL)")


# ============================================================
# MONTE CARLO A: BLOCK BOOTSTRAP
# ============================================================
def block_bootstrap(net_active, n_sims, block, rng):
    n = len(net_active)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_sims, k))
    idx = ((starts[:, :, None] + np.arange(block)) % n).reshape(n_sims, -1)[:, :n]
    return net_active[idx]


def bootstrap_stats(paths, capital):
    cum = np.cumsum(paths, axis=1)
    eq = capital + cum
    maxdd = (eq - np.maximum.accumulate(eq, axis=1)).min(axis=1)
    r = paths / capital
    sd = r.std(axis=1)
    sharpe = np.where(sd > 0, np.sqrt(252) * r.mean(axis=1) / np.where(sd > 0, sd, 1), np.nan)
    return cum, cum[:, -1], maxdd, sharpe


# ============================================================
# MONTE CARLO B: SYNTHETIC PRICES (calibration on training, evaluation afterwards)
# ============================================================

def fit_spread_model(pb, pw):
    y = np.log(pb)
    x = np.log(pw)

    b, a = np.polyfit(x, y, 1)
    spread = y - (b * x + a)

    phi, c = np.polyfit(
        spread[:-1],
        spread[1:],
        1
    )

    eps = spread[1:] - (
        phi * spread[:-1] + c
    )

    dx = np.diff(x)

    # Center both innovations
    dx = dx - dx.mean()
    eps = eps - eps.mean()

    # Joint resampling:
    # preserves the contemporaneous dependence
    pairs = np.column_stack([dx, eps])

    return dict(
        b=b,
        a=a,
        phi=phi,
        c=c,
        pairs=pairs,
        x0=x[-1],
        s0=spread[-1],
        s_mean=spread.mean()
    )



def simulate_paths(model, n, n_sims, phi_s, rng):
    pr = model["pairs"][rng.integers(0, len(model["pairs"]), size=(n_sims, n))]
    x = model["x0"] + np.cumsum(pr[:, :, 0], axis=1)
    c_s = model["s_mean"] * (1 - phi_s)
    s = np.empty((n_sims, n))
    prev = np.full(n_sims, model["s0"])
    for t in range(n):
        prev = c_s + phi_s * prev + pr[:, t, 1]
        s[:, t] = prev
    return np.exp(model["b"] * x + model["a"] + s), np.exp(x)


def mc_holdout(prices, cfg, train_frac, n_sims, rng):
    n = len(prices)
    cut = int(n * train_frac)
    tr = prices.iloc[:cut]
    model = fit_spread_model(tr["bz"].to_numpy(), tr["wti"].to_numpy())
    rep, art = period_report(prices, cfg, prices.index[cut], None, "holdout")
    actual = rep
    warm = cfg.window + 5
    phi_fit = float(np.clip(model["phi"], 0.0, 0.9999))
    scen = {"Fitted OU (train)": phi_fit, "Slower OU (HL x2)": phi_fit ** 0.5,
            "Random walk (null hypothesis)": 1.0}
    out = {}
    for name, phi_s in scen.items():
        PB, PW = simulate_paths(model, n - cut + warm, n_sims, phi_s, rng)
        fin, dd, sh, nt = [], [], [], []
        for k in range(n_sims):
            px = dict(bz=PB[k], wti=PW[k], bz_o=PB[k], wti_o=PW[k])
            sig = make_signals(PB[k], PW[k], cfg)
            st = stats(run_backtest(px, cfg, sig, warm), cfg, warm)
            fin.append(st["net_pnl"]); dd.append(st["maxdd"]); sh.append(st["sharpe"]); nt.append(st["n_trades"])
        out[name] = dict(final=np.array(fin), maxdd=np.array(dd), sharpe=np.array(sh),
                         n_trades=np.array(nt), phi=phi_s)
    return out, actual, model, cut


# ============================================================
# WALK-FORWARD
# ============================================================
def walk_forward(prices, cfg, train=756, test=126, entries=(1.5, 2.0, 2.5), windows=(30, 60, 90),
                 min_trades=3, coint_gate=False):
    n = len(prices)
    oos = np.zeros(n)
    folds = []
    start = train
    while n - start >= 40:
        tl = min(test, n - start)
        tr0 = start - train
        best, best_sc = None, -np.inf
        for w in windows:
            s0 = max(0, tr0 - w - 5)
            sub = prices.iloc[s0:start]
            px = to_px(sub)
            cw = replace(cfg, window=w)
            sig = make_signals(px["bz"], px["wti"], cw)
            ef = tr0 - s0
            f = max(ef, _first(sig, w))
            for e in entries:
                c = replace(cw, entry_z=e)
                st = stats(run_backtest(px, c, sig, ef), c, f)
                sc = st["sharpe"] if st["n_trades"] >= min_trades and not np.isnan(st["sharpe"]) else -np.inf
                if sc > best_sc:
                    best, best_sc, best_tr = (w, e), sc, st
        if best is None:
            best, best_tr = (cfg.window, cfg.entry_z), dict(sharpe=np.nan)
        w, e = best
        tr_prices = prices.iloc[tr0:start]
        cp = np.nan
        if HAS_SM:
            try:
                cp = coint(np.log(tr_prices["bz"]), np.log(tr_prices["wti"]), trend="c")[1]
            except Exception:
                pass
        c = replace(cfg, window=w, entry_z=e)
        pnl_t, n_tr = 0.0, 0
        if not (coint_gate and cp > 0.10):
            s0 = max(0, start - w - 5)
            sub = prices.iloc[s0:start + tl]
            px = to_px(sub)
            ef = start - s0
            res = run_backtest(px, c, make_signals(px["bz"], px["wti"], c), ef)
            oos[start:start + tl] = res["net"][ef:]
            pnl_t, n_tr = res["net"][ef:].sum(), sum(t["entry"] >= ef for t in res["trades"])
        folds.append(dict(test_start=prices.index[start], test_end=prices.index[start + tl - 1], window=w,
                          entry_z=e, sharpe_train=best_tr["sharpe"], coint_p_train=cp,
                          pnl_test=pnl_t, n_trades_test=n_tr))
        start += tl
    first_oos = train
    return oos, first_oos, pd.DataFrame(folds)


# ============================================================
# CHARTS
# ============================================================
GREEN, RED, PURPLE, GREY = "#2e8b57", "#c0392b", "#6a3d9a", "#7f8c8d"


def _style(ax, title, ylabel=None):
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def heatmap(ax, data, xlabels, ylabels, title, xlabel, ylabel, fmt="{:,.0f}"):
    vmax = np.nanmax(np.abs(data)) or 1
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(xlabels)), [str(v) for v in xlabels])
    ax.set_yticks(range(len(ylabels)), [str(v) for v in ylabels])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title(title, fontsize=10, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    return im


def plot_backtest(sub, sig, res, cfg, first):
    dates = sub.index
    z = sig["z"]
    tr = pd.DataFrame(res["trades"])
    fig, ax = plt.subplots(5, 1, figsize=(13, 14), sharex=True,
                           gridspec_kw={"height_ratios": [1.3, 1, 1, 1.2, 0.8]})
    a = ax[0]
    a.plot(dates, sub["bz"], color="#1f77b4", lw=1, label="Brent")
    a.plot(dates, sub["wti"], color="#e67e22", lw=1, label="WTI")
    _style(a, "Prices (continuous futures / spot)", "USD/bbl"); a.legend(ncol=2, fontsize=8)
    a = ax[1]
    a.plot(dates, z, color=RED, lw=0.9, label="z-score (out-of-sample residual)")
    for lvl in (cfg.entry_z, -cfg.entry_z):
        a.axhline(lvl, color=GREEN if lvl > 0 else "orange", ls="--", lw=1)
    a.axhline(0, color="black", lw=0.7)
    if len(tr):
        le, se = tr[tr.side == 1], tr[tr.side == -1]
        a.scatter(dates[le.entry], np.nan_to_num(z[le.entry]), marker="^", s=45, color=GREEN, edgecolor="k", zorder=5, label="Long spread")
        a.scatter(dates[se.entry], np.nan_to_num(z[se.entry]), marker="v", s=45, color=RED, edgecolor="k", zorder=5, label="Short spread")
        a.scatter(dates[tr.exit], np.nan_to_num(z[tr.exit]), marker="o", s=14, color="k", zorder=5, label="Exit")
    _style(a, "Z-score and trades (execution at the open of s+delay)"); a.legend(ncol=4, fontsize=8, loc="upper left")
    a = ax[2]
    a.plot(dates, sig["beta"], color=PURPLE, lw=1.2, label="hedge ratio β")
    a.axhline(np.nanmean(sig["beta"][first:]), color=GREY, ls=":", lw=1)
    _style(a, f"Hedge ratio ({cfg.hedge_method})", "β"); a.legend(fontsize=8)
    a = ax[3]
    a.plot(dates, np.cumsum(res["gross"]), color=GREY, ls="--", label="Gross P&L")
    a.plot(dates, np.cumsum(res["net"]), color=PURPLE, lw=1.6, label="Net P&L")
    a.axhline(0, color="k", lw=0.7)
    _style(a, "Cumulative P&L", "USD"); a.legend(loc="upper left", fontsize=8)
    a = ax[4]
    eq = cfg.capital + np.cumsum(res["net"])
    dd = (eq / np.maximum.accumulate(eq) - 1) * 100
    a.fill_between(dates, dd, 0, color=RED, alpha=0.35)
    _style(a, "Equity drawdown", "%")
    fig.tight_layout()
    return fig


def plot_years(tf, tc):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    yrs = [str(y) for y in tf.index]
    num = lambda df, c: pd.to_numeric(df[c], errors="coerce").to_numpy(float) if c in df else np.full(len(df), np.nan)

    def bars(a, vals, title, ylabel, fmt="{:,.0f}", color_sign=True):
        v = np.nan_to_num(vals)
        a.bar(yrs, v, color=[GREEN if x >= 0 else RED for x in v] if color_sign else PURPLE)
        for i, x in enumerate(vals):
            if not np.isnan(x):
                a.text(i, x, fmt.format(x), ha="center", va="bottom" if x >= 0 else "top", fontsize=8)
        a.axhline(0, color="k", lw=0.7)
        _style(a, title, ylabel)

    bars(ax[0, 0], num(tf, "net_pnl"), "Net P&L: from the given year to today", "USD")
    bars(ax[0, 1], num(tc, "net_pnl"), "Net P&L: calendar year only", "USD")
    x = np.arange(len(yrs)); w = 0.38
    a = ax[0, 2]
    a.bar(x - w / 2, np.nan_to_num(num(tf, "sharpe")), w, color=PURPLE, label="year to today")
    a.bar(x + w / 2, np.nan_to_num(num(tc, "sharpe")), w, color="#b39ddb", label="calendar year")
    a.set_xticks(x, yrs); a.axhline(0, color="k", lw=0.7); a.legend(fontsize=8)
    _style(a, "Annualized Sharpe", "Sharpe")
    a = ax[1, 0]
    for c, col, lab in (("coint_p", PURPLE, "Engle-Granger p"), ("adf_p_spread", "#e67e22", "ADF spread p"),
                        ("kpss_p", GREEN, "KPSS p (H0: stationary)")):
        a.plot(yrs, num(tf, c), marker="o", color=col, label=lab)
    a.axhline(0.05, color=RED, ls="--", lw=1)
    a.set_ylim(-0.02, 1.02); a.legend(fontsize=8)
    _style(a, "Evidence of mean reversion (5% threshold dashed)", "p-value")
    hl = np.minimum(num(tf, "half_life"), 250)
    bars(ax[1, 1], hl, "Spread half-life (capped at 250 days)", "days", "{:.0f}", color_sign=False)
    a = ax[1, 2]
    a.plot(yrs, num(tf, "beta_ols"), marker="o", color=PURPLE, label="β OLS (year to today)")
    a.plot(yrs, num(tc, "beta_ols"), marker="s", color="#b39ddb", label="β OLS (calendar year)")
    a.legend(fontsize=8)
    _style(a, "Hedge ratio stability", "β")
    fig.suptitle("Quantitative model data by starting year", fontweight="bold")
    fig.tight_layout()
    return fig


def plot_sensitivity(slips, comms, pnl_grid, sh_grid, ticks, be_line, be, entries, windows, p_grid):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    heatmap(ax[0, 0], pnl_grid, slips, comms, "Net P&L ($): slippage × commissions",
            "Slippage (ticks/leg/side)", "Commissions ($/contract/side)")
    heatmap(ax[0, 1], sh_grid, slips, comms, "Sharpe: slippage × commissions",
            "Slippage (ticks/leg/side)", "Commissions ($/contract/side)", fmt="{:.2f}")
    a = ax[1, 0]
    a.plot(ticks, be_line, color=PURPLE, marker="o", ms=3)
    a.axhline(0, color="k", lw=0.8)
    if not np.isnan(be):
        a.axvline(be, color=RED, ls="--", label=f"Break-even ≈ {be:.1f} ticks"); a.legend()
    a.set_xlabel("Slippage (ticks/leg/side)")
    _style(a, "How much slippage can the strategy withstand?", "Net P&L ($)")
    heatmap(ax[1, 1], p_grid, windows, entries, "Net P&L ($): entry z × window",
            "Rolling window (days)", "Entry z (σ)")
    fig.tight_layout()
    return fig


def plot_montecarlo(cum, fin, actual_path, actual_final, mc, actual_ho, rng):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    for k in rng.choice(len(cum), size=min(80, len(cum)), replace=False):
        a.plot(cum[k], color=GREY, alpha=0.15, lw=0.7)
    p5, p50, p95 = np.percentile(cum, [5, 50, 95], axis=0)
    a.fill_between(range(cum.shape[1]), p5, p95, color=PURPLE, alpha=0.18, label="5%-95%")
    a.plot(p50, color=PURPLE, lw=2, label="Median")
    a.plot(actual_path, color="k", lw=2, label="Actual path"); a.axhline(0, color="k", lw=0.7)
    a.set_xlabel("Trading days"); a.legend(fontsize=8)
    _style(a, "MC A: block bootstrap of daily P&L", "Cumulative P&L ($)")
    a = ax[0, 1]
    a.hist(fin, bins=40, color=PURPLE, alpha=0.7)
    a.axvline(0, color="k"); a.axvline(actual_final, color=RED, ls="--", label=f"Actual {actual_final:,.0f}$")
    a.axvline(np.percentile(fin, 5), color="orange", ls="--", label=f"5th pct {np.percentile(fin, 5):,.0f}$")
    a.legend(fontsize=8)
    _style(a, f"MC A: final P&L  |  P(>0) = {(fin > 0).mean():.0%}")
    cols = [GREEN, "orange", RED]
    for (name, d), c in zip(mc.items(), cols):
        ax[1, 0].hist(d["final"], bins=40, alpha=0.45, color=c, label=f"{name} (φ={d['phi']:.3f})")
        ax[1, 1].hist(d["maxdd"], bins=40, alpha=0.45, color=c, label=name)
    ax[1, 0].axvline(actual_ho["net_pnl"], color="k", lw=2, ls="--", label="Actual (holdout)")
    ax[1, 0].axvline(0, color=GREY, lw=0.8); ax[1, 0].legend(fontsize=8)
    _style(ax[1, 0], "MC B: P&L on synthetic prices (calibrated on training)")
    ax[1, 1].axvline(actual_ho["maxdd"], color="k", lw=2, ls="--", label="Actual (holdout)")
    ax[1, 1].legend(fontsize=8)
    _style(ax[1, 1], "MC B: max drawdown ($)")
    fig.tight_layout()
    return fig


def plot_walkforward(dates, oos, first_oos, folds, base_net, cfg):
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    a.plot(dates[first_oos:], np.cumsum(oos[first_oos:]), color=PURPLE, lw=1.8, label="Walk-forward OOS")
    a.plot(dates[first_oos:], np.cumsum(base_net[first_oos:]), color=GREY, ls="--", label="Fixed parameters (in-sample)")
    a.axhline(0, color="k", lw=0.7); a.legend(fontsize=8)
    _style(a, "Cumulative P&L: out-of-sample vs fixed parameters", "USD")
    a = ax[0, 1]
    xs = folds["test_start"]
    a.step(xs, folds["window"], where="post", color=PURPLE, label="window")
    a2 = a.twinx()
    a2.step(xs, folds["entry_z"], where="post", color="orange", label="entry z")
    a.set_ylabel("window (days)"); a2.set_ylabel("entry z")
    _style(a, "Parameters chosen on each fold's training set")
    a = ax[1, 0]
    a.scatter(folds["sharpe_train"], folds["pnl_test"], c=np.where(folds["pnl_test"] >= 0, GREEN, RED), edgecolor="k")
    a.axhline(0, color="k", lw=0.7); a.axvline(0, color="k", lw=0.7)
    a.set_xlabel("Training Sharpe")
    _style(a, "Does training predict the test?", "Test P&L ($)")
    a = ax[1, 1]
    a.bar(range(len(folds)), folds["coint_p_train"], color=[GREEN if p < 0.05 else GREY for p in folds["coint_p_train"]])
    a.axhline(0.05, color=RED, ls="--", lw=1)
    a.set_xlabel("fold")
    _style(a, "Engle-Granger p-value on training (green < 5%)", "p-value")
    fig.tight_layout()
    return fig


# ============================================================
# MAIN
# ============================================================
def show(df, cols=None, title=""):
    cols = [c for c in (cols or df.columns) if c in df.columns]
    print("\n" + "=" * 100 + f"\n {title}\n" + "=" * 100)
    print(df[cols].to_string(float_format=lambda v: f"{v:,.2f}", na_rep="n/a"))


PERF_COLS = ["source", "from_date", "to_date", "days", "gross_pnl", "costs", "net_pnl", "return_pct", "ann_return_pct",
             "ann_vol_pct", "sharpe", "sortino", "calmar", "psr", "t_stat", "maxdd", "maxdd_pct", "n_trades",
             "win_rate", "profit_factor", "avg_trade_pnl", "avg_days", "time_in_mkt_pct", "max_lev",
             "max_margin_pct", "margin_call", "daily_loss_stops"]
DIAG_COLS = ["source", "beta_ols", "beta_cv_pct", "phi", "half_life", "sd_spread_pct", "adf_p_bz", "adf_p_wti",
             "adf_p_spread", "coint_p", "kpss_p", "cusum_p", "outlier_spread"]


def main(sources=None, cfg=Config(), years=YEARS, n_sims=2500, seed=42, show_plots=True, save_dir=None,
         train_frac=0.6, wf_train=756, wf_test=126):
    rng = np.random.default_rng(seed)
    sources = sources or load_sources(cfg.start)
    for k, d in sources.items():
        print(f" Source {k:<6}: {d.index[0].date()} -> {d.index[-1].date()}  ({len(d)} days)")
    if "yahoo" not in sources:
        raise RuntimeError("Yahoo futures data not available.")

    prices = sources["yahoo"]

    tf, tc = years_tables(
        {"yahoo": prices},
        cfg,
        years
    )
    show(tf, PERF_COLS, "PERFORMANCE: from the given year to the last data point")
    show(tf, DIAG_COLS, "STATISTICAL DIAGNOSTICS: from the given year to the last data point")
    show(tc, PERF_COLS, "PERFORMANCE: calendar year only")
    show(tc, DIAG_COLS, "STATISTICAL DIAGNOSTICS: calendar year only")
    print("\n Reading guide: coint_p / adf_p_spread < 0.05 = evidence of mean reversion; kpss_p > 0.05 = stationarity"
          "\n is not rejected; cusum_p < 0.05 = unstable relationship; high outlier_spread = possible roll jumps."
          "\n In-sample statistics over the period: the real test is the walk-forward section.")

    # ---------- 2) base backtest, full sample ----------
    sub, px, sig, res, first = run_period(prices, cfg, prices.index[0])
    st = stats(res, cfg, first)
    print("\n" + "=" * 100 + f"\n BASE BACKTEST  ({sub.index[first].date()} -> {sub.index[-1].date()})\n" + "=" * 100)
    for k, v in st.items():
        print(f" {k:<18}: {v:,.2f}" if isinstance(v, (float, np.floating)) else f" {k:<18}: {v}")
    print(stress_test(cfg, px["bz"][-1], px["wti"][-1], sig["beta"][-1]).to_frame().to_string(
        float_format=lambda v: f"{v:,.0f}"))

    # ---------- 3) costs and parameters ----------
    sc = cost_scenarios(px, cfg, sig, first)
    show(sc, ["gross_pnl", "costs", "net_pnl", "return_pct", "sharpe", "maxdd_pct", "n_trades", "win_rate"],
         "COST SCENARIOS")
    slips, comms = [0, 1, 2, 3, 5, 10], [0, 2.5, 5, 10, 20]
    pnl_grid, sh_grid = cost_grid(px, cfg, sig, first, slips, comms)
    ticks, be_line, be = breakeven_slippage(px, cfg, sig, first)
    entries, windows = [1.5, 2.0, 2.5, 3.0], [30, 45, 60, 90, 120]
    p_grid = param_grid(px, cfg, entries, windows)
    print(f"\n Break-even slippage: {'>30' if np.isnan(be) else f'{be:.1f}'} ticks/leg/side "
          f"(1 tick = ${cfg.tick_size * cfg.multiplier:.0f}/contract)")

    # ---------- 4) Monte Carlo A ----------
    active = res["net"][first:]
    paths = block_bootstrap(active, n_sims, 10, rng)
    cum, fin_bs, dd_bs, sh_bs = bootstrap_stats(paths, cfg.capital)
    print("\n" + "=" * 100 + f"\n MONTE CARLO A: block bootstrap ({n_sims} sims, 10-day blocks)\n" + "=" * 100)
    print(f" Final P&L: median {np.median(fin_bs):,.0f}$  5th pct {np.percentile(fin_bs, 5):,.0f}$  "
          f"95th pct {np.percentile(fin_bs, 95):,.0f}$  |  P(>0) {(fin_bs > 0).mean():.1%}  |  "
          f"Median Max DD {np.median(dd_bs):,.0f}$")

    # ---------- 5) Monte Carlo B with holdout ----------
    mc, act_ho, model, cut = mc_holdout(prices, cfg, train_frac, n_sims, rng)
    hl = -np.log(2) / np.log(model["phi"]) if 0 < model["phi"] < 1 else np.inf
    print("\n" + "=" * 100)
    print(f" MONTE CARLO B: calibrated on {prices.index[0].date()}→{prices.index[cut - 1].date()}, "
          f"evaluated on {prices.index[cut].date()}→{prices.index[-1].date()}")
    print("=" * 100)
    print(f" Training: β={model['b']:.3f}  φ(AR1)={model['phi']:.4f}  half-life={hl:.1f} days")
    rows = {n_: dict(mean_pnl=d["final"].mean(), pnl_p5=np.percentile(d["final"], 5),
                     pnl_p95=np.percentile(d["final"], 95), pct_positive=(d["final"] > 0).mean() * 100,
                     mean_maxdd=d["maxdd"].mean(), mean_sharpe=np.nanmean(d["sharpe"]),
                     mean_n_trades=d["n_trades"].mean()) for n_, d in mc.items()}
    print(pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:,.2f}"))
    null = mc["Random walk (null hypothesis)"]["final"]
    pval = ((null >= act_ho["net_pnl"]).sum() + 1) / (len(null) + 1)
    print(f"\n Actual (holdout): P&L {act_ho['net_pnl']:,.0f}$  Sharpe {act_ho['sharpe']:.2f}"
          f"\n p-value vs random walk: {pval:.3f} -> "
          f"{'evidence of OOS edge' if pval < 0.05 else 'NOT distinguishable from chance'} (5% threshold)")

    # ---------- 6) walk-forward ----------
    oos, f_oos, folds = walk_forward(prices, cfg, wf_train, wf_test)
    pm = perf_metrics(oos[f_oos:], cfg.capital)
    print("\n" + "=" * 100 + f"\n WALK-FORWARD (train {wf_train} days, test {wf_test} days, {len(folds)} folds)\n" + "=" * 100)
    print(f" OOS period: {prices.index[f_oos].date()} -> {prices.index[-1].date()}")
    for k in ("net_pnl", "return_pct", "ann_return_pct", "sharpe", "sortino", "psr", "t_stat", "maxdd", "maxdd_pct"):
        print(f" {k:<14}: {pm[k]:,.2f}")
    print(f" Profitable folds: {(folds['pnl_test'] > 0).mean():.0%}  |  "
          f"Folds with cointegration (p<5%) on training: {(folds['coint_p_train'] < 0.05).mean():.0%}")

    # ---------- charts ----------
    figs = {
        "01_backtest": plot_backtest(sub, sig, res, cfg, first),
        "02_years": plot_years(tf, tc),
        "03_sensitivity": plot_sensitivity(slips, comms, pnl_grid, sh_grid, ticks, be_line, be, entries, windows, p_grid),
        "04_montecarlo": plot_montecarlo(cum, fin_bs, np.cumsum(active), st["net_pnl"], mc, act_ho, rng),
        "05_walkforward": plot_walkforward(prices.index, oos, f_oos, folds, run_backtest(to_px(prices), cfg)["net"], cfg),
    }
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(f"{save_dir}/{name}.png", dpi=110)
        tf.to_csv(f"{save_dir}/table_from_year.csv")
        tc.to_csv(f"{save_dir}/table_calendar_year.csv")
        folds.to_csv(f"{save_dir}/walk_forward_folds.csv", index=False)
    if show_plots:
        plt.show()
    return dict(from_year=tf, calendar=tc, stats=st, scenarios=sc, mc=mc, pval=pval, walk=folds, figs=figs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--save-dir", default="CHANGE HERE TO YOUR PATH") # <--- CHANGE THIS TO YOUR PATH !!
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--hedge", choices=["ols", "kalman"], default="ols")
    ap.add_argument("--risk-frac", type=float, default=0.0)
    a = ap.parse_args()
    main(cfg=Config(hedge_method=a.hedge, risk_frac=a.risk_frac), n_sims=a.sims,
         save_dir=a.save_dir, show_plots=not a.no_show)
