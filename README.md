# Brent–WTI Statistical Arbitrage: Rolling Cointegration and Mean Reversion

Code for the paper *Cross-Commodity Statistical Arbitrage in Crude Oil Futures* by Cristian Buccelli.
📄 Paper: *SSRN link coming soon*

## Overview
This is a market-neutral mean-reversion strategy on the Brent–WTI spread, using Yahoo Finance continuous futures `BZ=F` and `CL=F` from 2008 to 2026:
- **Hedge ratio and signal:** a rolling OLS hedge ratio and the z-score of the out-of-sample residual. The strategy enters at |z| ≥ 2 and fills orders at the next open.
- **Costs and risk:** commissions, slippage, roll costs, margin limits and a 30% drawdown circuit breaker.
- **Validation:** cointegration diagnostics, the Probabilistic Sharpe Ratio, Monte Carlo analysis and a 32-fold walk-forward test.

## Key results
| Test | Result |
|---|---|
| Start 2010 → 2026 | +$90,764, Sharpe 0.38, PSR 0.944 (not significant) |
| Start 2008 → 2026 | −$17,355 (circuit breaker hit in 2009) |
| Walk-forward, all folds | −$24,891 |
| Walk-forward, cointegrated training only | +$3,950 |

**Takeaway:** a fixed-parameter strategy has no robust edge. Profits appear only in cointegrated regimes.

## Installation
```bash
git clone https://github.com/Quant-CB/Brent-WTI-Statistical-Arbitrage-Rolling-Cointegration-and-Mean-Reversion.git
cd Brent-WTI-Statistical-Arbitrage-Rolling-Cointegration-and-Mean-Reversion
pip install numpy pandas scipy statsmodels matplotlib yfinance
```

## Disclaimer
This project is for research purposes only and is not investment advice.

## License
MIT
