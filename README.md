# Portfolio Optimization

Backtest mean-variance optimization, Black-litterman model to compute optimal weights given target and constraints. The Black-litterman strategy implements simple and rolling prediction while backtesting.

These strategies do not include stock selection, which has been done by the ETF manager. Based on the stocks of ETF, we optimize the weights of portfolio to gain smart beta.



Strategies:

* Mean-variance optimization (scipy.optimize)
* Black-litterman model (pyfolio, backtrader)



See notebooks under each folder to see examples.

