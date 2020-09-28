import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import yfinance as yf
import pyfolio as pf
import pypfopt as pyp


def build_data(data_dir):
    """
    Accustomized function to load data
    """
    mcap = pd.read_csv(os.path.join(data_dir, 'mcap.csv'))  # stock capacity
    prices = pd.read_csv(
        os.path.join(data_dir, 'prices.csv'), 
        parse_dates=['date']
    ).set_index('date')  # stock price
    view_confidence = pd.read_csv(os.path.join(data_dir, 'views.csv'), index_col=[0]) # view
    market_prices = yf.download("BSE-500.BO", period="max")["Adj Close"]  # benchmark

    tickers = [ticker[:-3] for ticker in mcap.Tickers]
    mcap = {ticker[:-3] : cap for ticker, cap in zip(
        mcap['Tickers'].values, mcap['Market Cap'].values
    )}
    views_dict = {ind : view_confidence['View'][ind] for ind in view_confidence.index}
    confidences = list(view_confidence.Confidences)
    return prices, market_prices, mcap, views_dict, confidences


def bl_optimize(
    prices,
    market_prices,
    mcap,
    views_dict,
    start_date=None,
    end_date=None,
    years_before_enddate=None,
    omega=None,
    confidences=None,
    weight_bounds=None,
    return_df=False, 
    save_dir=None, 
    save_name=None,
    visualize=False
):
    """
    :param prices: pd.DataFrame, asset prices
    :param market_prices: pd.DataFrame, market prices
    :param mcap: dict, mapping from assets to market capital
    :param views_dict: dict, mapping from assets to invester views
    :param start_date: datetime.datetime, select data since start date 
    :param end_date: datetime.datetime, select data before end date
    :param years_before_enddate: int, choose years when selecting data 
    :param omega: str
    :param confidences: list, confidence
    :param weight_bounds: tuple, lower and upper bounds
    :param return_df: boolean, whether to return returns and weights df
    :param save_dir: str, save data to local directory if provided
    :param visualize: boolean, whether to visualize
    :return: (pd.DataFrame, pd.DataFrame), returns df and weights df
    :raises TypeError: if prices or market_prices are not pandas DataFrame
    :raises ValueError: end date must not be none if years before date is provided
    """
    if not isinstance(prices, pd.DataFrame) or not isinstance(market_prices, pd.Series):
        raise TypeError('`prices` and `market_prices` must be pandas.DataFrame')

    if years_before_enddate and not end_date:
        raise ValueError('`end_date` must not be None if `years_before_enddate` is provided')

    if years_before_enddate:
        start_date_ = end_date - timedelta(days=365 * years_before_enddate)
        start_date = start_date_ if not start_date else max(start_date_, start_date)

    # Select data
    st = (prices.index >= start_date) if start_date else True
    et = (prices.index <= end_date) if end_date else True
    time_range = st & et
    prices = prices[time_range] if time_range is not True \
        else prices

    st = (market_prices.index >= start_date) if start_date else True
    et = (market_prices.index <= end_date) if end_date else True
    time_range = st & et
    market_prices = market_prices[time_range] if time_range is not True \
        else market_prices

    if (prices.index[-1] - prices.index[0]) // timedelta(days=365 * 2) < 1:
         warnings.warn('`prices` should have at least 2 years')
         return None, None
    print('Compute weights from {} to {}'.format(prices.index[0], prices.index[-1]))

    # Expected return and covariance
    mu = pyp.expected_returns.mean_historical_return(prices)
    S = pyp.risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    delta = pyp.black_litterman.market_implied_risk_aversion(
        market_prices, 
        risk_free_rate=0.05
    )

    # Black litterman model
    market_prior = pyp.black_litterman.market_implied_prior_returns(mcap, delta, S)
    bl = pyp.BlackLittermanModel(
        S, 
        pi=market_prior, 
        absolute_views=views_dict,
        omega=omega,
        view_confidences=confidences
    )

    # Posterior estimate of returns and covariances
    bl_return = bl.bl_returns()
    S_bl = bl.bl_cov()

    # Create returns dataframe
    returns_df = pd.DataFrame(
        [market_prior, mu, bl_return, pd.Series(views_dict)], 
        index=['prior', 'historical', 'posterior', 'views']
    ).T

    # Portfolio optimization
    ef = pyp.EfficientFrontier(
        bl_return, 
        S_bl, 
        weight_bounds=weight_bounds if not weight_bounds else (0, 1), 
        gamma=0
    )
    ef.add_objective(pyp.objective_functions.L2_reg, gamma=0.1)
    ef.min_volatility()
    ef.portfolio_performance(verbose=visualize)
    weights = ef.clean_weights()

    # Create weights dataframe
    weights_df = pd.DataFrame(
        [weights],
        index=['weights'],
        columns=weights.keys()
    ).T * 100

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = save_name if save_name else prices.index[-1].strftime('%Y%m%d')
        returns_df.to_csv(
            os.path.join(save_dir, 'returns_{}.csv'.format(save_name))
        )
        S_bl.to_csv(
            os.path.join(save_dir, 'S_bl_{}.csv'.format(save_name))
        )
        weights_df.to_csv(
            os.path.join(save_dir, 'weights_{}.csv'.format(save_name)), 
            header=False
        )

    if visualize:
        market_prior.plot.barh(figsize=(12, 6), title='Market priors', grid=True)
        returns_df.plot.bar(figsize=(14, 6), title='Return estimates', grid=True)
        pyp.plotting.plot_covariance(S_bl)
        weights_df.plot.bar(
            figsize=(14,6), 
            title='Asset allocation', 
            grid=True, 
            legend=False
        )
        plt.ylabel('Percentage')

    if return_df:
        return returns_df, weights_df


def plot_performance(
    returns, 
    benchmark_prices,
    plot_stats=False,
    startcash=None,
    log_returns=False,
    save_dir=None
):
    """
    :param returns: pd.Series, return data
    :param benchmark_prices: pd.Series, benchmark return data
    :param startcash: int, rebase the benchmark if provided
    :return: None
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Rebase benchmark prices, the same as portfolio prices
    if startcash is not None:
        benchmark_prices = (benchmark_prices / benchmark_prices.iloc[0]) * startcash
    benchmark_rets = pyp.expected_returns.returns_from_prices(benchmark_prices)

    if log_returns:
        portfolio_value = returns.cumsum().apply(np.exp) * startcash
    else:
        portfolio_value = (1 + returns).cumprod() * startcash

    # Performance statistics
    if plot_stats:
        pf.show_perf_stats(returns)
        pf.show_perf_stats(benchmark_rets)
    
    # Fig 1: price and return
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=[14, 8])
    portfolio_value.plot(ax=ax[0], label='Portfolio')
    benchmark_prices.plot(ax=ax[0], label='Benchmark')
    ax[0].set_ylabel('Price')
    ax[0].grid(True)
    ax[0].legend()
    
    returns.plot(ax=ax[1], label='Portfolio', alpha=0.5)
    benchmark_rets.plot(ax=ax[1], label='Benchmark', alpha=0.5)
    ax[1].set_ylabel('Return')

    fig.suptitle('Black–Litterman Portfolio Optimization', fontsize=16)
    plt.grid(True)
    plt.legend()
    plt.show()
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'price_and_return'), dpi=300)

    # Fig 2: return performance
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 9), constrained_layout=False)
    axes = ax.flatten()
    pf.plot_rolling_beta(returns=returns, factor_returns=benchmark_rets, ax=axes[0])
    pf.plot_return_quantiles(returns=returns, ax=axes[1])
    pf.plot_annual_returns(returns=returns, ax=axes[2])
    pf.plot_monthly_returns_heatmap(returns=returns, ax=axes[3])
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    fig.suptitle('Return performance', fontsize=16, y=1.0)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'return_performance'), dpi=300)
        
    # Fig 3: risk performance
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), constrained_layout=False)
    axes = ax.flatten()
    pf.plot_drawdown_periods(returns=returns, ax=axes[0])
    pf.plot_rolling_volatility(returns=returns, factor_returns=benchmark_rets, ax=axes[1])
    pf.plot_drawdown_underwater(returns=returns, ax=axes[2])
    pf.plot_rolling_sharpe(returns=returns, ax=axes[3])
    axes[0].grid(True)
    axes[1].grid(True)
    axes[2].grid(True)
    axes[3].grid(True)
    fig.suptitle('Risk performance', fontsize=16, y=1.0)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, 'risk_performance'), dpi=300)
        