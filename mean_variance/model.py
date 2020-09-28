import os
import datetime

import numpy as np 
import pandas as pd 
from glob import glob
from shutil import copyfile 
from matplotlib import pyplot as plt 
from scipy.optimize import minimize

RISK_FREE_FILE = '../data/TBillRate/TB3MS.csv'



def read_OHLC(filepath, only_close=False):
    """read OHLC file
    """
    df = pd.read_csv(filepath) 
    df.columns = [col.lower() for col in df.columns]
    if only_close:
        df = df[['date', 'close']].set_index('date')
    else:
        df = df[['date', 'open', 'high', 'low', 'close']].set_index('date')
    df.index = pd.to_datetime(df.index)
    return df


class ETFPortfolio(object):
    """Backtest mean-variance model for ETF portfolio, compare the performance of 
       optimized strategy with its original benchmark
    """
    def __init__(self, 
            market,
            min_data_len=10
        ):
        self.market = market
        self.min_data_len = min_data_len

    def load_data(self, portf_path, portf_OHLC):
        """Leave assets with more than 10 years returns and load them into dataframe
        params:
          portf_path: str, assets information file of portfolio
          portf_OHLC: str, OHLC price file of portfolio
        """
        self.portfolio_assets = pd.read_csv(portf_path).set_index('Ticker')
        self.portfolio_assets.index = self.portfolio_assets.index.astype(str)
        self.portfolio_OHLC = read_OHLC(portf_OHLC)

        # Calculate how many years of data available for each asset
        # and leave those with more than 10 years
        assets_summary = pd.DataFrame(columns=['Asset', 'Data Yrs', 'ETF Weight']).set_index('Asset')
        for asset in self.portfolio_assets.index:
            filepath = 'Data/Stocks/{}.{}.txt'.format(asset.lower(), self.market)
            if os.path.isfile(filepath):
                asset_data = read_OHLC(filepath)
                port_weight = self.portfolio_assets.loc[asset, 'Weight (%)']
                date_span = (asset_data.index[-1] - asset_data.index[0]) / np.timedelta64(1, 'Y')
                assets_summary.loc[asset, :] = date_span, port_weight

        self.assets_left = assets_summary[assets_summary['Data Yrs'] > self.min_data_len]

        # Begin DF with oldest asset, this way we dont lose older data
        oldest_asset = '600028' # 'ADSK' #'600028' 
        filepath = '../data/Stocks/{}.{}.txt'.format(oldest_asset, self.market)
        oldest_data = read_OHLC(filepath)

        self.asset_matrix = pd.DataFrame()
        self.asset_matrix[oldest_asset] = oldest_data['close']

        # Load each asset file and save to a dataframe
        for asset in self.assets_left.index:
            print(asset)
            filepath = '../data/Stocks/{}.{}.txt'.format(asset.lower(), self.market)
            asset_df = read_OHLC(filepath)
            self.asset_matrix[asset] = asset_df['close']


    def train_test_split(self, training_period):
        """Split the return matrix
        Params:
          training_period: int, years of training data
        """
        self.training_period = training_period
        self.end_date_test = self.asset_matrix.index[-1]
        self.end_date_train = self.end_date_test - datetime.timedelta(
            days=(self.min_data_len - self.training_period) * 365)
        self.start_date_train = self.end_date_test - datetime.timedelta(
            days=self.min_data_len * 365 + 4)

        # Split close price of each asset in a portfolio
        self.asset_matrix_train = self.asset_matrix.loc[self.start_date_train:self.end_date_train]
        self.asset_matrix_test = self.asset_matrix.loc[self.end_date_train:self.end_date_test]
        print('Training data: from {} to {}'.format(self.start_date_train, self.end_date_train))
        print('Test data: from {} to {}'.format(self.end_date_train, self.end_date_test))

        # Split close price of total portfolio
        portf_test = self.portfolio_OHLC.loc[self.end_date_train:self.end_date_test]
        portf_test.loc[:, 'close'] *= (100 / portf_test.iloc[0]['close'])
        self.portf_test = portf_test


    def process_data(self):
        """Generate the vector of annualized returns, the asset variance-
           covariance matrix, risk-free rate measure
        """
        stocks_universe = self.asset_matrix_train.columns
        self.mu_vector = pd.DataFrame(index=stocks_universe, columns=['mu'])

        for stock in stocks_universe:
            series = self.asset_matrix_train[stock]
            log_returns = np.log(series / series.shift(1)).dropna()
            ann_log_return = np.sum(log_returns) / self.training_period
            self.mu_vector.loc[stock] = ann_log_return

        self.log_returns_matrix = np.log(self.asset_matrix_train / self.asset_matrix_train.shift(1))
        self.vcv_matrix = self.log_returns_matrix.cov() * 252

        # Risk-free rate is taken as the first month after the beginning of training period
        rf_raw = pd.read_csv(RISK_FREE_FILE, parse_dates=['DATE']).set_index('DATE')
        self.rf = rf_raw.loc[[rf_raw.index > self.start_date_train][0]].iloc[0, 0] / 100
        self.rf_test = rf_raw.loc[[rf_raw.index > self.end_date_train][0]].iloc[0, 0] / 100


    def optimize(self, constraint=None, weightcap=None):
        """Optimize the target and get the asset weights
        Params:
          constraint: str, `longonly` or None, whether assets can be shorted. When the  
            constraint is None, there is closed-form solution
          weightcap: if constraint is `longonly`, set the maximum weight of each asset
        """
        def minimum_sharpe_ratio_fn(mu, vcv, rf):
            """Closed-form solution to minizing portfolio sharpe ratio, s.t. weights summing
               up to 1
            Params:
              mu: vector, logged annualized returns
              vcv: matrix, annualized covariance matrix of returns
              rf: float, risk-free rate
            Return:
              w: vector, weights for given return
              var: float, expected variance
            """
            vcv_inverse = np.linalg.inv(vcv)
            ones_vect = np.ones(len(mu))[:, np.newaxis]

            num = vcv_inverse @ (mu - rf * ones_vect)
            den = ones_vect.T @ vcv_inverse @ (mu - rf * ones_vect)
            w = num.to_numpy() / den.to_numpy()

            exc_ret = w.T @ mu.to_numpy() - rf
            std_dev = (w.T @ vcv.to_numpy() @ w) ** 0.5
            sharpe = exc_ret / std_dev

            return w, sharpe[0][0]

        def inverse_sharpe_ratio_fn(w):
            """Target function to minimize, i.e. maximizing sharpe ratio
            Params:
              w: vector, asset weights of portfolio
            Return:
              float, inverse sharpe ratio of portfolio
            """
            exc_ret = w.T @ self.mu_vector - self.rf
            std_dev = (w.T @ self.vcv_matrix @ w) ** 0.5
            return -(exc_ret / std_dev)

        self.constraint = constraint
        self.weightcap = weightcap

        if constraint is None:
            weights, sharpe = minimum_sharpe_ratio_fn(self.mu_vector, self.vcv_matrix, self.rf)

        elif constraint == 'longonly':
            #  Constraint 1: weights summing up to 1
            if weightcap is None:
                weightcap = 1
            cons = ({'type': 'eq', 'fun': lambda x: x.T @ np.ones(len(x))[:np.newaxis] - 1})

            # Constraint 2: weights range between 0 ~ self.weightcap
            bounds = []
            for j in range(self.mu_vector.shape[0]):
                bounds.append((0, self.weightcap))

            x_0 = np.zeros(len(self.mu_vector))
            x_0[0] = 1

            # Optimize
            weights = minimize(inverse_sharpe_ratio_fn, x_0, bounds=bounds, constraints=cons).get('x')
        else:
            raise ValueError('Unsupported constraint type')

        self.weights = pd.DataFrame(index=self.mu_vector.index, columns=['Weight'])
        self.weights['Weight'] = weights


    def build_frontier(self):
        """Given a range of returns, get minimum variances(risk), which form the frontier
        """
        def minimum_variance_fn(mu, vcv, pi):
            """Closed-form solution to minizing portfolio variance, s.t. weights summing
               up to 1 and a fixed return
            Params:
              mu: vector, logged annualized returns
              vcv: matrix, annualized covariance matrix of returns
              pi: float, target level of portfolio return
            Return:
              w: vector, weights for given return
              var: float, expected variance
            """
            vcv_inverse = np.linalg.inv(vcv)
            ones_vect = np.ones(len(mu))[:, np.newaxis]

            a = ones_vect.T @ vcv_inverse @ ones_vect
            b = mu.T @ vcv_inverse @ ones_vect
            c = mu.T.to_numpy() @ vcv_inverse @ mu

            a = a[0][0]
            b = b.loc['mu', 0]
            c = c.loc[0, 'mu']

            num1 = (a * vcv_inverse @ mu - b * vcv_inverse @ ones_vect) * pi
            num2 = (c * vcv_inverse @ ones_vect - b * vcv_inverse @ mu)
            den = a * c - b**2

            w = (num1 + num2) / den

            var = (w.T.to_numpy() @ vcv.to_numpy() @ w.to_numpy()) ** 0.5

            return w, var
        
        def variance_fn(w):
            """Target function to minimize, i.e. minimizing variance
            Params:
              w: vector, weights
            Return:
              float, portfolio variance
            """
            return w.T @ self.vcv_matrix @ w

        # Mean and variance of each stocks
        annualized_returns = self.log_returns_matrix.sum() / self.training_period
        annualized_variance = self.log_returns_matrix.var() * 252
        self.mean_variance_df = pd.DataFrame(index=annualized_returns.index)
        self.mean_variance_df['mu'] = annualized_returns
        self.mean_variance_df['sigma'] = annualized_variance**0.5

        ranked_positive_returns = [i for i in annualized_returns if i > 0]
        ranked_positive_returns.sort()

        if self.constraint is None:
            lo_bound_return = ranked_positive_returns[0]
            hi_bound_return = ranked_positive_returns[-1] + 1.5

            frontier_df = pd.DataFrame(columns=['var'], index=np.arange(
                lo_bound_return, hi_bound_return, (hi_bound_return - lo_bound_return)/20))

            for pi in frontier_df.index:
                _, var = minimum_variance_fn(self.mu_vector, self.vcv_matrix, pi)
                frontier_df.loc[pi] = var

        elif self.constraint == 'longonly':
            lo_bound_return = ranked_positive_returns[0]
            hi_bound_return = ranked_positive_returns[-1]

            frontier_df = pd.DataFrame(columns=['var'], index=np.append(np.arange(
                lo_bound_return, hi_bound_return, (hi_bound_return - lo_bound_return)/20), hi_bound_return))

            for pi in frontier_df.index:
                #  Constraint 1: portfolio return equal to a fixed return and 
                #                weights summing up to 1
                cons = (
                    {'type': 'eq', 'fun': lambda x: x.T @ self.mu_vector - pi},
                    {'type': 'eq', 'fun': lambda x: x.T @ np.ones(len(x))[:np.newaxis] - 1}
                )

                # Constraint 2: weights range between 0 ~ self.weightcap
                bounds = []
                for j in range(self.mu_vector.shape[0]):
                    bounds.append((0, self.weightcap))

                x_0 = np.zeros(len(self.mu_vector))
                x_0[0] = 1

                # Optimize
                w = minimize(variance_fn, x_0, bounds=bounds, constraints=cons).get('x')
                var = (w.T @ self.vcv_matrix @ w) ** 0.5
                frontier_df.loc[pi] = var

        self.frontier_df = frontier_df
        
        # Capital allocation line, portfolio consists of risk-free asset and risk assets
        cal_df = pd.DataFrame(columns=['mu'], index=np.arange(0, 0.9, 0.05))
        exp_ret = (self.weights.T @ self.mu_vector).iloc[0, 0]
        exp_vol = (self.weights.T @ self.vcv_matrix @ self.weights).iloc[0, 0] ** 0.5
        self.tangency_port = (exp_ret, exp_vol) 

         # The slope is same as sharpe, the bigger the better
        slope = (exp_ret - self.rf / 100) / exp_vol 
        for i in cal_df.index:
            cal_df.loc[i] = self.rf / 100 + i * slope
        self.cal_df = cal_df


    def backtest(self):
        """Backtest the data given the optimal weights
        """
        portf_weights = self.assets_left.copy()
        portf_weights['Our Weights'] = self.weights * 100
        self.portf_weights = portf_weights.reindex(self.weights.index)

        # Predict the test data using optimized weights
        log_returns_test = np.log(self.asset_matrix_test / self.asset_matrix_test.shift(1)) + 1
        portf_pred = pd.DataFrame(index=log_returns_test.index, columns=log_returns_test.columns)

        portf_pred.iloc[1] = log_returns_test.iloc[1].mul(self.portf_weights['Our Weights'])
        for i in range(2, portf_pred.shape[0]):
            portf_pred.iloc[i] = log_returns_test.iloc[i].mul(portf_pred.iloc[i-1])

        portf_pred = portf_pred.iloc[1:]

        portf_pred['Overall Portfolio'] = portf_pred.sum(axis=1)
        self.portf_pred = portf_pred

        # Calculate statistics
        self.strat_ann_ret = (self.portf_pred['Overall Portfolio'].iloc[-1] \
            / self.portf_pred['Overall Portfolio'].iloc[0]) ** (1 / (self.min_data_len - self.training_period)) - 1
        self.strat_ann_vol = (self.portf_pred['Overall Portfolio'].pct_change().var() * 252) ** 0.5
        self.strat_sharpe_ratio = (self.strat_ann_ret - self.rf_test) / self.strat_ann_vol

        strat_weights = list(self.weights['Weight'])
        strat_weights.sort(reverse = True)
        self.strat_top5_concentration = np.sum(strat_weights[0:5]) * 100

        self.fund_ann_ret = (self.portf_test['close'].iloc[-1] \
            / self.portf_test['close'].iloc[0]) ** (1 / (self.min_data_len - self.training_period)) - 1
        self.fund_ann_vol = (self.portf_test['close'].pct_change().var() * 252) ** 0.5

        self.fund_sharpe_ratio = (self.fund_ann_ret - self.rf_test) / self.fund_ann_vol
        self.fund_top5_concentration = self.assets_left['ETF Weight'].iloc[0:5].sum()


    def visualize(self):
        """Visualize results and compares to the ETF's holdings and 
           how the performance looks
        """
        # Plot efficient frontier and CAL
        plt.figure(figsize=(12, 8))
        plt.plot(self.frontier_df['var']*100, self.frontier_df.index*100, label='Efficient Frontier')
        plt.plot(self.cal_df.index*100, self.cal_df['mu']*100,
            label='Capital Allocation Line', linestyle='dashed', linewidth=1)
        plt.scatter(self.mean_variance_df['sigma']*100, self.mean_variance_df['mu']*100,
            s=2, c='r', marker='x', label='Individual Stocks')
        plt.scatter(self.tangency_port[1]*100, self.tangency_port[0]*100, label='Tangency portfolio', marker='x')
        plt.title('Efficient frontier')
        plt.xlabel('Annualized Standard Deviation (%)')
        plt.ylabel('Annualized Return (%)')
        plt.legend()
        img_path = 'Outputs/Frontier_cons_{}_cap_{}_train_{}yrs.png'.format(
            self.constraint, self.weightcap, self.training_period)
        plt.savefig(img_path, bbox_inches='tight', dpi=400)

        # Plot backtest results
        fig, (ax1, ax2) = plt.subplots(2, figsize=(12,8))
        fig.suptitle('Fund X vs Replication Strategy Overview')

        width = 0.35
        bar_chart_x = np.arange(len(self.portf_weights.index))
        ax1.bar(bar_chart_x - width/2, self.portf_weights['Our Weights'], width, label='Replication Strategy Weights')
        ax1.bar(bar_chart_x + width/2, self.portf_weights['ETF Weight'], width, label='Fund X Weights')
        ax1.tick_params(labelrotation=90, labelsize=8)
        ax1.set_xticks(bar_chart_x)
        ax1.set_xticklabels(self.portf_weights.index)
        ax1.set_ylabel('Allocated weight (%)')
        ax1.axhline(y=0, linewidth=0.4, color='k')
        ax1.legend()

        ax2.plot(self.portf_pred.index, self.portf_pred['Overall Portfolio'], label='Replication Strategy Performance')
        ax2.plot(self.portf_test.index, self.portf_test['close'], label='Fund X Performance')
        ax2.tick_params(labelsize=8)
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Performance (rebased to 100)')
        ax2.legend()
        img_path = 'Outputs/Performance_cons_{}_cap_{}_train_{}yrs.png'.format(
            self.constraint, self.weightcap, self.training_period)
        plt.savefig(img_path, dpi=400, bbox_inches='tight')


    def summary(self, return_portf=False):
        """Output 4 metrics for portfolio and current strategy:
           sharpe ratio, annualized return, annualized volatility
           average weight in top 5 positions
        Params:
          return_portf: whether to return portfolio metrics
        Return:
          strat_sum or (strat_sum, portf_sum)
        """
        strat_sum = (
            self.strat_sharpe_ratio,\
            self.strat_ann_ret,\
            self.strat_ann_vol,\
            self.strat_top5_concentration
        )

        if return_portf:
            portf_sum = (
                self.fund_sharpe_ratio,\
                self.fund_ann_ret,\
                self.fund_ann_vol,\
                self.fund_top5_concentration
            )
            return portf_sum, strat_sum
        
        return strat_sum
        

if __name__ == '__main__':

    portf_path = '../data/ETF/iSharesExpTechSoftware.csv'
    portf_OHLC = '../data/ETF/iSharesExpTechSoftwarePerf.csv'
    portf = ETFPortfolio()

    experiments=(
        (None, None),\
        ('longonly', None),\
        ('longonly', 0.1),\
        ('longonly', 0.2)
    )

    summary = {
        "Fund X": '',\
        "Long/Short": '',\
        "Long Only": '',\
        "LO 10%Cap": '',\
        "LO 20%Cap": ''
    }

    portf.load_data(portf_path, portf_OHLC)
    portf.train_test_split(training_period=7)
    portf.process_data()

    k = 1
    for exp in experiments:
        portf.optimize(constraint=exp[0], weightcap=exp[1])
        portf.backtest()
        # portf.build_frontier()
        # portf.visualize()
        if k == 1:
            summary['Fund X'], summary['Long/Short'] = portf.summary(return_portf=True)
        else:
            summary[list(summary.keys())[k]] = portf.summary(return_portf=False)
        k+=1

    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    for key, val in summary.items():
        color = 'tab:blue' if key == 'Fund X' else 'tab:orange'
        axs[0,0].bar(key, val[0], label = key, color=color)
        axs[0,0].set_title('Sharpe Ratios')
        axs[0,0].set_ylabel('Sharpe Ratio')
        axs[0,0].tick_params(labelrotation=45, labelsize = 8)
        axs[0,1].bar(key, val[1] * 100, label = key, color=color)
        axs[0,1].set_title('Annualized Performance')
        axs[0,1].set_ylabel('Return (%)')
        axs[0,1].tick_params(labelrotation=45, labelsize = 8)
        axs[1,0].bar(key, val[2] * 100, label = key, color=color)
        axs[1,0].set_title('Annualized Volatility')
        axs[1,0].set_ylabel('Std. Deviation (%)')
        axs[1,0].tick_params(labelrotation=45, labelsize = 8)
        axs[1,1].bar(key, val[3], label = key, color=color)
        axs[1,1].set_title('Top 5 Holdings Concentration')
        axs[1,1].set_ylabel('(%) Allocation')
        axs[1,1].tick_params(labelrotation=45, labelsize = 8)
    
    fig.suptitle('Summary Statistics')
    plt.savefig('Outputs/Summary.png', dpi=400, bbox_inches='tight')

