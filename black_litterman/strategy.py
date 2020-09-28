import datetime
import backtrader as bt

from model import build_data, bl_optimize


class SimpleWeightsStrategy(bt.Strategy):
    """
    Build portfolio with weights which are optimized from Black-litterman model.
    Note: it has look-ahead bias because the future stocks prices and market prices
      are exploited to calculate weights
    """
    params = dict(
        assets = [],
        rebalance_months = set([1,3,6,9])
    )
 
    def __init__(self):
        # Create weight dictionary
        #     { ticker: {'rebalanced': False, 'target_percent': target%} }
        self.rebalance_dict = dict()
        for i, d in enumerate(self.datas):
            ticker = d._name
            self.rebalance_dict[ticker] = dict()
            self.rebalance_dict[ticker]['rebalanced'] = False
            for asset in self.p.assets:
                if asset[0] == d._name:
                    self.rebalance_dict[ticker]['target_percent'] = asset[1]
 
    def next(self):
        # Rebalance assets at the first trading day of months in self.p.rebalance_months
        for i, d in enumerate(self.datas):
            dt = d.datetime.datetime()
            ticker = d._name
            pos = self.getposition(d).size
 
            if dt.month in self.p.rebalance_months and self.rebalance_dict[ticker]['rebalanced'] == False:
                print('{} Sending Order: {} | Month {} | Rebalanced: {} | Pos: {}'.
                    format(dt, ticker, dt.month, self.rebalance_dict[ticker]['rebalanced'], pos))
            
                self.order_target_percent(d, target=self.rebalance_dict[ticker]['target_percent']/100)
                self.rebalance_dict[ticker]['rebalanced'] = True
 
            # Reset the flag
            if dt.month not in self.p.rebalance_months:
                self.rebalance_dict[ticker]['rebalanced'] = False
                
    def notify_order(self, order):
        # Notify the order if completed
        date = self.data.datetime.datetime().date()
 
        if order.status == order.Completed:
            print('{} >> Order Completed >> Stock: {},  Ref: {}, Size: {}, Price: {}'.
                format(date, order.data._name, order.ref, order.size,
                    'NA' if not order.price else round(order.price,5)
                ))

    def notify_trade(self, trade):
        # Notify the trade if completed        
        date = self.data.datetime.datetime().date()
        if trade.isclosed:
            print('{} >> Notify Trade >> Stock: {}, Close Price: {}, Profit, Gross {}, Net {}'.
                format(date, trade.data._name, trade.price, 
                    round(trade.pnl, 2), round(trade.pnlcomm, 2))
            )
            

class RollingWeightsStrategy(bt.Strategy):
    """
    Build portfolio with weights which are changing using latest historical prices.
    """
    params = dict(
        rebalance_months = set([1, 3, 6, 9])
    )
    
    def __init__(self):
        self.prices, self.market_prices, self.mcap, self.views_dict, \
            self.confidences = build_data('../data/NSE')
        self.rebalance_dict = dict()
        self.optimized = False

    def update_weights(self, asset_weights):
        for ticker,w in asset_weights:
            if ticker not in self.rebalance_dict:
                self.rebalance_dict[ticker] = {
                    'rebalanced': False
                }
            self.rebalance_dict[ticker]['target_percent'] = w
 
    def next(self):
        # Compute weights using latest history data
        dt = self.data.datetime.datetime()
        if dt.month in self.p.rebalance_months and self.optimized is False:
            _, weights_df = bl_optimize(
                self.prices,
                self.market_prices,
                self.mcap,
                self.views_dict,
                end_date=dt,
                years_before_enddate=5,
                omega=None,
                confidences=self.confidences,
                weight_bounds=(0, 0.1),
                return_df=True,
                save_dir=None,
                visualize=False
            )
            if weights_df is None:
                return
                
            asset_weights = list(zip(weights_df.index, weights_df['weights']))
            self.update_weights(asset_weights)
            self.optimized = True

        if dt.month not in self.p.rebalance_months:
            self.optimized = False

        if not self.rebalance_dict:
            return

        # Rebalance assets at the first trading day of months in self.p.rebalance_months
        for i, d in enumerate(self.datas):
            dt = d.datetime.datetime()
            ticker = d._name
            pos = self.getposition(d).size
 
            if dt.month in self.p.rebalance_months and self.rebalance_dict[ticker]['rebalanced'] is False:
                print('{} Sending Order: {} | Month {} | Rebalanced: {} | Pos: {}'.
                    format(dt, ticker, dt.month, self.rebalance_dict[ticker]['rebalanced'], pos))
            
                self.order_target_percent(d, target=self.rebalance_dict[ticker]['target_percent']/100)
                self.rebalance_dict[ticker]['rebalanced'] = True
 
            # Reset the flag
            if dt.month not in self.p.rebalance_months:
                self.rebalance_dict[ticker]['rebalanced'] = False
                
    def notify_order(self, order):
        # Notify the order if completed
        date = self.data.datetime.datetime().date()
 
        if order.status == order.Completed:
            print('{} >> Order Completed >> Stock: {},  Ref: {}, Size: {}, Price: {}'.
                format(date, order.data._name, order.ref, order.size,
                    'NA' if not order.price else round(order.price,5)
                ))

    def notify_trade(self, trade):
        # Notify the trade if completed        
        date = self.data.datetime.datetime().date()
        if trade.isclosed:
            print('{} >> Notify Trade >> Stock: {}, Close Price: {}, Profit, Gross {}, Net {}'.
                format(date, trade.data._name, trade.price, 
                    round(trade.pnl, 2), round(trade.pnlcomm, 2))
            )
