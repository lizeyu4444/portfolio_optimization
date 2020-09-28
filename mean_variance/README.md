# Mean-variance model

Originally forked from [MarkowitzPortfolioOptimization](https://github.com/rshemet/MarkowitzPortfolioOptimization).


## Improves:

* Modulize. Very similar to the workflow of machine learning project, which includes processing data, train and test split, optimize, backtest, build frontier, visualize.

* The original work imports look-ahead bias where the weights are computed from the whole data, which may result in extremely high return. 
  Here just use the function `train_test_split(training_period=5)` to split train and test data. The weights is computed from training data and applied on test data.

## Supports:

- Long/short
- Long only with no constraints 
- Long only with constraints

## Weakness:

- Not very stable because the expected return and variance are predicted on history data.
- The result is not very promising but still a good way to understand portfolio optimization.
