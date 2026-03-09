# transaction-cost-aware-portfolio-optimization
Portfolio optimization with regime switching and transaction costs. Uses rolling estimates and convex optimization to rebalance a portfolio under changing market conditions.


regime-switching-portfolio-optimizer/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── sample/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_regime_detection.ipynb
│   ├── 03_optimization_experiments.ipynb
│   └── 04_backtest_results.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── regime_detection.py
│   ├── estimators.py
│   ├── optimizer.py
│   ├── transaction_costs.py
│   ├── backtest.py
│   ├── metrics.py
│   └── config.py
│
├── scripts/
│   ├── run_pipeline.py
│   ├── run_backtest.py
│   └── generate_report.py
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── logs/
│
└── tests/
    ├── test_regime_detection.py
    ├── test_optimizer.py
    ├── test_transaction_costs.py
    └── test_backtest.py
