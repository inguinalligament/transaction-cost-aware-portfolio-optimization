"""
Usage (in main.py):
    from data_preprocessor import PortfolioDataPreprocessor

    preprocessor = PortfolioDataPreprocessor()
    preprocessor.run()

    monthly_returns   = preprocessor.monthly_returns    # pd.DataFrame, shape (T, 3)
    monthly_regime    = preprocessor.monthly_regime     # pd.Series,    shape (T,)
    daily_log_returns = preprocessor.daily_log_returns  # pd.DataFrame, shape (D, 3) — for Sigma estimation
    daily_df          = preprocessor.daily_df           # pd.DataFrame, full daily data
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


class PortfolioDataPreprocessor:
    INITIAL_WEIGHTS = np.array([1/3, 1/3, 1/3])
    def __init__(
        self,
        tickers=None,
        lookback_years: int = 2,
        vol_window: int = 20,
        vol_annualize: int = 252,
        regime_ticker: str = "SPY",
        trading_days_per_month: int = 21,
    ):
        self.tickers = ["SPY", "TLT", "GLD"]
        self.lookback_years = lookback_years
        self.vol_window = vol_window
        self.vol_annualize = vol_annualize
        self.regime_ticker = regime_ticker
        self.trading_days_per_month = trading_days_per_month

        self.daily_df = None
        self.daily_log_returns = None
        self.monthly_returns = None
        self.monthly_regime = None

        self.end_date   = datetime.today().strftime("%Y-%m-%d")
        self.start_date = (
            datetime.today() - timedelta(days=lookback_years * 365)
        ).strftime("%Y-%m-%d")

    def run(self) -> None:
        print("=" * 55)
        print("  Portfolio Preprocessing Pipeline")
        print("=" * 55)

        df = self.download()
        df = self.clean(df)
        df = self._compute_daily_features(df)
        df = self.assign_daily_regime(df)

        self.daily_df = df
        self.daily_log_returns = self._build_daily_log_returns(df)
        self.monthly_returns = self.resample_monthly_returns(df)
        self.monthly_regime = self.resample_monthly_regime(df)

        self.validate()
        self._summary()

    def download(self) -> pd.DataFrame:
        print(f"\n[1/5] Downloading data: {self.start_date} → {self.end_date}")
        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )
        raw.columns = [
            "_".join([col[1], col[0]]).strip() for col in raw.columns
        ]
        print(f"Raw shape: {raw.shape}")
        return raw

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[2/5] Cleaning data")

        missing_before = df.isnull().sum().sum()
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        missing_after = df.isnull().sum().sum()
        print(f"Missing values:")
        print(f"Before: {missing_before}")
        print(f"After: {missing_after}")
        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            print(f"Dropping {n_dupes} duplicate date(s)")
        df = df[~df.index.duplicated(keep="first")]

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)

        return df

    def _compute_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[3/5] Computing daily features")

        for ticker in self.tickers:
            close = f"{ticker}_Close"

            df[f"{ticker}_LogReturn"] = np.log(
                df[close] / df[close].shift(1)
            )

            df[f"{ticker}_RealizedVol"] = (
                df[f"{ticker}_LogReturn"]
                .rolling(self.vol_window)
                .std()
                * np.sqrt(self.vol_annualize)
            )

        print(f"Log returns and {self.vol_window}-day realized vol computed")
        return df

    def _build_daily_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret_cols = [f"{t}_LogReturn" for t in self.tickers]
        daily_log_returns = (
            df[log_ret_cols]
            .dropna()                   
            .copy()
        )
        daily_log_returns.columns = self.tickers 
        print(f"    Daily log returns shape (post warm-up drop): {daily_log_returns.shape}")
        return daily_log_returns

    def assign_daily_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n[4/5] Assigning daily regime labels")

        vol_col = f"{self.regime_ticker}_RealizedVol"

        threshold = df[vol_col].median()
        print(f"    Regime threshold (median vol of {self.regime_ticker}): "
              f"{threshold:.4f}")

        df["Regime"] = np.where(df[vol_col] <= threshold, 1, 2)

        df.loc[df[vol_col].isna(), "Regime"] = 0

        counts = df["Regime"].value_counts().sort_index()
        print(f"Regime counts: {dict(counts)}")
        return df

    def resample_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        close_cols = [f"{t}_Close" for t in self.tickers]
        monthly_close = df[close_cols].resample("ME").last()

        monthly_returns = monthly_close.pct_change().dropna()

        monthly_returns.columns = self.tickers

        print(f"Monthly returns shape: {monthly_returns.shape}")
        return monthly_returns

    def resample_monthly_regime(self, df: pd.DataFrame) -> pd.Series:
        regime_clean = df["Regime"].replace(0, np.nan)
        monthly_regime = regime_clean.resample("ME").agg(
            lambda x: x.mode().iloc[0] if not x.dropna().empty else np.nan
        ).dropna().astype(int)

        monthly_regime = monthly_regime.reindex(self.monthly_returns.index).dropna().astype(int)
        return monthly_regime

    def validate(self) -> None:
        assert self.daily_log_returns.isnull().sum().sum() == 0, \
            "NaNs found in daily_log_returns"
        assert self.monthly_returns.isnull().sum().sum() == 0, \
            "NaNs found in monthly_returns"
        assert self.monthly_regime.isnull().sum() == 0, \
            "NaNs found in monthly_regime"
        assert set(self.monthly_regime.unique()).issubset({1, 2}), \
            "Unexpected regime values (expected 1 or 2)"
        assert self.monthly_returns.index.equals(self.monthly_regime.index), \
            "monthly_returns and monthly_regime indices do not align"
        assert list(self.daily_log_returns.columns) == self.tickers, \
            "daily_log_returns columns do not match tickers"
