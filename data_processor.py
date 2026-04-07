import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
 
 
class Processor:
    """
    Parameters:
    tickers : default ["SPY", "TLT", "GLD"]
    lookback_years : Number of years of historical data to download.
    vol_window : Rolling window in trading days for realized volatility.
    vol_annualize : Trading days per year used to annualize realized volatility.
    trading_days_per_month : Scale factor for covariance: Sigma_monthly = 21 * Sigma_daily.
    """
    INITIAL_WEIGHTS = np.array([1/3, 1/3, 1/3])
 
    def __init__(
        self,
        tickers=None,
        lookback_years: int = 2,
        vol_window: int = 20,
        vol_annualize: int = 252,
        trading_days_per_month: int = 21,
    ):
        self.tickers = tickers or ["SPY", "TLT", "GLD"]
        self.lookback_years = lookback_years
        self.vol_window = vol_window
        self.vol_annualize = vol_annualize
        self.trading_days_per_month = trading_days_per_month
 
        # Run the run() function first and then you will have these values
        self.daily_df = None
        self.daily_log_returns = None
        self.monthly_returns = None
 
        self.end_date   = datetime.today().strftime("%Y-%m-%d")
        self.start_date = (
            datetime.today() - timedelta(days=lookback_years * 365)
        ).strftime("%Y-%m-%d")

    def run(self) -> None:
        df = self.download()
        df = self.clean(df)
        df = self.compute_daily_features(df)
 
        self.daily_df = df
        self.daily_log_returns = self.build_daily_log_returns(df)
        self.monthly_returns = self.resample_monthly_returns(df)

    def download(self) -> pd.DataFrame:
        print(f"\nDownloading data: {self.start_date} -> {self.end_date}")
        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )
        raw.columns = ["_".join([col[1], col[0]]).strip() for col in raw.columns]
        print(f"Raw shape: {raw.shape}")
        return raw
 

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Just some basic steps from MSML603 to clean the data first
        print("\nCleaning data...")
        missing_before = df.isnull().sum().sum()
        df.ffill(inplace=True)
        df.bfill(inplace=True)
        missing_after = df.isnull().sum().sum()
        print(f"Missing values: {missing_before} -> {missing_after}")

        n_dupes = df.index.duplicated().sum()
        if n_dupes > 0:
            print(f"Dropping {n_dupes} duplicate date(s)")
        df = df[~df.index.duplicated(keep="first")]

        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
 
        return df
 

    def compute_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nComputing daily features...")
 
        for ticker in self.tickers:
            close = f"{ticker}_Close"
            df[f"{ticker}_LogReturn"] = np.log(df[close] / df[close].shift(1))
            df[f"{ticker}_RealizedVol"] = (
                df[f"{ticker}_LogReturn"]
                .rolling(self.vol_window)
                .std()
                * np.sqrt(self.vol_annualize)
            )
 
        print(f"Log returns and {self.vol_window}-day realized vol computed")
        return df

    def build_daily_log_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        log_ret_cols = [f"{t}_LogReturn" for t in self.tickers]
        daily_log_returns = df[log_ret_cols].dropna().copy()
        daily_log_returns.columns = self.tickers
        print(f"Daily log returns shape (post warm-up drop): {daily_log_returns.shape}")
        return daily_log_returns
 

    def resample_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        close_cols = [f"{t}_Close" for t in self.tickers]
        monthly_close = df[close_cols].resample("ME").last()
        monthly_returns = monthly_close.pct_change().dropna()
        monthly_returns.columns = self.tickers
        print(f"Monthly returns shape: {monthly_returns.shape}")
        return monthly_returns
