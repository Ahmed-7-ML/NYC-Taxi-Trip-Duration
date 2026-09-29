"""
Download NYC TLC parquet data directly from the official source.
df.sample: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.sample.html
pd.concat: https://pandas.pydata.org/docs/reference/api/pandas.concat.html
    - concatenate two or more pandas objects like DataFrames or Series along a particular axis
"""

from config.config import DataConfig
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataAcquisition:
    """Download NYC TLC parquet data directly by year and month."""
    
    def __init__(self, config:DataConfig):
        self.config = config
    
    def download_month(self, year:int, month:int) -> pd.DataFrame:
        """
        - Download one month of TLC data directly from the official source.
        - Retries are handled by Prefect @task(retries=3) in flow.py.
        """
        url = self.config.tlc_url_template.format(year=year, month=month)
        logger.info(f"Downloading {year}-{month:02d}:{url}")
        
        df = pd.read_parquet(url, columns=self.config.raw_columns)
        logger.info("Data Downloaded")
        
        return df
    
    def sample_month(self, df:pd.DataFrame, year:int, month:int) -> pd.DataFrame:
        """Sample rows from a single month, log what was kept."""
        # Take samples per month and Check it is less than the dataframe length
        n = min(self.config.sample_per_month, len(df))
        sample_df = df.sample(n=n, random_state=42)
        logger.info(f"{year}-{moth:02d}: sampled {n:,} / {len(df):,} rows")
        return sample_df
    
    def validate_data(self, df:pd.DataFrame) -> bool:
        """Validate loaded data has required columns."""
        required_cols = set(self.config.prediction_time_features + ['tpep_dropoff_datetime'])
        missing = required - set(df.columns)
        
        if missing:
            logger.error(f"❌Missing Cols: {missing}")
            return False
        
        logger.info(f"✅Schema Validation Success - {len(df.columns)} columns."
                    f"{df.isnull().sum():,} Missing Values.")
        return True

    def run(self) -> pd.DataFrame:
        """
        Download and combine all configured years and months.

        Supports multi-year retraining (e.g. train_years=[2019, 2020]).
        Peak memory = one full month + accumulated samples.
        """
        total_periods = len(self.config.train_years) * len(self.config.train_months)
        logger.info(f"📥 Downloading TLC data: years={self.config.train_years}")
        logger.info(f"   Months    : {self.config.train_months}")
        logger.info(f"   Per period: {self.config.samples_per_month:,} rows")
        logger.info(f"   Total     : ~{self.config.sample_size:,} rows "
                    f"({total_periods} periods)")
        
        chunks = []
        for year in self.config.train_years:
            for month in self.config.train_months:
                df_month = self.download_month(year, month)
                samples_df = self.sample_month(df_month, year, month)
                chunks.append(samples_df)
                del df_month
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"✅ Combined: {len(df):,} rows from {total_periods} periods")
        
        if not self.validate_data(df):
            raise ValueError("Schema Validation Failed - Missing Required Columns")

        # If it is downloaded safely
        return df
