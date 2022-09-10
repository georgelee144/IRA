from typing_extensions import Self
import numpy as np
import pandas as pd
import datetime
import io
import os

import requests
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

TIINGO_API_KEY = os.environ['TIINGO_API_KEY']


class security:
    
    def __init__(self, ticker, **kwargs):

        self.ticker = ticker
        self.timeframe =  self.__check_timeframe__(kwargs.get("timeframe","monthly"))

        #default end_date is today
        self.end_date = kwargs.get("end_date",datetime.datetime.today().strftime("%Y-%m-%d"))
        #default start_date will be end_date - 5 years
        self.start_date = kwargs.get("start_date",(datetime.datetime.strptime(self.end_date,"%Y-%m-%d") - datetime.timedelta(days=1826)).strftime("%Y-%m-%d"))

        self.df = self.get_historical_data(self.ticker,self.timeframe, self.start_date, self.end_date)

        self.fama_french_model = None
        self.df_fama_french_model = None
        self.df_fama_french_rolling_model = None

    def __check_timeframe__(self,timeframe:str)->str:

        assert timeframe in ['daily','weekly','monthly','annually'] , "timeframe needs to be one of these values 'daily','weekly','monthly', or 'annual' "

        return timeframe

    @staticmethod
    def get_historical_data(ticker, timeframe="monthly", start_date=None, end_date=None):
        # start and end in RFC-3339 format "YYYY-MM-DD"
        # vaild intervals = Min, Hour, Day, Week and Month

        if start_date is None and end_date is None:
            end_date = datetime.datetime.today().strftime("%Y-%m-%d")
            # 5 years of days = 365*5+ 1 leap day
            start_date = (datetime.datetime.strptime(end_date,"%Y-%m-%d") - datetime.timedelta(days=1826)).strftime("%Y-%m-%d")

        payload = {
                  'startDate': start_date
                , 'endDate': end_date
                , 'token': TIINGO_API_KEY
                , 'format': 'csv'
                , 'resampleFreq': timeframe
                
                   }        
        url = f'https://api.tiingo.com/tiingo/daily/{ticker}/prices'
        r = requests.get(url, params=payload)

        for key,value in payload.items():
            print(key,value)
        # important_columns = ['date', 'close', 'adjClose', 'divCash', 'splitFactor']

        if r.status_code == 200:
            df = pd.read_csv(io.BytesIO(r.content))
        else:
            raise Exception(f"Response from TIINGO was not 200, TIINGO response code is {r.status_code}, {payload}") 

        def calculate_cumulative_split__(some_iter):
            array = np.array(some_iter)[::-1]
            cumprod_array = np.cumprod(array)[::-1]
            
            for index,value in enumerate(cumprod_array):

                if index == len(cumprod_array)-1:
                    break
                    
                elif value != cumprod_array[index+1]:
                    cumprod_array[index] = cumprod_array[index+1]
                    
                else:
                    pass
            
            return pd.DataFrame(cumprod_array)

        #calulating split factor as 
        df['cumulative_split_factor'] = calculate_cumulative_split__(df['splitFactor'])
        df['calculated_adj_divCash'] = df['divCash'] / df['cumulative_split_factor']

        df['return'] = (df['adjClose'] + df['calculated_adj_divCash']) / df['adjClose'].shift(1) - 1 
        df['ln_return'] = df['return'].apply(lambda returns: np.log(1+returns))

        if timeframe == "monthly":
            df['Date'] = df['date'].apply(lambda date_str: f"{date_str[:4]}{date_str[5:7]}")
        
        elif timeframe == "daily" or timeframe == "weekly":
            df['Date'] = df['date'].apply(lambda date_str: f"{date_str[:4]}{date_str[5:7]}{date_str[7:]}")
        
        elif timeframe == "annually":
            df['Date'] = df['date'].apply(lambda date_str: f"{date_str[:4]}")

        columns_to_return = ['Date', 'close', 'high', 'low', 'open', 'volume', 'adjClose', 'adjHigh',
       'adjLow', 'adjOpen', 'adjVolume', 'divCash', 'splitFactor',
       'cumulative_split_factor', 'calculated_adj_divCash', 'return','ln_return']
        return df[columns_to_return]

    @staticmethod
    def get_current_risk_free_rate(timeframe='10 yr'):
        """_summary_

        Args:
            timeframe (str, optional): _description_. Defaults to ''.

        Returns:
            _type_: _description_
        """
        treasury_timeframes = [timeframe.lower() for timeframe in [
            '1 Mo', '2 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr', '7 Yr', '10 Yr', '20 Yr', '30 Yr']]

        assert timeframe in treasury_timeframes, f"timeframe should be one one of these values {treasury_timeframes}"

        current_year_month = datetime.datetime.today().strftime("%Y%m")
        URL = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/{current_year_month}?type=daily_treasury_yield_curve&field_tdr_date_value_month={current_year_month}&page&_format=csv"
        df = pd.read_csv(URL)

        most_recent_row = df.iloc[0]

        return most_recent_row[timeframe]/100
    
    @staticmethod
    def clean_date_str(date_str):

        if len(date_str) == 4:
            return datetime.date(date_str, 12, 31)

        elif len(date_str) == 6:
            return datetime.date(date_str[:4], date_str[4:], 1)

    @staticmethod
    def get_fama_french(factors: int = 5, frequency: str = "monthly") -> pd.DataFrame:
        def get_df_from_fama_french_site(url, frequency=None):

            r = requests.get(url)
            df = pd.read_csv(io.BytesIO(r.content),
                            compression="zip", skiprows=[0, 1, 2])
            df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)

            if frequency == "annual":
                annual_stopping_index = df.loc[df["Date"] ==
                                            " Annual Factors: January-December "].index[0]

                df = df.iloc[annual_stopping_index + 2:]

            elif frequency == "monthly":
                annual_stopping_index = df.loc[df["Date"] ==
                                            " Annual Factors: January-December "].index[0]

                df = df.iloc[:annual_stopping_index]

            else:

                pass

            columns = df.columns[1:]

            for column in columns:
                df[column] = pd.to_numeric(df[column])

            return df

        if factors == 5:
            if frequency == "annual":
                url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

                return get_df_from_fama_french_site(url, frequency)

            elif frequency == "monthly":
                url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
                return get_df_from_fama_french_site(url, frequency)

            elif frequency == "daily":
                url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
                return get_df_from_fama_french_site(url)

            else:

                raise ValueError(
                    f"frequency, {frequency}, is not a vaild choice. Needs to be ['annual','monthly','daily']")

        elif factors == 3:
            if frequency == "annual":
                url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
                return get_df_from_fama_french_site(url, frequency)

            elif frequency == "monthly":
                url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
                return get_df_from_fama_french_site(url, frequency)

            elif frequency == "weekly":
                url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip"
                return get_df_from_fama_french_site(url)

            elif frequency == "daily":
                url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
                return get_df_from_fama_french_site(url)

            else:
                raise ValueError(
                    f"frequency, {frequency}, is not a vaild choice. Needs to be either 'annual','monthly', or 'daily']")
        else:
            raise ValueError(
                f"factors, {factors}, is not a vaild choice. Needs to be either 3 or 5")

    @classmethod
    def fama_french_regression(cls, factors=5, frequency="monthly", rolling_window=None):

        fama_french_5_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']  # and RF
        fama_french_3_factors = ['Mkt-RF', 'SMB', 'HML']  # and RF


        df_benchmark = cls.get_fama_french(factors=factors, frequency=frequency)

        df_full = cls.df.merge(df_benchmark, how='left', on=["Date"])

        df_full['Security-RF'] = df_full['return'] - df_full['RF']

        df_full.dropna(inplace=True)

        if rolling_window:
            if factors == 5:
                mod = RollingOLS(
                    df_full['Security-RF'],  df_full[fama_french_5_factors], window=rolling_window,)

                ols_model = mod.fit()

                cls.df_fama_french_rolling_model = pd.concat([df_full, ols_model])
                cls.df_fama_french_model = df_full

                return df_full, ols_model

            elif factors == 3:

                mod = RollingOLS(
                    df_full['Security-RF'],  df_full[fama_french_3_factors], window=rolling_window)

                ols_model = mod.fit()

                cls.df_fama_french_rolling_model = pd.concat([df_full, ols_model])
                cls.df_fama_french_model = df_full

                return df_full, pd.concat([df_full, ols_model])

        else:

            if factors == 5:
                mod = sm.OLS(df_full['Security-RF'],
                            df_full[fama_french_5_factors])

                ols_model = mod.fit()

                cls.df_fama_french_model = df_full
                cls.fama_french_model = ols_model

                return df_full, ols_model

            elif factors == 3:

                mod = sm.OLS(df_full['Security-RF'],
                            df_full[fama_french_3_factors])

                ols_model = mod.fit()

                cls.df_fama_french_model = df_full
                cls.fama_french_model = ols_model

                return df_full, ols_model

        return None

    

    @staticmethod
    def sharpe_ratio(**kwargs):
        """    
        Sharpe Ratio

        Sharpe, William F. "Mutual Fund Performance." The Journal of Business 39, no. 1 (1966): 119-38. Accessed March 14, 2021. http://www.jstor.org/stable/2351741.
        updated in 1994 "The Sharpe Ratio" https://web.stanford.edu/~wfsharpe/art/sr/SR.htm

        Returns:
            sharpe_ratio: float
        """

        risk_free_rate = kwargs.get("risk_free_rate", security.get_current_risk_free_rate())
        returns = kwargs.get("returns", [])

        ddof = kwargs.get("ddof", 1)

        period_return = kwargs.get("period_return", np.mean(returns))
        variance = kwargs.get("period_variance", np.var(returns, ddof=ddof))
        stdev = kwargs.get("period_stdev", np.sqrt(variance))

        sharpe_ratio = (period_return - risk_free_rate) / stdev

        return sharpe_ratio
