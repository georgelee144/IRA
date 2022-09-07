import numpy as np
import pandas as pd
import datetime
import io

import requests
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS


def get_ln_return(price_t, price_0, t=1):

    try:
        compound_return = np.log(price_t/price_0)

    except:
        compound_return = np.nan

    return compound_return/t


def get_return(price_t, price_0, t=1):

    try:
        compound_return = price_t/price_0

    except:

        compound_return = np.nan

    return np.power(compound_return, 1/t) - 1

# def get_ln_returns(df,price_column='adjClose',shift=1,n_periods=1):
#     df[f'{price_column}_ln_price_returns'] = df[price_column].pct_change(shift).apply(lambda returns: np.log(1+returns)/n_periods)
#     return df
# def get_returns(df,price_column='adjClose',shift=1,n_periods=1):
#     df[f'{price_column}_ln_price_returns'] = df[price_column].pct_change(shift).apply(lambda returns: np.power(returns, 1/n_periods) - 1)
#     return df


def get_ln_returns(some_iter, shift=1, n_periods=1):

    price_series = pd.Series(some_iter)
    return_series = price_series.pct_change(shift).apply(
        lambda returns: np.log(1+returns)/n_periods)

    return return_series


def get_returns(some_iter, shift=1, n_periods=1):

    price_series = pd.Series(some_iter)
    return_series = price_series.pct_change(shift).apply(
        lambda returns: np.power(returns, 1/n_periods) - 1)

    return return_series


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


def get_historical_risk_free():

    return None


def clean_date_str(date_str):

    if len(date_str) == 4:
        return datetime.date(date_str, 12, 31)

    elif len(date_str) == 6:
        return datetime.date(date_str[:4], date_str[4:], 1)


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


def get_fama_french(factors: int = 5, frequency: str = "monthly") -> pd.DataFrame:

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


def fama_french_regression(df_security, df_benchmark=None, factors=5, frequency="monthly", rolling_window=None):

    fama_french_5_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']  # and RF
    fama_french_3_factors = ['Mkt-RF', 'SMB', 'HML']  # and RF

    if df_benchmark is None:
        df_benchmark = get_fama_french(factors=factors, frequency=frequency)

    df_full = df_security.merge(df_benchmark, how='left', on=["Date"])

    df_full['Security-RF'] = df_full['return'] - df_full['RF']

    df_full.dropna(inplace=True)

    if rolling_window:
        if factors == 5:
            mod = RollingOLS(df_full['Security-RF'],  df_full[fama_french_5_factors], window=rolling_window,)

            ols_model = mod.fit()

            return df_full, ols_model

        elif factors == 3:

            mod = RollingOLS(df_full['Security-RF'],  df_full[fama_french_3_factors], window=rolling_window)

            ols_model = mod.fit()

            return df_full, pd.concat([df_full,ols_model])

    else:
        if factors == 5:
            mod = sm.OLS(df_full['Security-RF'],
                         df_full[fama_french_5_factors])

            ols_model = mod.fit()

            return df_full, ols_model

        elif factors == 3:

            mod = sm.OLS(df_full['Security-RF'],
                         df_full[fama_french_3_factors])

            ols_model = mod.fit()

            return df_full, ols_model

    return df_full


def sharpe_ratio(**kwargs):
    """    
    Sharpe Ratio

    Sharpe, William F. "Mutual Fund Performance." The Journal of Business 39, no. 1 (1966): 119-38. Accessed March 14, 2021. http://www.jstor.org/stable/2351741.
    updated in 1994 "The Sharpe Ratio" https://web.stanford.edu/~wfsharpe/art/sr/SR.htm

    Returns:
        sharpe_ratio: float
    """

    risk_free_rate = kwargs.get("risk_free_rate", get_current_risk_free_rate())
    returns = kwargs.get("returns", [])

    ddof = kwargs.get("ddof", 1)

    period_return = kwargs.get("period_return", np.mean(returns))
    variance = kwargs.get("period_variance", np.var(returns, ddof=ddof))
    stdev = kwargs.get("period_stdev", np.sqrt(variance))

    sharpe_ratio = (period_return - risk_free_rate) / stdev

    return sharpe_ratio
