import numpy as np
import pandas as pd
import datetime
import io

import requests


def get_ln_return(price_t,price_0,t=1):

    try:
        compound_return = np.log(price_t/price_0)

    except:
        compound_return = np.nan

    return compound_return/t

def get_return(price_t,price_0,t=1):

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

def get_ln_returns(some_iter,shift=1,n_periods=1):

    price_series = pd.Series(some_iter)
    return_series = price_series.pct_change(shift).apply(lambda returns: np.log(1+returns)/n_periods)
    
    return return_series

def get_returns(some_iter,shift=1,n_periods=1):

    price_series = pd.Series(some_iter)
    return_series = price_series.pct_change(shift).apply(lambda returns: np.power(returns, 1/n_periods) - 1)
    
    return return_series
    
def get_current_risk_free_rate(timeframe = '10 yr'):
    """_summary_

    Args:
        timeframe (str, optional): _description_. Defaults to ''.

    Returns:
        _type_: _description_
    """
    treasury_timeframes = [timeframe.lower() for timeframe in ['1 Mo', '2 Mo', '3 Mo', '6 Mo', '1 Yr', '2 Yr', '3 Yr', '5 Yr','7 Yr', '10 Yr', '20 Yr', '30 Yr']]


    assert timeframe in treasury_timeframes, f"timeframe should be one one of these values {treasury_timeframes}"

    current_year_month = datetime.datetime.today().strftime("%Y%m")
    URL = f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/all/{current_year_month}?type=daily_treasury_yield_curve&field_tdr_date_value_month={current_year_month}&page&_format=csv"
    df = pd.read_csv(URL)

    most_recent_row = df.iloc[0]

    return most_recent_row[timeframe]/100

def get_historical_risk_free():

def get_fama_french(factors:int = 5,frequency="monthly")->pd.DataFrame:

    if factors == 5:
        if frequency == "monthly":
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"
        elif frequency == "daily":
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
        else:
            raise ValueError

    elif factors == 3:
        if frequency == "monthly":
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
        elif frequency == "weekly":
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_weekly_CSV.zip"
        elif frequency == "daily":
            url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        else:
            raise ValueError
    else:
        raise ValueError

    r = requests.get(url)
    df = pd.read_csv(io.BytesIO(r.content), compression="zip", skiprows=[0,1,2] )

    return df

def sharpe_ratio(**kwargs):

    """    
    Sharpe Ratio
    
    Sharpe, William F. "Mutual Fund Performance." The Journal of Business 39, no. 1 (1966): 119-38. Accessed March 14, 2021. http://www.jstor.org/stable/2351741.
    updated in 1994 "The Sharpe Ratio" https://web.stanford.edu/~wfsharpe/art/sr/SR.htm

    Returns:
        sharpe_ratio: float
    """
    
    risk_free_rate = kwargs.get("risk_free_rate",get_current_risk_free_rate())
    returns = kwargs.get("returns",[])

    ddof = kwargs.get("ddof",1)

    period_return = kwargs.get("period_return",np.mean(returns))
    variance = kwargs.get("period_variance",np.var(returns,ddof=ddof))
    stdev = kwargs.get("period_stdev",np.sqrt(variance))

    print(period_return,risk_free_rate,stdev)
    sharpe_ratio = ( period_return - risk_free_rate) / stdev
    
    return sharpe_ratio