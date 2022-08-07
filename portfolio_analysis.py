import numpy as np
import pandas as pd



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
    
def get_current_risk_free_rate():

    return None

def sharpe_ratio(**kwargs):
    '''
    Sharpe Ratio/Safety First Ratio
    
    Sharpe, William F. "Mutual Fund Performance." The Journal of Business 39, no. 1 (1966): 119-38. Accessed March 14, 2021. http://www.jstor.org/stable/2351741.
    '''
    
    
    risk_free_rate = kwargs.get("risk_free",get_current_risk_free_rate())

    returns = kwargs.get("period_return",kwargs.get("returns",[]))
    variance = kwargs.get("period_variance",None)

    ratio = (np.mean(returns) - risk_free_rate) / variance
    
    return ratio