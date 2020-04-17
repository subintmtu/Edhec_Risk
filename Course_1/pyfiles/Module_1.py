#!/usr/bin/env python
# coding: utf-8

# ### Module 1 Test
# Question 1
# Read in the data in the file “Portfolios_Formed_on_ME_monthly_EW.csv” as we did in the lab sessions.We performed a series of analysis on the ‘Lo 10’ and the ‘Hi 10’ columns which are the returns of the lowest and highest decile portfolios. For purposes of this assignment, we will use the lowest and highest quintile portfolios, which are labelled ‘Lo 20’ and ‘Hi 20’ respectively.

# In[1]:


import pandas as pd
import edhec_risk_kit_129 as erk
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the dataset and calculate the returns

# In[2]:


def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 20', 'Hi 20']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


# ### Fn : Calculates annualized returns, default is monthly

# In[3]:


def get_annualized_returns(returnDataFrame : pd.DataFrame, n_periods_per_year = 12) :
    compounded_growth = (1 + returnDataFrame).prod()
    n_periods = returnDataFrame.shape[0]
    return (compounded_growth**(n_periods_per_year/n_periods) - 1)*100


# ### Fn : Calculates annualized volatility, default monthly

# In[4]:


def get_annualized_volatility(returnDataFrame : pd.DataFrame, n_periods_per_year = 12) :
    standard_deviation = returnDataFrame.std()
    return (standard_deviation*(n_periods_per_year)**0.5)*100


# ### Fn : Calculates drawdown

# In[5]:


def drawDown(return_series : pd.DataFrame) :
    wealthIndex = 1000*(1+return_series).cumprod()
    previous_peaks = wealthIndex.cummax()
    drawdowns = (wealthIndex - previous_peaks)/previous_peaks
    return pd.DataFrame({"Wealth": wealthIndex, 
                         "Previous Peak": previous_peaks, 
                         "Drawdown": drawdowns})


# In[6]:


ffme = get_ffme_returns()


# ### 1. What was the Annualized Return of the Lo 20 portfolio over the entire period?
# 
# To calculate the annualized returns over the entire period, the mean monthly return was calculated and annualized for the period.
# 
# Annualized return : $\Pi(1 + r_i)^{(T/t)}$ - 1
# 
# t : total number of periods
# 
# T : 4 - quarter, 12 - month, 252 -  days 

# In[7]:


round(get_annualized_returns(ffme["SmallCap"]),2)


# ### 2. What was the Annualized Volatility of the Lo 20 portfolio over the entire period? 
# 
# The volatility is the standard deviation of the returns annualized for the period. To annualize over the period,
# 
# Annualized volatility : r.std()*$\sqrt{T}$

# In[8]:


round(get_annualized_volatility(ffme["SmallCap"],12),2)


# ### 3. What was the Annualized Return of the Hi 20 portfolio over the entire period?

# In[9]:


round(get_annualized_returns(ffme["LargeCap"]),2)


# ### 4. What was the Annualized Volatility of the Hi 20 portfolio over the entire period ?

# In[10]:


round(get_annualized_volatility(ffme["LargeCap"],12),2)


# ### 5. What was the Annualized Return of the Lo 20 portfolio over the period 1999 - 2015 (both inclusive)?

# In[11]:


round(get_annualized_returns(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,0]),2)


# ### 6. What was the Annualized Volatility of the Lo 20 portfolio over the period 1999 - 2015 (both inclusive)? 
# 

# In[12]:


round(get_annualized_volatility(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,0]),2)


# ### 7. What was the Annualized Return of the Hi 20 portfolio over the period  1999 - 2015 (both inclusive)?

# In[13]:


round(get_annualized_returns(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,1]),2)


# ### 8. What was the Annualized Volatility of the Hi 20 portfolio over the period 1999 - 2015 (both inclusive)?

# In[14]:


round(get_annualized_volatility(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,1],12),2)


# ### 9. What was the Max Drawdown (expressed as a positive number) experienced over the 1999-2015 period in the SmallCap (Lo 20) portfolio?
# 

# In[15]:


abs(drawDown(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,0]).min()*100)


# ### 10. Over the period 1999-2015, at the end of which month did that maximum drawdown of the SmallCap (Lo 20) portfolio occur?

# In[16]:


drawDown(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,0]).idxmin()


# ### 11. What was the Max Drawdown (expressed as a positive number) experienced over the 1999-2015 period in the LargeCap (Hi 20) portfolio?

# In[17]:


abs(drawDown(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,1]).min()*100)


# ### 12. At the end of which month over the period 1999-2015 did that maximum drawdown on the LargeCap (Hi 20) portfolio occur?

# In[18]:


drawDown(ffme[(ffme.index >= "1999-01") & (ffme.index < "2016-01")].iloc[:,1]).idxmin()


# ## For the remaining questions, use the EDHEC Hedge Fund Indices data set that we used in the lab assignment and load them into Python.

# In[19]:


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

hfi = get_hfi_returns()


# ### 13. Looking at the data since 2009 (including all of 2009) through 2018 which Hedge Fund Index has exhibited the highest semideviation?

# In[20]:


hfi[hfi<0.0].loc['2009-01':].std(ddof=0).idxmax()


# ### 14. Looking at the data since 2009 (including all of 2009) which Hedge Fund Index has exhibited the lowest semideviation?

# In[21]:


hfi[hfi<0.0].loc['2009-01':].std(ddof=0).idxmin()


# ### 15. Looking at the data since 2009 (including all of 2009) which Hedge Fund Index has been most negatively skewed?

# In[22]:


hfi[hfi.index >= "2009-01"].skew().idxmin()


# ### 16. Looking at the data since 2000 (including all of 2000) through 2018 which Hedge Fund Index has exhibited the highest kurtosis?
# 

# In[23]:


hfi[hfi.index >= "2000-01"].kurtosis().idxmax()

