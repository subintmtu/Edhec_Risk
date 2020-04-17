#!/usr/bin/env python
# coding: utf-8

# Load the 30 Value Weighted industry portfolio returns. Limit the analysis to returns of the 30 VW portfolios from 1997 onwards (1997 included, 1996 not included, 2018 included). Also load the Market Caps of each of the 30 industries. Run a backtest of comparing a CapWeighted vs an EW portfolio over the period. Though these two weighting schemes do not need any estimation, use an estimation period of 36 months so that we can compare it in the next few questions.

# In[1]:


import pandas as pd
import numpy as np
import edhec_risk_kit_206 as erk
import statsmodels.api as sm

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ind30_rets = erk.get_ind_returns(weighting="vw", n_inds=30)["1997":]
ind30_mcap = erk.get_ind_market_caps(n_inds=30, weights=True)["1997":]


# In[3]:


ewr = erk.backtest_ws(ind30_rets, weighting=erk.weight_ew, estimation_window=36)
cwr = erk.backtest_ws(ind30_rets, weighting=erk.weight_cw, cap_weights=ind30_mcap, estimation_window=36)


# In[4]:


btr = pd.DataFrame({"EW": ewr, "CW": cwr})


# 1. What was the annualized return over period for the Cap Weighted Industry Portfolio?
# 2. What was the annualized volatility over the period for the Cap Weighted Industry Portfolio?
# 3. What was the annualized return over period for the Equal Weighted Industry Portfolio?
# 4. What was the annualized volatility over the period for the Equal Weighted Industry Portfolio?

# In[5]:


erk.summary_stats(btr.dropna())


# Now using the same data and same period, re-run the EW backtest but this time create a tethered EW portfolio by removing Microcap industries using a threshold of 1% and a max cap-weight multiple of 2X

# In[6]:


ewtr = erk.backtest_ws(ind30_rets, cap_weights=ind30_mcap, max_cw_mult=2, microcap_threshold=.01, estimation_window=36 )
btr = pd.DataFrame({"EW": ewr, "EW-Tethered": ewtr, "CW": cwr})


# 5. What was the annualized return over the period for the tethered Equal Weighted Industry Portfolio?
# 6. What was the annualized volatility over the period for the tethered Equal Weighted Industry Portfolio?

# In[7]:


erk.summary_stats(btr.dropna())


# 7. What was Tracking Error between the Pure EW (without any tethering) portfolio and the CW portfolio?
# 8. What was Tracking Error between the Tethered EW portfolio and the CW portfolio?

# In[8]:


erk.tracking_error(ewr, cwr),erk.tracking_error(ewtr, cwr)


# Run a backtest for the same period (1997 onwards i.e. 1997 included, 1996 not included, 2018 included), using an estimation window of 36 months as above, to build the Global Minimum Variance Portfolio by estimating the Covariance matrix using Sample Covariance. (This might take a minute or so to run depending on your computer power!)
#
# 9. What was the annualized return over the period for the GMV Portfolio?
# 10. Use the same GMV portfolio as the previous question. What was the annualized volatility over the period for the GMV Portfolio?

# In[9]:


mv_s_r = erk.backtest_ws(ind30_rets, estimation_window=36, weighting=erk.weight_gmv, cov_estimator=erk.sample_cov)
btr = pd.DataFrame({"EW": ewr, "CW": cwr, "GMV-Sample": mv_s_r})
erk.summary_stats(btr.dropna())


# Run a backtest for the same period as the previous question, and again using an estimation window of 36 months as above, to build the Global Minimum Variance Portfolio but this time, estimating the Covariance matrix using Shrinkage between the Constant Correlation and Sample Covariance estimates using a delta of 0.25. (This might take a minute or so to run depending on your computer power!)
#
# 11. What was the annualized return over the period for this new Shrinkage-GMV Portfolio?
# 12. Using the same Shrinkage-GMV portfolio return series you used in the previous question, what was the annualized volatility over the period for the Shrinkage-GMV Portfolio?

# In[11]:


mv_sh_r = erk.backtest_ws(ind30_rets, estimation_window=36, weighting=erk.weight_gmv, cov_estimator=erk.shrinkage_cov, delta=0.25)
btr = pd.DataFrame({"EW": ewr, "CW": cwr, "GMV-Sample": mv_s_r, 'GMV-Shrink 0.25': mv_sh_r})
erk.summary_stats(btr.dropna())
