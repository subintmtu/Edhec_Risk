#!/usr/bin/env python
# coding: utf-8

# Load the 49 Value Weighted industry portfolio returns. Limit the analysis to returns of the 49 VW portfolios from the most recent 5 years for which you have data i.e 2014-2018 both years inclusive. Also load the Market Caps of each of the 49 industries. Assume that the cap-weights as of the first month (2014-01) are the cap-weights we’ll use for this analysis.
#

# In[1]:


import pandas as pd
import numpy as np
import edhec_risk_kit_206 as erk
import statsmodels.api as sm

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["2014":]
ind49_mcap = erk.get_ind_market_caps(n_inds=49, weights=True)["2014":]
# ind_w = pd.DataFrame(np.transpose([ind49_mcap.values[0]/ind49_mcap.values[0].sum()]), index=ind49_rets.columns, columns=["Weights"])


# In[3]:


ind_w = ind49_mcap.values[0]/ind49_mcap.values[0].sum()
ind49_cov = ind49_rets.cov()


# 1. Which Industry had the highest risk contribution in the cap-weighted portfolio?

# In[4]:


EW = erk.risk_contribution(ind_w,ind49_cov).idxmax()


# 2. What was the highest risk contribution from any one industry in the cap-weighted portfolio?

# In[5]:


erk.risk_contribution(ind_w,ind49_cov).values.max()*100


# 3. Which Industry had the highest risk contribution in the equal-weighted portfolio?

# In[6]:


erk.risk_contribution(erk.weight_ew(ind49_rets), ind49_cov).idxmax()


# 4. What was the highest risk contribution from any one industry in the equal-weighted portfolio?

# In[7]:


erk.risk_contribution(erk.weight_ew(ind49_rets), ind49_cov).values.max()*100


# Using the Sample Covariance over the 5 year period, and the initial capweights over that period, compute the weights of the ERC portfolio.

# In[8]:


erc_w = pd.DataFrame( np.transpose(erk.equal_risk_contributions(ind49_cov)), index = ind49_rets.columns, columns=["Weight"] )


#  5. What sector portfolio is assigned the highest weight in the ERC portfolio?

# In[9]:


erc_w.idxmax()


# 6. What is the weight of the sector portfolio that is assigned the highest weight in the ERC portfolio?

# In[10]:


erc_w.values.max()*100


# 7. What sector portfolio is assigned the lowest weight in the ERC portfolio?

# In[11]:


erc_w.idxmin()


# 8. What is the weight of the sector portfolio that is assigned the lowest weight in the ERC portfolio?

# In[12]:


erc_w.values.min()*100


# In[13]:


CW = erk.risk_contribution(ind_w,ind49_cov)


# In[14]:


EW = erk.risk_contribution(erk.weight_ew(ind49_rets), ind49_cov)


# 9. In the cap-weighted portfolio what is the difference in risk contributions between the largest contributor to portfolio risk and the smallest contributor to portfolio risk?

# In[15]:


(CW.max() - CW.min())*100


# 10. In the equal-weighted portfolio what is the difference in risk contributions between the largest contributor to portfolio risk and the smallest contributor to portfolio risk?

# In[16]:


(EW.max() - EW.min())*100
