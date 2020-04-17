#!/usr/bin/env python
# coding: utf-8

# ### Module 2 Assignment
# 

# In[1]:


import pandas as pd
import numpy as np
import edhec_risk_kit_129 as erk
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Use the EDHEC Hedge Fund Indices data set that we used in the lab assignment as well as in the previous week’s assignments. Load them into Python and perform the following analysis based on data since 2000 (including all of 2000)

# In[2]:


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

hfi = erk.get_hfi_returns()


# ### 1. What was the Monthly Parametric Gaussian VaR at the 1% level (as a +ve number) of the Distressed Securities strategy?

# In[3]:


round(erk.var_gaussian(hfi[hfi.index >= "2000-01"].loc[:,'Distressed Securities'],1)*100,2)


# ###  2. Use the same data set at the previous question. What was the 1% VaR for the same strategy after applying the Cornish-Fisher Adjustment?

# In[4]:


round(erk.var_gaussian(hfi[hfi.index >= "2000-01"].loc[:,'Distressed Securities'],1, modified=True)*100,2)


# ### 3. Use the same dataset as the previous question. What was the Monthly Historic VaR at the 1% level (as a +ve number) of the Distressed Securities strategy?

# In[5]:


round(erk.var_historic(hfi[hfi.index >= "2000-01"].loc[:,'Distressed Securities'],1)*100,2)


# ### Next, load the 30 industry return data using the erk.get_ind_returns() function that we developed during the lab sessions. For purposes of the remaining questions, use data during the 5 year period 2013-2017 (both inclusive) to estimate the expected returns as well as the covariance matrix. To be able to respond to the questions, you will need to build the MSR, EW and GMV portfolios consisting of the “Books”, “Steel”, "Oil", and "Mines" industries. Assume the risk free rate over the 5 year period is 10%.

# In[6]:


ind = erk.get_ind_returns()
l = ["Books", "Steel", "Oil", "Mines"]
ind = ind.loc['2013-01':'2017-12',l]
rfr  = 0.1
er = erk.annualize_rets(ind,12)
cov = ind.cov()


# ### 4. What is the weight of Steel in the EW Portfolio?

# In[7]:


100/len(l)


# ### 5. What is the weight of the largest component of the MSR portfolio?

# In[8]:


wmsr = erk.msr(rfr, er, cov)
round(wmsr.max()*100,2)


# ### 6. Which of the 4 components has the largest weight in the MSR portfolio?

# In[9]:


l[erk.msr(rfr, er, cov).argmax(axis=0)]


# ### 7. How many of the components of the MSR portfolio have non-zero weights?

# In[10]:


sum(erk.msr(rfr, er, cov) > 1e-5) # Allocation of the order of e-15 is meaningless in practice


# ### 8. What is the weight of the largest component of the GMV portfolio?

# In[11]:


wgmv = erk.gmv(cov)
round(wgmv.max()*100,2)


# ### 9. Which of the 4 components has the largest weight in the GMV portfolio?

# In[12]:


l[erk.gmv(cov).argmax(axis=0)]


#  ### 10. How many of the components of the GMV portfolio have non-zero weights?

# In[13]:


sum(erk.gmv(cov) > 1e-5)


# ### Assume two different investors invested in the GMV and MSR portfolios at the start of 2018 using the weights we just computed. Compute the annualized volatility of these two portfolios over the next 12 months of 2018? (Hint: Use the portfolio_vol code we developed in the lab and use ind[“2018”][l].cov() to compute the covariance matrix for 2018, assuming that the variable ind holds the industry returns and the variable l holds the list of industry portfolios you are willing to hold. Don’t forget to annualize the volatility)
# 

# In[14]:


ind = erk.get_ind_returns()
l = ["Books", "Steel", "Oil", "Mines"]
ind2 = ind.loc['2018-01':'2018-12',l]
cov2 = ind2.cov()


# ### 11. What would be the annualized volatility over 2018 using the weights of the MSR portfolio?

# In[15]:


vmsr = erk.portfolio_vol(wmsr,cov2)
round(vmsr * 100 * (12)**0.5,2)


# ### 12. What would be the annualized volatility over 2018 using the weights of the GMV portfolio? 

# In[16]:


vgmv = erk.portfolio_vol(wgmv,cov2)
round(vgmv * 100 * (12)**0.5,2)

