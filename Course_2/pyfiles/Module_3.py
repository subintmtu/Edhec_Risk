#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import edhec_risk_kit_206 as erk
import statsmodels.api as sm

get_ipython().run_line_magic('run', 'lab_23.ipynb')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the 49 industries Value weighted returns and cap weights, and use the period 2013-2018 both included. For the period, use the starting cap weights of the period. Limit yourself to the following 5 industry sectors: 'Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food'.
# You will need to compute the correlation matrix as well as the volatilities. (Hint: Remember to annualize the volatilities by multiplying the volatility you get from the monthly data by the sqrt iof 12)
# Using the same value of delta used in the He-Litterman paper of 2.5 and using the same sigma prior methodology used in the notebook and in the paper, compute the implied returns vector.

# In[2]:


columns = ['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food']
ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)["2013":][columns]
ind49_mcap = erk.get_ind_market_caps(n_inds=49, weights=True)["2013":][columns]
ind_w = ind49_mcap.values[0]/ind49_mcap.values[0].sum()


# In[3]:


volsM = pd.DataFrame(ind49_rets.std(),index=['Hlth', 'Fin', 'Whlsl', 'Rtail', 'Food'], columns=["vol"])
vols = volsM*np.sqrt(12)
rho = ind49_rets.corr()
VOL = vols['vol']
sigma_prior =VOL.dot(VOL.T) * rho


# 1. Which industry sector has the highest capweight?
# 2. Use the same data as the previous question, which industry sector has the highest implied return?
# 3. Which industry sector has the lowest implied return?

# In[4]:


pi = implied_returns(delta=2.5, sigma=sigma_prior, w=ind_w)
pi*100


# In[5]:


w_eq = pd.DataFrame(ind_w, index=columns, columns=["CapWeight"])


# Impose the subjective relative view that Hlth will outperform Rtail and Whlsl by 3%  (Hint: Use the same logic as View 1 in the He-Litterman paper)
#
# 4. What is the entry you will use for the Pick Matrix P for Whlsl
# 5. What is the entry you will use for the Pick Matrix P for Rtail.

# In[6]:


q = pd.Series([0.03])
p = pd.DataFrame([0.]*len(columns), index=columns).T
w_rtail =  w_eq.loc["Rtail"]/(w_eq.loc["Rtail"]+w_eq.loc["Whlsl"])
w_whlsl =  w_eq.loc["Whlsl"]/(w_eq.loc["Rtail"]+w_eq.loc["Whlsl"])
p.iloc[0]['Hlth'] = 1.
p.iloc[0]['Rtail'] = -w_rtail
p.iloc[0]['Whlsl'] = -w_whlsl
p


# Impose the subjective relative view that Hlth will outperform Rtail and Whlsl by 3%  (Hint: Use the same logic as View 1 in the He-Litterman paper)
# Once you impose this view (use delta = 2.5 and tau = 0.05 as in the paper),
#
# 6. Which sector has the lowest implied return?

# In[7]:


delta = 2.5
tau = 0.05 # from Footnote 8
# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_eq, sigma_prior, p, q, tau=0.05)
(bl_mu*100).round(2).idxmin()


# 7. Which sector now has the highest weight in the MSR portfolio using the Black-Litterman model?
# 8. Which sector now has the lowest weight in the MSR portfolio using the Black-Litterman model?

# In[8]:


w_eq  = w_msr(delta*sigma_prior, pi, scale=False)


# In[9]:


wstar = w_star(delta=2.5, sigma=bl_sigma, mu=bl_mu)
# display w*
[(wstar*100).round(1).idxmax(), (wstar*100).round(1).idxmin()]


# Now, let’s assume you change the relative view. You still think that it Hlth will outperform Rtail and Whlsl but you think that the outperformance will be 5% not the 3% you originally anticipated.
#
# 10. Under this new view which sector does the Black-Litterman model assign the highest weight?

# In[10]:


q = pd.Series([0.05])
p = pd.DataFrame([0.]*len(columns), index=columns).T
w_rtail =  w_eq.loc["Rtail"]/(w_eq.loc["Rtail"]+w_eq.loc["Whlsl"])
w_whlsl =  w_eq.loc["Whlsl"]/(w_eq.loc["Rtail"]+w_eq.loc["Whlsl"])
p.iloc[0]['Hlth'] = 1.
p.iloc[0]['Rtail'] = -w_rtail
p.iloc[0]['Whlsl'] = -w_whlsl
delta = 2.5
tau = 0.05 # from Footnote 8
# Find the Black Litterman Expected Returns
bl_mu, bl_sigma = bl(w_eq, sigma_prior, p, q, tau=0.05)
(bl_mu*100).round(2).idxmax()


# 11. Under this new view which sector has the highest expected return?

# In[11]:


w_eq  = w_msr(delta*sigma_prior, pi, scale=False)
wstar = w_star(delta=2.5, sigma=bl_sigma, mu=bl_mu)
# display w*
(wstar*100).round(1).idxmax()
