#!/usr/bin/env python
# coding: utf-8

# ## Module 1

# In[1]:


import pandas as pd
import numpy as np
import edhec_risk_kit_206 as erk
import statsmodels.api as sm

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('matplotlib', 'inline')


# Load the 49 Value Weighted industry portfolio returns. Limit the analysis to returns of the 49 VW portfolios from 1991 onwards (1991 included, 1990 not included, 2018 included). Also load the Fama French Research Factors over the same period. Use the Mkt-RF as in the lab notebooks to compute the CAPM betas.

# In[2]:


ind49_rets = erk.get_ind_returns(weighting="vw", n_inds=49)
fff = erk.get_fff_returns()


# #### 1. What is the CAPM (Single Factor) Beta when evaluated over the entire period (1991-2018) of Beer?

# In[3]:


ind49f_excess = ind49_rets["1991":"2018"] - fff.loc["1991":"2018", ['RF']].values
mktf_excess = fff.loc["1991":"2018",['Mkt-RF']]
expf_var = mktf_excess.copy()
expf_var["Constant"] = 1
lm = sm.OLS(ind49f_excess["Beer"], expf_var).fit()
lm.summary()
lm.params["Mkt-RF"]


# #### 2. Using the same data as the previous question, what is the CAPM Beta when evaluated over the entire period (1991-2018) of Steel?

# In[4]:


ind49f_excess = ind49_rets["1991":"2018"] - fff.loc["1991":"2018", ['RF']].values
mktf_excess = fff.loc["1991":"2018",['Mkt-RF']]
expf_var = mktf_excess.copy()
expf_var["Constant"] = 1
lm = sm.OLS(ind49f_excess["Steel"], expf_var).fit()
lm.summary()
lm.params["Mkt-RF"]


# #### 3. Using the same data as the previous question, what is the CAPM Beta when evaluated over the 2013-2018 (both included) period of Beer?

# In[5]:


ind49f_excess = ind49_rets["2013":"2018"] - fff.loc["2013":"2018", ['RF']].values
mktf_excess = fff.loc["2013":"2018",['Mkt-RF']]
expf_var = mktf_excess.copy()
expf_var["Constant"] = 1
lm = sm.OLS(ind49f_excess["Beer"], expf_var).fit()
lm.summary()
lm.params["Mkt-RF"]


# #### 4. Using the same data set as the previous question, what is the CAPM Beta when evaluated over the 2013-2018 (both included) period of Steel?

# In[6]:


ind49f_excess = ind49_rets["2013":"2018"] - fff.loc["2013":"2018", ['RF']].values
mktf_excess = fff.loc["2013":"2018",['Mkt-RF']]
expf_var = mktf_excess.copy()
expf_var["Constant"] = 1
lm = sm.OLS(ind49f_excess["Steel"], expf_var).fit()
lm.summary()
lm.params["Mkt-RF"]


# In[7]:


ind49f_excess = ind49_rets["1991":"1993"] - fff.loc["1991":"1993", ['RF']].values
mktf_excess = fff.loc["1991":"1993",['Mkt-RF']]

xdict = {}
for industry in ind49f_excess.columns:
    expf_var = mktf_excess.copy()
    expf_var["Constant"] = 1
    lm = sm.OLS(ind49f_excess[industry], expf_var).fit()
    xdict[industry] = lm.params["Mkt-RF"]


# #### 5. Using the same data as the previous question, which of the 49 industries had the highest CAPM Beta when evaluated over the 1991-1993 (both included) period? (Use the same industry names as in the files). Enter the name as a text string, and remember to exactly match the column headers in the data file.
# 

# In[8]:


max(xdict, key=xdict.get)


# #### 6. Using the same data as the previous question, which of the 49 industries had the lowest CAPM Beta when evaluated over the 1991-1993 (both included) period? (Use the same industry names as in the files) . Enter the answer as text and remember to exactly match the column headers in the data file.

# In[9]:


min(xdict, key=xdict.get)


# For the next 4 questions use the Full 3 Factor Fama-French model using the research data supplied in the following data file:
# F-F_Research_Data_Factors.csv
# and the same 1991-2018 period you have just used for the previous questions.

# In[10]:


ff_BI = pd.read_csv("data/F-F_Research_Data_Factors.CSV", parse_dates=True, index_col=0)
ff_BI.index = pd.to_datetime(ff_BI.index, format="%Y%m").to_period('M')


# In[11]:


import pandas as pd
import numpy as np
ind49_excess1 = ind49_rets["1991":"2018"] - fff.loc["1991":"2018", ['RF']].values
mkt_excess1 = fff.loc["1991":"2018",['Mkt-RF']]
exp_var1 = mkt_excess1.copy()
exp_var1["Constant"] = 1
exp_var1["Value"] = fff.loc["1991":"2018",['HML']]
exp_var1["Size"] = fff.loc["1991":"2018",['SMB']]
results = pd.DataFrame(index = ["beta","value","size"], columns =ind49_rets.columns)
for index in ind49_excess1:
    industry_name=index
    lm=sm.OLS(ind49_excess1[industry_name], exp_var1).fit()
    results[index]=(lm.params[0], lm.params[2], lm.params[3])
    
results


# #### 7. Of the 49 industries, which displayed the highest Small Cap tilt when analyzed over the entire 1991-2018 period?
# #### 8. Using the same dataset and period as the previous question, of the 49 industries, which displayed the highest Large Cap tilt when analyzed over the entire period?
# #### 9. Using the same data as period as the previous question, of the 49 industries, which displayed the highest Value tilt when analyzed over the entire period?
# #### 10. Using the same data set and period as the previous question, of the 49 industries, which displayed the highest Growth tilt when analyzed over the entire period?

# In[12]:


results.idxmax(axis=1)


# In[13]:


results.idxmin(axis=1)


# #### 7. FabPr
# #### 8. Beer
# #### 9. Txtls
# #### 10. Softw
