from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd
df = pd.read_excel('data12.xlsx', usecols=[1,2])
grangercausalitytests(df, maxlag=3)