import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller

data=pd.read_excel("data12.xlsx")
DR20 = data.DR20
DRS = data.DRS

R20_diff = np.diff(DR20)
RS_diff = np.diff(DRS)
print(adfuller(R20_diff))
print(adfuller(RS_diff))
print(coint(DR20,DRS))