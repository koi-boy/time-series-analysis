import pandas as pd
import statsmodels.api as sm

data = pd.read_excel("data12.xlsx")
R20 = data.R20
RS = data.RS

X = sm.add_constant(RS)#给自变量中加入常数项
model1 = sm.OLS(R20,X).fit()
print(model1.summary())
print(model1.params)

Y = sm.add_constant(R20)
model2 = sm.OLS(RS,Y).fit()
print(model2.summary())
print(model2.params)