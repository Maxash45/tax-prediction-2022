## Using the model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump, load

data = pd.read_csv("IncomeDB.csv")          #      IFS	IFHP	IFBP	OI	LIC	PPF	MEDI	CTF	IPH	MF	DTSC	TAX

train_set, test_set  = train_test_split(data, test_size=0.2, random_state=42)                    # i.e 20/80 devision

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['IFS']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]


corr_matrix = data.corr()
corr_matrix['IFS'].sort_values(ascending=False)

data = strat_train_set.drop("TAX", axis=1)
data_labels = strat_train_set["TAX"].copy()


my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

data_num_tr = my_pipeline.fit_transform(data)

## Selecting a desired model

#model = LinearRegression()          # RMSE = 16967.959000185583
#model = DecisionTreeRegressor()      # 0.0
model = RandomForestRegressor()       # 3843.136806192451
model.fit(data_num_tr, data_labels)

'''
v1 = int(input('Enter Income From Salary:'))
v2 = int(input('Income from house property :'))
v3 = int(input('Income From Business/ Profession :'))
v4 = int(input("Other - income from savings bank accounts,fixed deposit's interest,family pensions or gifts received. :"))
v5 = int(input('LIC policy  :'))
v6 = int(input('PPF/EPF/UPLFs/NLP/ELSS :'))
v7 = int(input('Mediclaim :'))
v8 = int(input('Childern tution fees :'))
v9 = int(input('Interest paid on mortgages, student loans, Home loan and business loans. :'))
v10 = int(input('Mutual Funds :'))
v11 = int(input('Any kind of Donations towards Social Causes. :'))
x = [v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11]'''
x = [[240000, 0,0,0,0,0,0,0,0,0,0]]

#print("data ", x)
prepared_data = my_pipeline.transform(x)
#print(list(prepared_data))
pred = model.predict(list(prepared_data))

print("Tax Amount ",pred[0])