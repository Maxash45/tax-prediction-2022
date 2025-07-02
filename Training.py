import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump

data = pd.read_csv("IncomeDB.csv")              #      IFS	IFHP	IFBP	OI	LIC	 PPF	MEDI	CTF	  IPH	MF	DTSC	TAX


train_set, test_set  = train_test_split(data, test_size=0.2, random_state=42)                    # i.e 20/80 devision
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['IFS']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]


data = strat_train_set.drop("TAX", axis=1)
data_labels = strat_train_set["TAX"].copy()

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

data_num_tr = my_pipeline.fit_transform(data)
print("Train_set shape :", data_num_tr.shape)

model = RandomForestRegressor()       # 3843.136806192451
model.fit(data_num_tr, data_labels)

# Saving the model

dump(model, 'tax_pred1.joblib')

