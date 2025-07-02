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

print(data.head())
print(data.info())

#print(data['IFS'].value_counts())
#print(data.describe())

## Train-Test Splitting
# For learning purpose

train_set, test_set  = train_test_split(data, test_size=0.2, random_state=42)                    # i.e 20/80 devision
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data['IFS']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

print(strat_test_set['IFS'].value_counts())

print(strat_train_set['IFS'].value_counts())

corr_matrix = data.corr()
corr_matrix['IFS'].sort_values(ascending=False)

data = strat_train_set.drop("TAX", axis=1)
data_labels = strat_train_set["TAX"].copy()


## Scikit-learn Design

# Primarily, three types of objects

# 1. Estimators - It estimates some parameter based on a dataset. Eg. imputer. It has a fit method and transform method.
# Fit method - Fits the dataset and calculates internal parameters
# 2. Transformers - transform method takes input and returns output based on the learnings from fit(). It also has a
# convenience function called fit_transform() which fits and then transforms.
# 3. Predictors - LinearRegression model is an example of predictor. fit() and predict() are two common functions.
# It also gives score() function which will evaluate the predictions.

## Feature Scaling

# Primarily, two types of feature scaling methods:
# 1.Min - max scaling(Normalization) (value - min) / (max - min) Sklearn provides a class called MinMaxScaler for this
# 2. Standardization (value - mean) / std Sklearn provides a
# class called StandardScaler for this

## Creating a Pipeline

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ..... add as many as you want in your pipeline
    ('std_scaler', StandardScaler()),
])

data_num_tr = my_pipeline.fit_transform(data)
print("Train_set shape :", data_num_tr.shape)

## Selecting a desired model

#model = LinearRegression()          # RMSE = 16967.959000185583
#model = DecisionTreeRegressor()      # 0.0
model = RandomForestRegressor()       # 3843.136806192451
model.fit(data_num_tr, data_labels)

some_data = data.iloc[:5]
print("some_data ", some_data)
some_labels = data_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)

prediction = model.predict(prepared_data)
print("model prediction for 5 : ", prediction)
print("Actual Values : ", some_labels)

## Evaluating the model
data_predictions = model.predict(data_num_tr)
mse = mean_squared_error(data_labels, data_predictions)
rmse = np.sqrt(mse)

print("rmse :", rmse )

## Using better evaluation technique - Cross Validation
# 1 2 3 4 5 6 7 8 9 10

scores = cross_val_score(model, data_num_tr, data_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def print_scores(scores):
    print("Scores:", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

print_scores(rmse_scores)

## Saving the model

dump(model, 'tax_pred.joblib')

## Testing the model on test data

X_test = strat_test_set.drop("TAX", axis=1)
Y_test = strat_test_set["TAX"].copy()

X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("final predictions on test set", final_predictions)
print("Actual values ", list(Y_test))
print('final_rmse ', final_rmse)

print("some_data ", some_data)
print(prepared_data[0])




