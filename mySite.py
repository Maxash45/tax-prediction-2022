# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request,session,Response
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from flask_ngrok import run_with_ngrok
import csv

data = pd.read_csv("IncomeDB.csv")          #      IFS	IFHP	IFBP	OI	LIC	PPF	MEDI	CTF	IPH	MF	DTSC	TAX

train_set, test_set  = train_test_split(data, test_size=0.2, random_state=42)                    # i.e 20/80 devision

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

model = RandomForestRegressor()       # 3843.136806192451
model.fit(data_num_tr, data_labels)


model = load('tax_pred.joblib')

app = Flask(__name__)
run_with_ngrok(app)

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('home'))
    return render_template('login.html', error=error)

@app.route('/home', methods=['GET', 'POST'])
def home():
    msg = None
    if request.method == 'POST':
        if request.form['submit_button'] == 'Calculate Tax':
            FSI = int(request.form['para2'])
            IFHP = int(request.form['para3'])
            IFBP = int(request.form['para4'])
            OI = int(request.form['para5'])
            #Deductions
            LIC = int(request.form['para6'])
            PPF = int(request.form['para7'])
            MEDI = int(request.form['para8'])
            CTF = int(request.form['para9'])
            IPO = int(request.form['para10'])
            MF = int(request.form['para11'])
            DTSC = int(request.form['para12'])
            PTAX = int(request.form['para13'])
            taxable_amount = (int(FSI) + int(IFHP) + int(IFBP) + int(OI)) - (int(LIC) + int(PPF) + int(MEDI) + int(CTF) + int(IPO) + int(MF) + int(DTSC))

            features = [[FSI, IFHP,  IFBP,  OI,    LIC,    PPF,   MEDI,  CTF,  IPO,  MF,  DTSC ]]                  #, int(IFBP), int(OI), (int(LIC), int(PPF), int(MEDI), int(CTF), int(IPH), int(MF), int(DTSC)]])
            prepared_data = my_pipeline.transform(features)
            tax_amount = model.predict(list(prepared_data))

            with open('Newdata.csv', 'a') as f:
                wr = csv.writer(f, dialect='excel', delimiter=',', lineterminator="\n")
                wr.writerow([FSI, IFHP,  IFBP,  OI,    LIC,    PPF,   MEDI,  CTF,  IPO,  MF,  DTSC, PTAX])
            f.close()

            DIFF = tax_amount[0] - PTAX
            print(DIFF)
            if(DIFF > 2000 ):
                msg = "Detected Fraud of " + str(DIFF) + ' rs'
            else:
                msg = ' '
            return render_template('input.html', msg = msg, tax_amount = tax_amount[0], taxable_amount = taxable_amount )

    return render_template('input.html')


if __name__ == '__main__':
    app.run()

'''   0 - 3L   0 % 
    3L – 5L   10 % 
    5L – 10L  20 % 
    Above
    10L       30 %
'''