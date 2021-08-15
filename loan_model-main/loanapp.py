#!/usr/bin/env python
# coding: utf-8


#import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LogisticRegression
#from sklearn import tree
#from sklearn.metrics import classification_report
#from sklearn.ensemble import GradientBoostingClassifier
#from xgboost import XGBClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
import numpy as np
from pickle import load, dump
#import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
#from flask_compress import Compress
import os




'''

loan_amnt, emp_length, annual_inc, delinq_2yrs, inq_last_6mths, mths_since_last_delinq, mths_since_last_record, open_acc, pub_rec, revol_bal, revol_util, total_acc, purpose, year   

'''


'''
print(debt_consolidation)#2
print(car)#1
print(credit_card)#0
print(educational)#2
print(home_improvement)#1
print(house)#1
print(major_purchase)#0
print(medical)#2
print(moving)#2
print(other)#2
print(renewable_energy)#2
print(small_business)#2
print(vacation)#1
print(wedding)#0

'''


# scikit-learn==0.22.2.post1


model_ = load(open('model___.pkl', 'rb'))
scaler_ = load(open('scaler.pkl', 'rb'))
pca_ = load(open('pca.pkl', 'rb'))



app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'



@app.route('/predict',methods=['POST'])
@cross_origin()
def results():

    data = request.get_json(force=True)
    X_one_ = np.array(list(data.values())).reshape(1, -1)
    X_one_ = scaler_.transform(X_one_)
    X_one_ = pca_.transform(X_one_)
    prediction = model_.predict_proba(X_one_)
    prediction = np.float64(prediction)
    print(prediction)
    #prediction = model_.predict([np.array(list(data.values()))])

    output = {'score':np.int(350+550*prediction[0][1])}
    return(jsonify(output))


if __name__ == "__main__":
    port = int(os.environ.get('PORT',5000))
    app.run(host='0.0.0.0',port=port)





