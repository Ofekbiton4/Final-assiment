# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 13:33:00 2023

@author: Ofek biton & Shahaf Malka
"""
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt  # Fix import statement: matplotlib as plt -> matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV
from madlan_data_prep import prepare_data

excel_file = 'output_all_students_Train_v10.xlsx'
dataframe = pd.read_excel(excel_file)
from madlan_data_prep import prepare_data
dataframe = prepare_data(dataframe)

drop_columns = ['price']
category_columns = ['City', 'type']
numeric_columns = ['room_number', 'Area']

X = dataframe.drop(drop_columns, axis=1)
y = dataframe['price']

encoder = OneHotEncoder()
scaler = MinMaxScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', encoder, category_columns),
        ('scaler', scaler, numeric_columns)
    ], remainder='passthrough')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_scaled = preprocessor.fit_transform(X_train)
feature_names = list(preprocessor.get_feature_names_out())
feature_names = [x.split('_')[-1] for x in feature_names]
X_train_scaled = pd.DataFrame(X_train_scaled.toarray(), columns=feature_names)
model = ElasticNet(alpha=1, l1_ratio=0.87, random_state=42)
cv = KFold(n_splits=10)
scores = cross_val_score(estimator=model, X=X_train_scaled, y=y_train, scoring='neg_mean_squared_error', cv=cv)
print(scores.mean(), scores)

X_test_scaled = preprocessor.transform(X_test)
X_test_scaled = pd.DataFrame(X_test_scaled.toarray(), columns=feature_names)

model.fit(X_train_scaled, y_train)

y_pred = cross_val_predict(model, X_test_scaled, y_test, cv=10)
mse = mean_squared_error(y_test, y_pred)
print('The RMSE: ' + str(np.sqrt(mse)))  


importance = permutation_importance(model, X_train_scaled, y_train, n_repeats=10, random_state=42)
feature_importances = importance.importances_mean
importance_dict = dict(zip(feature_names, feature_importances))
sorted_importances = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importances:
    print(f"{feature}: {importance}")

import pickle
pickle.dump(model, open("trained_model.pkl", "wb"))
    
