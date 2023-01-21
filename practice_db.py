import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
# SVC - Support Vector Classification

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline


data = pd.read_csv('C:/Users/anwes/Downloads/archive/diabetes.csv')
print()
print(data.head())
print(data.tail())
print("Numbers of Rows: ",data.shape[0])
print("Numbers of Columns: ",data.shape[1])
print()
x = data.drop('Outcome',axis=1)
y = data['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)
# Training 20% of the data and leaving the rest 80% for testing
print(x)
print(y)

# Feature scaling for Logistic Regression (Standard Scaler Output is input for logistic regression)
pipeline_lr = Pipeline([('scalar1',StandardScaler()),'lr_classifier',LogisticRegression()])
pipeline_knn = Pipeline([('scalar2',StandardScaler()),'lr_classifier',KNeighborsClassifier()])
pipeline_svc = Pipeline([('scalar3',StandardScaler()),'lr_classifier',SVC()])
# Feature Scaling is not required for non-machine learning algorithms like Decision Tree and Random forest
pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier())])
pipeline_rf = Pipeline([('dt_classifier', RandomForestClassifier())])
pipeline_gbc = Pipeline([('dt_classifier', GradientBoostingClassifier())])

pipelines = [pipeline_lr,pipeline_knn, pipeline_svc, pipeline_dt, pipeline_rf, pipeline_gbc]
# print(pipelines)

for pipe in pipelines:
    pipe.fit(x_train,y_train)

pipe_dict = {0:'LR', 1:'KNN', 2:'SVC', 3:'DT', 4:'RF', 5:'GBC'}

# Printing the accuracy of our models
# for i,model in enumerate(pipelines):
    # print("Test Accuracy : ".model.score(x_test,y_test)*100)

X = data.drop('Outcome',axis=1)
Y = data.drop('Outcome')
rf = RandomForestClassifier()
print(rf.fit())







