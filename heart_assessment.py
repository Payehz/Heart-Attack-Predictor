# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:21:41 2022

@author: User
"""

import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay



#%% Functions

def cramers(confusionmatrix):
    chi2 = ss.chi2_contingency(confusionmatrix)[0]
    n = confusionmatrix.sum()
    phi2 = chi2/n
    r,k = confusionmatrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2/(n-1))
    kcorr = k - ((k-1)**2/(n-1))
    return np.sqrt(phi2corr / min((kcorr-1),(rcorr-1)))


#%% Statics

CSV_PATH = os.path.join(os.getcwd(),'heart.csv')
BEST_PIPE_PATH = os.path.join(os.getcwd(),'best_pipe.pkl')
BEST_MODEL_PATH = os.path.join(os.getcwd(),'best_model.pkl')

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%% Step 2) Data Inspection

df.head()
df.tail()

df.info()
df.describe().T

df.boxplot()

df.isna().sum() # No NaN values

df.duplicated().sum() # 1 duplicated data

categorical = ['sex','cp','fbs','restecg','exng','slp','caa','thall','output']

continuous = ['age','trtbps','chol','thalachh','oldpeak']

# To visualise data
# Categorical
for i in categorical:
    plt.figure()
    sns.countplot(df[i])
    plt.show()

# Continuous
for j in continuous:
    plt.figure()
    sns.distplot(df[j])
    plt.show()

#%% Step 3) Data Cleaning

# Drop duplicated data

df = df.drop_duplicates()

# No NaN to be imputed

#%% Step 4) Features Selection

# Categorical vs categorical
# Use Cramer's V

for i in categorical:
    confusionmatrix = pd.crosstab(df[i], df['output']).to_numpy()
    print(i + ": " + str(cramers(confusionmatrix)))

# Since cp and thall has a correlation more than 0.5,
# those will be selected as our features.


# Continuous vs categorical
# Use logistic regression

for i in continuous:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[i],axis=-1),df['output'])
    print(i + ": " + str(lr.score(np.expand_dims(df[i],axis=-1),df['output'])))

# We select features that has more than 50% accuracy,
# hence age, trtbps, chol, thalachh and oldpeak will be selected.


#%% Step 5) Data Preprocessing

X = df.loc[:,['age','cp','trtbps','chol','thalachh','oldpeak','thall']]
y = df.loc[:,'output']

# Scalling is done in the pipeline

# No need to do label encoder as the data is already in integers

# Train test split
X_train,X_test,y_train,y_test = train_test_split(X, y,
                                                 test_size=0.3,
                                                 random_state=123)

#%% Machine learning development

# Using pipeline to determine which scalling and ML model gives the best accuracy

step_mms_lr = Pipeline([('MinMax', MinMaxScaler()),
            ('lr', LogisticRegression())])

step_ss_lr = Pipeline([('SS', StandardScaler()),
            ('lr', LogisticRegression())])

step_mms_rf = Pipeline([('MinMax', MinMaxScaler()),
            ('rf', RandomForestClassifier())])

step_ss_rf = Pipeline([('SS', StandardScaler()),
            ('rf', RandomForestClassifier())])

step_mms_tree = Pipeline([('MinMax', MinMaxScaler()),
            ('tree', DecisionTreeClassifier())])

step_ss_tree = Pipeline([('SS', StandardScaler()),
            ('tree', DecisionTreeClassifier())])

step_mms_knn = Pipeline([('MinMax', MinMaxScaler()),
            ('knn', KNeighborsClassifier())])

step_ss_knn = Pipeline([('SS', StandardScaler()),
            ('knn', KNeighborsClassifier())])

step_mms_svc = Pipeline([('MinMax', MinMaxScaler()),
            ('svc', SVC())])

step_ss_svc = Pipeline([('SS', StandardScaler()),
            ('svc', SVC())])

pipelines = [step_mms_lr, step_ss_lr, step_mms_rf, step_ss_rf,
             step_mms_tree, step_ss_tree, step_mms_knn, step_ss_knn, 
             step_mms_svc, step_ss_svc]

for pipe in pipelines:
    pipe.fit(X_train, y_train)
    
best_accuracy = 0

for i, model in enumerate(pipelines):
    print(model.score(X_test,y_test))
    if model.score(X_test,y_test) > best_accuracy:
        best_accuracy = model.score(X_test,y_test)
        best_pipeline = model

print(best_pipeline)
print(best_accuracy)

# The best pipeline is MMS + LogisticRegression

#%% Fine tuning the model

step_mms_lr = [('MinMax', MinMaxScaler()),
            ('Logistic', LogisticRegression())]

pipeline_logistic = Pipeline(step_mms_lr)

grid_param = [{'Logistic':[LogisticRegression()],
                'Logistic__penalty':['l1','l2'],
                'Logistic__C': np.logspace(-4,4,20),
                'Logistic__solver':['liblinear']}]

gridsearch = GridSearchCV(pipeline_logistic,grid_param,cv=5,verbose=1,n_jobs=-1)
best_model = gridsearch.fit(X_train,y_train)
print(best_model.score(X_test, y_test))
print(best_model.best_index_)
print(best_model.best_params_)

# Best parameter is Penalty = 'l1', C = 0.615848211066026, solver = 'liblinear'

with open(BEST_PIPE_PATH,'wb') as file:
    pickle.dump(best_model,file)

#%% Retrain model

# Retrain the model to save the best model with the respective parameters

step_mms_lr = Pipeline([('MinMax', MinMaxScaler()),
            ('Logistic', LogisticRegression(penalty='l1',C=0.615848211066026,
                                            solver='liblinear'))])

step_mms_lr.fit(X_train,y_train)

with open(BEST_MODEL_PATH,'wb') as file:
    pickle.dump(step_mms_lr,file)

#%% Model analysis

y_true = y_test
y_pred = best_model.predict(X_test)

cm = confusion_matrix(y_true,y_pred)
cr = classification_report(y_true,y_pred)
acc_score = accuracy_score(y_true,y_pred)
print(cm)
print(cr)
print(acc_score)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

#%% Discussion

# The model is able to predict whether a person has a heart attack or not
# with an accuracy of 78%.
# The best pipeline used was MinMaxScaler with LogisticRegression
# During the finetuning, we found out that penalty='l1', C = 0.615848211066026
# and solver = 'liblinear' is the best parameters for the highest accuracy.
# The accuracy can be improved if we have more data, since we only deal with 300.









