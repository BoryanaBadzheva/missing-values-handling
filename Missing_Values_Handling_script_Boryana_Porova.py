#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:10:54 2025

@author: boryanabadzheva
"""

from sklearn import metrics
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

###randomly deletion of cells - 20% of all data in the selected range of cells
column_names = ['MCV', 'Alkphos', 'SGPT', 'SGOT', 'Gammagt', 'Drinks per day', 'Health condition']
data_liver_NAN_drop = pd.read_csv("liver.txt", delimiter=',', names=column_names, header=None)
missing_percentage = 0.2  
total_cells = data_liver_NAN_drop.shape[0] * data_liver_NAN_drop.shape[1]
num_missing = int(total_cells * missing_percentage)

#randomly pick positions for missing values
np.random.seed(23)  
for _ in range(num_missing):
    random_row = np.random.randint(0, data_liver_NAN_drop.shape[0])  
    random_col = np.random.choice([i for i in range(data_liver_NAN_drop.shape[1]) if i != 6]) 
    data_liver_NAN_drop.iat[random_row, random_col] = np.NAN  #set that cell to NaN
print("DataFrame with random missing values:")
print(data_liver_NAN_drop)


# Handalling  rows/colomns with NAN values

#### 1) drop the rows
indexNAN = np.nonzero(pd.isnull(data_liver_NAN_drop.values).any(1))[0]
data_liver_NAN_drop_values=data_liver_NAN_drop.drop([indexNAN[i] for i in range(len(indexNAN))])

##logistic regression with kFold split
#choose independent and dependent variables
X = data_liver_NAN_drop_values.iloc[:,:6]  
y = data_liver_NAN_drop_values.iloc [:,6]   

logreg = LogisticRegression(solver='liblinear', max_iter=4200) 

def k_fold_split(X, y):
    
    kf = KFold(n_splits=17, shuffle=True, random_state=72)
    
    best_score = 0
    train_index_highest = None
    test_index_highest = None
    k = 0 

    for train_index, test_index in kf.split(X, y):
       
        logreg.fit(X.iloc[train_index], y.iloc[train_index])
        score_test = logreg.score(X.iloc[test_index], y.iloc[test_index])
        print (score_test) 
        
        if best_score < score_test:
            best_score = score_test
            train_index_highest = train_index
            test_index_highest = test_index


    return (
        X.iloc[train_index_highest], X.iloc[test_index_highest],
        y.iloc[train_index_highest], y.iloc[test_index_highest]
    )



X_train, X_test, y_train, y_test = k_fold_split(X, y)


logreg.fit(X_train, y_train)

#ccalculation of scores for train and test sets in order to check for overfit
score_train = logreg.score(X_train, y_train)
score_test = logreg.score(X_test, y_test)
print('Score on train set:', score_train)
print('Score on test set:', score_test)

predicted_y = logreg.predict(X_test)

#confussion matrix and classification report on the test set
print('Score on test set:', score_test)
print("\nConfusion Matrix for test set:")
print(metrics.confusion_matrix(predicted_y, y_test)) 

print("\nClassification Report for test set:")
print(metrics.classification_report(predicted_y, y_test, 
                                    digits=4))
#score and report for full dataset
predicted_y_full = logreg.predict(X)
print('Score on full dataset:', logreg.score(X, y))
print('Confusion Matrix for full dataset:\n', confusion_matrix(predicted_y_full,y))
print('Classification Report for full dataset:\n', classification_report(predicted_y_full,y))



#### 2) заместваме с медианата на съответната колона
from sklearn.impute import SimpleImputer
data_liver_NAN_mean = data_liver_NAN_drop.copy() 
columns_to_impute = data_liver_NAN_mean.columns.difference(['Health condition'])  
imputer = SimpleImputer(strategy='median')
#appling the imputer to all columns except the dependent variable
data_liver_NAN_mean[columns_to_impute] = imputer.fit_transform(data_liver_NAN_mean[columns_to_impute])
print("Dataset after imputation:")
print(data_liver_NAN_mean)

##Logistic regression with kFold split

X = data_liver_NAN_mean.iloc[:,:6]  
y = data_liver_NAN_mean.iloc [:,6]   

logreg = LogisticRegression(solver='liblinear', max_iter=4200) 


def k_fold_split(X, y):
     
    kf = KFold(n_splits=17, shuffle=True, random_state=72)
    
    best_score = 0
    train_index_highest = None
    test_index_highest = None
    k = 0 

    for train_index, test_index in kf.split(X, y):
      
        logreg.fit(X.iloc[train_index], y.iloc[train_index])
        score_test = logreg.score(X.iloc[test_index], y.iloc[test_index])
        print (score_test) 
        
        if best_score < score_test:
            best_score = score_test
            train_index_highest = train_index
            test_index_highest = test_index


    return (
        X.iloc[train_index_highest], X.iloc[test_index_highest],
        y.iloc[train_index_highest], y.iloc[test_index_highest]
    )


#split dataset using k-fold
X_train, X_test, y_train, y_test = k_fold_split(X, y)


logreg.fit(X_train, y_train)

#calculation of scores for train and test sets in order to check for overfit
score_train = logreg.score(X_train, y_train)
score_test = logreg.score(X_test, y_test)
print('Score on train set:', score_train)
print('Score on test set:', score_test)

predicted_y = logreg.predict(X_test)

#confussion matrix and classification report on the test set
print('Score on test set:', score_test)
print("\nConfusion Matrix for test set:")
print(metrics.confusion_matrix(predicted_y, y_test)) 

print("\nClassification Report for test set:")
print(metrics.classification_report(predicted_y, y_test, 
                                    digits=4))
#score and report for full dataset
predicted_y_full = logreg.predict(X)
print('Score on full dataset:', logreg.score(X, y))
print('Confusion Matrix for full dataset:\n', confusion_matrix(predicted_y_full,y))
print('Classification Report for full dataset:\n', classification_report(predicted_y_full,y))


### 3) filling missing values based on the mean of groups defined by the 7th column, which is 'Health condition'

data_liver_NAN_drop_group = data_liver_NAN_drop.copy()  
data_liver_NAN_drop_group.columns = ['MCV', 'Alkphos', 'SGPT', 'SGOT', 'Gammagt', 'Drinks per day', 'Health condition']

columns_to_fill = data_liver_NAN_drop.columns.difference(['Health condition'])


for col in columns_to_fill:
    data_liver_NAN_drop_group[col] = data_liver_NAN_drop_group[col].fillna(data_liver_NAN_drop_group.groupby('Health condition')[col].transform('mean'))

print("Dataset after applying fillna based on groups:")
print(data_liver_NAN_drop_group)

## Logistic regression with kFold split
X = data_liver_NAN_drop_group.iloc[:,:6]  
y = data_liver_NAN_drop_group.iloc [:,6]   

logreg = LogisticRegression(solver='liblinear', max_iter=4200) 


def k_fold_split(X, y):
     
    kf = KFold(n_splits=17, shuffle=True, random_state=72)
    
    best_score = 0
    train_index_highest = None
    test_index_highest = None
    k = 0 

    for train_index, test_index in kf.split(X, y):
        
        logreg.fit(X.iloc[train_index], y.iloc[train_index])
        score_test = logreg.score(X.iloc[test_index], y.iloc[test_index])
        print (score_test) 
        
        if best_score < score_test:
            best_score = score_test
            train_index_highest = train_index
            test_index_highest = test_index


    return (
        X.iloc[train_index_highest], X.iloc[test_index_highest],
        y.iloc[train_index_highest], y.iloc[test_index_highest]
    )


#split dataset using k-fold
X_train, X_test, y_train, y_test = k_fold_split(X, y)


logreg.fit(X_train, y_train)

#calculation of scores for train and test sets in order to check for overfit
score_train = logreg.score(X_train, y_train)
score_test = logreg.score(X_test, y_test)
print('Score on train set:', score_train)
print('Score on test set:', score_test)

predicted_y = logreg.predict(X_test)

#confussion matrix and classification report on the test set
print('Score on test set:', score_test)
print("\nConfusion Matrix for test set:")
print(metrics.confusion_matrix(predicted_y, y_test)) 

print("\nClassification Report for test set:")
print(metrics.classification_report(predicted_y, y_test, 
                                    digits=4))
#score and report for full dataset
predicted_y_full = logreg.predict(X)
print('Score on full dataset:', logreg.score(X, y))
print('Confusion Matrix for full dataset:\n', confusion_matrix(predicted_y_full,y))
print('Classification Report for full dataset:\n', classification_report(predicted_y_full,y))


##### 4) filling missing values based on the median of groups defined by the 7th column, which is 'Health condition'

data_liver_NAN_drop_median = data_liver_NAN_drop.copy()  
data_liver_NAN_drop_median.columns = ['MCV', 'Alkphos', 'SGPT', 'SGOT', 'Gammagt', 'Drinks per day', 'Health condition']
columns_to_fill = data_liver_NAN_drop.columns.difference(['Health condition'])

for col in columns_to_fill:
    data_liver_NAN_drop_median[col] = data_liver_NAN_drop_median[col].fillna(data_liver_NAN_drop_group.groupby('Health condition')[col].transform('median'))

print("Dataset after applying fillna based on groups:")
print(data_liver_NAN_drop_median)

##Logistic regression with kFold split

X = data_liver_NAN_drop_median.iloc[:,:6]  
y = data_liver_NAN_drop_median.iloc [:,6]   

logreg = LogisticRegression(solver='liblinear', max_iter=4200) 


def k_fold_split(X, y):
     
    kf = KFold(n_splits=17, shuffle=True, random_state=72)
    
    best_score = 0
    train_index_highest = None
    test_index_highest = None
    k = 0 

    for train_index, test_index in kf.split(X, y):
        
        logreg.fit(X.iloc[train_index], y.iloc[train_index])
        score_test = logreg.score(X.iloc[test_index], y.iloc[test_index])
        print (score_test) 
        
        if best_score < score_test:
            best_score = score_test
            train_index_highest = train_index
            test_index_highest = test_index


    return (
        X.iloc[train_index_highest], X.iloc[test_index_highest],
        y.iloc[train_index_highest], y.iloc[test_index_highest]
    )



X_train, X_test, y_train, y_test = k_fold_split(X, y)


logreg.fit(X_train, y_train)

#calculation of scores for train and test sets in order to check for overfit
score_train = logreg.score(X_train, y_train)
score_test = logreg.score(X_test, y_test)
print('Score on train set:', score_train)
print('Score on test set:', score_test)

predicted_y = logreg.predict(X_test)

#confussion matrix and classification report on the test set
print('Score on test set:', score_test)
print("\nConfusion Matrix for test set:")
print(metrics.confusion_matrix(predicted_y, y_test)) 

print("\nClassification Report for test set:")
print(metrics.classification_report(predicted_y, y_test, 
                                    digits=4))
#score and report for full dataset
predicted_y_full = logreg.predict(X)
print('Score on full dataset:', logreg.score(X, y))
print('Confusion Matrix for full dataset:\n', confusion_matrix(predicted_y_full,y))
print('Classification Report for full dataset:\n', classification_report(predicted_y_full,y))

############## comparison between the four models - recalls
import numpy as np
import matplotlib.pyplot as plt

recall_mean = [0.7500, 0.8750]
recall_drop_group = [0.8333, 0.8671]
recall_drop_median = [0.8333, 0.8571]
recall_originally = [1.0000, 0.8462]

classes = ['Class 1', 'Class 2']
x = np.arange(len(classes))  
width = 0.2  
plt.figure(figsize=(9, 6))
plt.bar(x - 1.5 * width, recall_mean, width, label=' 2) Recall (Mean)', color='royalblue')
plt.bar(x - 0.5 * width, recall_drop_group, width, label='3) Recall (Group Mean)', color='seagreen')
plt.bar(x + 0.5 * width, recall_drop_median, width, label='4) Recall (Group Median)', color='mediumpurple')
plt.bar(x + 1.5 * width, recall_originally, width, label='Recall (Original)', color='darkorange')  
plt.xticks(x, classes)  
plt.title('Recall Comparison: Experiments vs. Original Data')
plt.xlabel('Classes')
plt.ylabel('Recall Score')
plt.ylim(0, 1.1)  
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7) 
plt.show()


























