# -*- coding: utf-8 -*-
"""
Created on Wed Nov 07 08:09:41 2018

@author: patkarm
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

#loaded the dataset

titanic = pd.read_csv("train.csv")
print(titanic.shape)#tells about rows*column of the dataset


#preprocessing the data as it contains a lot of strings and we can't process those
# there are 3 stations , we have created 3 more columns and gave a binary representations , 0 if not arrived at that station, 1 if arrived at that station
port = pd.get_dummies(titanic.Embarked, prefix='Embarked')

titanic = titanic.join(port)
titanic.drop(['Embarked'], axis=1, inplace=True)
#print(titanic)

def clean_cabin(x):
  try:
    return x[0]
  except TypeError:
    return "N"

#titanic.Sex = titanic.Sex.map({'male':0}, 'female':1})
titanic.Sex = titanic.Sex.map({'male':0, 'female':1})


#print(titanic['Sex'])


y = titanic.Survived.copy()
#print(y)

#in = titanic.drop(['Survived'], axis=1)
X = titanic.drop(['Survived'], axis=1)
#print(X)

#droping not so important  items

#X.drop(['Cabin'], axis =1, inplace =True)
X.drop(['Ticket'] , axis =1 , inplace=True)
X.drop(['PassengerId'], axis =1 , inplace =True)
X.drop(['Name'], axis =1 , inplace=True)
#checking summary of columns
#X.info()
#age has only 714 entries, rest should be null
#check if any missing values

#print(X.isnull().values.any())

#true 


X.Age.fillna(X.Age.mean(), inplace=True)
print(X.isnull().values.any())
#false, no NaN values , good to go


X['Cabin'] = X.Cabin.apply(clean_cabin)
port1 = pd.get_dummies(X.Cabin, prefix='Cabin')

X = X.join(port1)
X.drop(['Cabin'], axis=1, inplace=True)


X.loc[X['Age']<= 10, 'Age']= 0
X.loc[(X['Age'] > 10 ) & (X['Age']<= 20), 'Age']= 1
X.loc[(X['Age'] > 20 ) & (X['Age']<= 30), 'Age']= 2
X.loc[(X['Age'] > 30 ) & (X['Age']<= 40), 'Age']= 3
X.loc[(X['Age'] > 40 ) & (X['Age']<= 50), 'Age']= 4
X.loc[(X['Age'] > 50 ) & (X['Age']<= 60), 'Age']= 5
X.loc[(X['Age'] > 60 ) & (X['Age']<= 70), 'Age']= 6


print ("max fare " , X['Fare'].max())
print("min fare ", X['Fare'].min())
print(" mean fare", X['Fare'].mean())
print("std deviation")


X.loc[X['Fare']<= 7, 'Fare']= 0
X.loc[(X['Fare'] > 7 ) & (X['Fare']<= 14), 'Fare']= 1
X.loc[(X['Fare'] > 14 ) & (X['Fare']<= 25), 'Fare']= 2
X.loc[(X['Fare'] > 25 ) & (X['Fare']<= 35), 'Fare']= 3
X.loc[(X['Fare'] > 35 ) & (X['Fare']<= 50), 'Fare']= 4
X.loc[(X['Fare'] > 50 ) & (X['Fare']<= 100), 'Fare']= 5
X.loc[(X['Fare'] > 100 ) & (X['Fare']<= 513), 'Fare']= 6


X['Family_size'] = X['SibSp'] +X['Parch'] +1
print("/n/n family size/n/n",X['Family_size']) 

X.drop(['SibSp'] , axis =1 , inplace = True)
X.drop(['Parch'], axis =1 , inplace = True)

#spilting the dataset
X_train, X_valid , Y_train , Y_valid = train_test_split(X, y ,test_size=0.2 ,random_state=7 )

print("X_train  \n\n\n", X_train)
print("X_valid \n\n",X_valid)
print("Y_train \n\n\n",Y_train)
print("Y_valid \n\n\n", Y_valid)
print("\n\n\n")


#xgboost

model = XGBClassifier()
model.fit(X_train, Y_train)
y_pred1 = model.predict(X_valid)
print("Xgboost \n\n")
print(y_pred1)
predictions1 = [round(value) for value in y_pred1]
# evaluate predictions
accuracy = accuracy_score(Y_valid, predictions1)
print("Accuracy: %.2f%%" % (accuracy * 100.0))





#ks test
from scipy.stats import ks_2samp
print("KS stat \n\n\n")
print(ks_2samp(Y_valid, y_pred1))