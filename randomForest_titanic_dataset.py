# -*- coding: utf-8 -*-
"""
Created on Wed Nov 07 08:06:03 2018

@author: patkarm
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import roc_curve, roc_auc_score
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

import seaborn as sns

#loaded the dataset

titanic = pd.read_csv("train.csv")
#print(titanic.shape)#tells about rows*column of the dataset
#print("categorical data \n\n\n ", titanic.describe(include=['all']))

#visualisation
#to be learnt


survived = 'survived'
not_survived= 'not survived'
fig,axes = plt.subplots(nrows=1,ncols=2 ,figsize=(10,4))
women = titanic[titanic['Sex']=='female']
men = titanic[titanic['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

FacetGrid = sns.FacetGrid(titanic, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

sns.barplot(x='Pclass', y='Survived', data=titanic)


grid = sns.FacetGrid(titanic, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();


#preprocessing the data as it contains a lot of strings and we can't process those
# there are 3 stations , we have created 3 more columns and gave a binary representations , 0 if not arrived at that station, 1 if arrived at that station

#loaded the dataset

#titanic = pd.read_csv('train.csv')
titanic_test = pd.read_csv('test.csv')
#print("the new dataset is  n\n\n\n",titanic_test.head())
#print(titanic.shape)#tells about rows*column of the dataset


#preprocessing the data as it contains a lot of strings and we can't process those
# there are 3 stations , we have created 3 more columns and gave a binary representations , 0 if not arrived at that station, 1 if arrived at that station
port = pd.get_dummies(titanic.Embarked, prefix='Embarked')
port3 = pd.get_dummies(titanic_test.Embarked, prefix='Embarked')

titanic = titanic.join(port)
titanic.drop(['Embarked'], axis=1, inplace=True)

titanic_test = titanic_test.join(port)
titanic_test.drop(['Embarked'], axis=1, inplace=True)
#print(titanic)

def clean_cabin(x):
  try:
    return x[0]
  except TypeError:
    return "N"

#titanic.Sex = titanic.Sex.map({'male':0}, 'female':1})
titanic.Sex = titanic.Sex.map({'male':0, 'female':1})
titanic_test.Sex = titanic_test.Sex.map({'male':0, 'female':1})

test = titanic_test
#bucket_array = np.linspace(1,100,10)
#print (bucket_array)

#bins = [0, 1, 5, 10, 25, 50, 100]
#titanic['binned_age'] = pd.cut(titanic['Age'], bins)
#print (titanic)
#print(titanic.['Age'].max)



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

test.drop(['Ticket'] , axis =1 , inplace=True)
test.drop(['PassengerId'], axis =1 , inplace =True)
test.drop(['Name'], axis =1 , inplace=True)

#checking summary of columns
#X.info()
#age has only 714 entries, rest should be null
#check if any missing values

#print(X.isnull().values.any())

#true 


X.Age.fillna(X.Age.mean(), inplace=True)
#print(X.isnull().values.any())
#false, no NaN values , good to go

test.Age.fillna(test.Age.mean(), inplace=True)
#print(X.isnull().values.any())


X['Cabin'] = X.Cabin.apply(clean_cabin)
port1 = pd.get_dummies(X.Cabin, prefix='Cabin')

X = X.join(port1)
X.drop(['Cabin'], axis=1, inplace=True)

test['Cabin'] = test.Cabin.apply(clean_cabin)
port2 = pd.get_dummies(test.Cabin, prefix='Cabin')

test = test.join(port2)
test.drop(['Cabin'], axis=1, inplace=True)

#making bins

#bins = [0, 10, 20, 30, 40, 50, 60 ,70,80,90,100]
#X['binned_age'] = pd.cut(X['Age'], bins)

#X.drop(['Age'], axis=1 , inplace=True)
#float[0:10]:0,'(10, 20]':1,'(20, 30]':2,'(30, 40]':3,'(40, 50]':4,'(50, 60]':5,'(60, 70]':6,'(70, 80]':7,'(80, 90]':8, '(90, 100]':100,
#X.binned_age = X.Age.map({for x in columns:
#if 0<=x<10 :0
#elif 10<= x<20 :1
#elif 20<= x<30 :2
#elif 30<= x<40 :3
#elif 40<= x<50 :4
#elif 50<= x<60 :5
#elif 60<= x<70 :6
#elif 70<= x<80 :7
#elif 80<= x<90 :8
#else 90<= x<100 :9 })
#print (X.binned_age)

#X.loc[X['Age']<= 10, 'Age']= 0
#X.loc[(X['Age'] > 10 ) & (X['Age']<= 20), 'Age']= 1
#X.loc[(X['Age'] > 20 ) & (X['Age']<= 30), 'Age']= 2
#X.loc[(X['Age'] > 30 ) & (X['Age']<= 40), 'Age']= 3
#X.loc[(X['Age'] > 40 ) & (X['Age']<= 50), 'Age']= 4
#X.loc[(X['Age'] > 50 ) & (X['Age']<= 60), 'Age']= 5
#X.loc[(X['Age'] > 60 ) & (X['Age']<= 70), 'Age']= 6
#X.loc[(X['Age'] > 70 ) & (X['Age']<= 80), 'Age']= 7


X.loc[X['Age']<= 5, 'Age']= 0
X.loc[(X['Age'] > 5 ) & (X['Age']<= 10), 'Age']= 1
X.loc[(X['Age'] > 10 ) & (X['Age']<= 15), 'Age']= 2
X.loc[(X['Age'] > 15 ) & (X['Age']<= 20), 'Age']= 3
X.loc[(X['Age'] > 20 ) & (X['Age']<= 25), 'Age']= 4
X.loc[(X['Age'] > 25 ) & (X['Age']<= 30), 'Age']= 5
X.loc[(X['Age'] > 30 ) & (X['Age']<= 35), 'Age']= 6
X.loc[(X['Age'] > 35 ) & (X['Age']<= 40), 'Age']= 7
X.loc[(X['Age'] > 40 ) & (X['Age']<= 45), 'Age']= 8
X.loc[(X['Age'] > 45 ) & (X['Age']<= 50), 'Age']= 9
X.loc[(X['Age'] > 50 ) & (X['Age']<= 55), 'Age']= 10
X.loc[(X['Age'] > 55 ) & (X['Age']<= 60), 'Age']= 11
X.loc[(X['Age'] > 60 ) & (X['Age']<= 65), 'Age']= 12
X.loc[(X['Age'] > 65 ) & (X['Age']<= 70), 'Age']= 13
X.loc[(X['Age'] > 70 ) & (X['Age']<= 75), 'Age']= 14
X.loc[(X['Age'] > 75 ) & (X['Age']<= 80), 'Age']= 15


test.loc[X['Age']<= 5, 'Age']= 0
test.loc[(X['Age'] > 5 ) & (test['Age']<= 10), 'Age']= 1
test.loc[(X['Age'] > 10 ) & (test['Age']<= 15), 'Age']= 2
test.loc[(X['Age'] > 15 ) & (test['Age']<= 20), 'Age']= 3
test.loc[(X['Age'] > 20 ) & (test['Age']<= 25), 'Age']= 4
test.loc[(X['Age'] > 25 ) & (test['Age']<= 30), 'Age']= 5
test.loc[(X['Age'] > 30 ) & (test['Age']<= 35), 'Age']= 6
test.loc[(X['Age'] > 35 ) & (test['Age']<= 40), 'Age']= 7
test.loc[(X['Age'] > 40 ) & (test['Age']<= 45), 'Age']= 8
test.loc[(X['Age'] > 45 ) & (test['Age']<= 50), 'Age']= 9
test.loc[(X['Age'] > 50 ) & (test['Age']<= 55), 'Age']= 10
test.loc[(X['Age'] > 55 ) & (test['Age']<= 60), 'Age']= 11
test.loc[(X['Age'] > 60 ) & (test['Age']<= 65), 'Age']= 12
test.loc[(X['Age'] > 65 ) & (test['Age']<= 70), 'Age']= 13
test.loc[(X['Age'] > 70 ) & (test['Age']<= 75), 'Age']= 14
test.loc[(X['Age'] > 75 ) & (test['Age']<= 80), 'Age']= 15


#print ("max fare " , X['Fare'].max())
#print("min fare ", X['Fare'].min())
#print(" mean fare", X['Fare'].mean())
#print("std deviation")


X.loc[X['Fare']<= 7, 'Fare']= 0
X.loc[(X['Fare'] > 7 ) & (X['Fare']<= 14), 'Fare']= 1
X.loc[(X['Fare'] > 14 ) & (X['Fare']<= 25), 'Fare']= 2
X.loc[(X['Fare'] > 25 ) & (X['Fare']<= 35), 'Fare']= 3
X.loc[(X['Fare'] > 35 ) & (X['Fare']<= 50), 'Fare']= 4
X.loc[(X['Fare'] > 50 ) & (X['Fare']<= 100), 'Fare']= 5
X.loc[(X['Fare'] > 100 ) & (X['Fare']<= 513), 'Fare']= 6

test.loc[X['Fare']<= 7, 'Fare']= 0
test.loc[(X['Fare'] > 7 ) & (test['Fare']<= 14), 'Fare']= 1
test.loc[(X['Fare'] > 14 ) & (test['Fare']<= 25), 'Fare']= 2
test.loc[(X['Fare'] > 25 ) & (test['Fare']<= 35), 'Fare']= 3
test.loc[(X['Fare'] > 35 ) & (test['Fare']<= 50), 'Fare']= 4
test.loc[(X['Fare'] > 50 ) & (test['Fare']<= 100), 'Fare']= 5
test.loc[(X['Fare'] > 100 ) & (test['Fare']<= 513), 'Fare']= 6



X['Family_size'] = X['SibSp'] +X['Parch'] +1
#print("/n/n family size/n/n",X['Family_size']) 

X.drop(['SibSp'] , axis =1 , inplace = True)
X.drop(['Parch'], axis =1 , inplace = True)


test['Family_size'] = test['SibSp'] +test['Parch'] +1
#print("/n/n family size/n/n",test['Family_size']) 

test.drop(['SibSp'] , axis =1 , inplace = True)
test.drop(['Parch'], axis =1 , inplace = True)


#improves ks stat

X.drop(['Cabin_T'], axis = 1, inplace = True)
print("test is ",test.head())


#spilting the dataset
X_train, X_valid , Y_train , Y_valid = train_test_split(X, y ,test_size=0.2 ,random_state=7 )

sns.barplot('Pclass','Survived', data= titanic)

#print("X_train  \n\n\n", X_train)
#print("X_valid \n\n",X_valid)
#print("Y_train \n\n\n",Y_train)
#print("Y_valid \n\n\n", Y_valid)

print("valid  is ",X_train.head())









#Random forest

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
#print("abhi much mein kahi , baki thoda hai zindagi",random_forest.oob_score)

Y_prediction = random_forest.predict(X_valid)
print("Random forest output")
print(Y_prediction)
#print(random_forest.score(X_train, Y_train))
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

print (roc_auc_score(Y_valid, Y_prediction))

fpr, tpr, _ = roc_curve(Y_valid, Y_prediction)

plt.clf()
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()
#random_forest.oob_score


#optimizing no. of trees

#results =[]
#kresults =[]
#options_rf = [50,100,150,200,500,1000]

#for trees in options_rf:
  #random_forest = RandomForestClassifier(trees)
  #random_forest.fit(X_train, Y_train)
  #acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
  #print("score ", acc_random_forest)

  #results.append(acc_random_forest)
  #Y_prediction = random_forest.predict(X_valid)
  #from scipy.stats import ks_2samp
  #k = ks_2samp(Y_valid, Y_prediction)
  #print(k)
  #kresults.append(k)


#print(results)
#print(kresults)
  
def calc_iv(df, feature, target, pr=True):
   

    lst = []

    df[feature] = df[feature].fillna("NULL")

    for i in range(df[feature].nunique()):
        val = list(df[feature].unique())[i]
        print("val is \n\n",val)
        print("\n\n i is  ", i)
        lst.append([feature,                                                        # Variable
                    val,                                                            # Value
                    df[df[feature] == val].count()[feature],                        # All
                    df[(df[feature] == val) & (df[target] == 0)].count()[feature],  # Good (think: Fraud == 0)
                    df[(df[feature] == val) & (df[target] == 1)].count()[feature]]) # Bad (think: Fraud == 1)
    #print("list is \n\n\n",lst)
    
    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())
    #print("\n\n\n dat is \n\n\n",data)

    iv = data['IV'].sum()
    # print(iv)

    return iv, data


df_1 = X_valid
df_1['predict'] = Y_prediction

iv , data = calc_iv(df_1 ,'Age', 'predict')
print("iv is ", iv)
print(data.head())
    
iv1 , data1 = calc_iv(df_1 ,'Fare', 'predict')  

print("iv1 is ", iv1)



#kstest
from scipy.stats import ks_2samp
print(ks_2samp(Y_valid, Y_prediction))


#random_forest.feature_importances_
#feature_importances = pd.Series(random_forest.feature_importances_, index= X.columns)
#print(feature_importances.sort())
