import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
#from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.feature_selection import RFE

#loaded the dataset

titanic = pd.read_csv('train.csv')
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
print("X is ", X)

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


#logistic regression


model = LogisticRegression()
model.fit(X_train , Y_train)
print("Model score is on training set \n")
print(model.score(X_train, Y_train))
print("\n\n\n")
print("Model score on test set \n\n")
print(model.score(X_valid, Y_valid))
print("Logistic regression output")
print(model.predict(X_valid))
Y_predict =model.predict(X_valid)
print("\n\n\n")

Y_predict1 = model.predict(test)
#print("test output is ",Y_predict1)
titanic_test['Survived_prediction'] = Y_predict1
print(titanic_test)

rfe = RFE(model, 3)
rfe = rfe.fit(X_valid, Y_valid)
print(rfe.support_)
print(X_valid)
print(rfe.ranking_)


#def iv_calculator(Y_predict, Y_valid):
 #   lst1 = []
  #  lst2 = []
    
   # for i in Y_predict:
    #    if (Y_predict == Y_valid):
     #       lst1[i]= 1
      #  else:
       #     lst2[i] = 1
    
    #goods = []
    #bads = []
    
    #goods[0] = sum(lst1[1:10])
    #goods[1] = sum(lst1[11:20])
    #goods[2] = sum(lst1[21:30])
    #goods[3] = sum(lst1[31:40])
    #goods[4] = sum(lst1[41:50])
    #goods[5] = sum(lst1[51:60])
    #goods[6] = sum(lst1[61:70])
    #goods[7] = sum(lst1[71:80])
    
    #bads[0] = sum(lst2[1:10])
    #bads[1] = sum(lst2[11:20])
    #bads[2] = sum(lst2[21:30])
    #bads[3] = sum(lst2[31:40])
    #bads[4] = sum(lst2[41:50])
    #bads[5] = sum(lst2[51:60])
    #bads[6] = sum(lst2[61:70])
    #bads[7] = sum(lst2[71:80])
    
    #print ('goods are ', goods)
    #print ('bads are ', bads)
    
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
df_1['predict'] = Y_predict

iv , data = calc_iv(df_1 ,'Age', 'predict')
print("iv is ", iv)
print(data.head())
    
iv1 , data1 = calc_iv(df_1 ,'Fare', 'predict')  

        
    
    
    



#import statsmodels.api as sm
#sm_model = sm.Logit(Y_train, sm.add_constant(X_train)).fit(disp=0)
#print(sm_model.pvalues)
#sm_model.summary()
#print("\n\n\n")

#import scikitplot as skplt
#import matplotlib.pyplot as plt

#y_true = # ground truth labels
#y_probas = # predicted probabilities generated by sklearn classifier
#skplt.metrics.plot_roc_curve(Y, y_probas)
#plt.show()

#pvalues
#X2 = sm.add_constant(X_valid)
#est = sm.OLS(Y_valid, X2)
#est2 = est.fit()
#print(est2.summary())
#print("\n\n\n")

#auc
#y_pred_proba = model.predict_proba(X_valid)[::,1]
#fpr, tpr, _ = metrics.roc_curve(Y_valid,  y_pred_proba)
#auc = metrics.roc_auc_score(Y_valid, y_pred_proba)
#plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
#plt.legend(loc=4)
#plt.show()
#print ("AUC out put \n\n")
#print(auc)
#print("\n\n\n")
#kstest
from scipy.stats import ks_2samp
print("KS test output \n\n\n\n")
print(ks_2samp(Y_valid, Y_predict))
