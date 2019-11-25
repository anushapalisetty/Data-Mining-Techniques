# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


pd.set_option('expand_frame_repr',False)

df=pd.read_csv("diabetes.csv")
#ax=sns.pairplot(df,hue='Outcome')
print(df.describe())

print(df.isnull().values.any())


for i, col in enumerate(df.columns):
    plt.figure(i)
    sns.boxplot(x=df[col])

corr=df.corr()

pos_corr=corr[corr>0]
pos_corr.to_csv('pos_corr.csv')

corr.to_csv('corr.csv')
plt.figure(11)
sns.heatmap(corr,annot=True,cmap='Blues')


corr_coeff=corr[corr>0.3]


plt.figure(12)
sns.scatterplot(x="Pregnancies", y="Age", data=df,hue='Outcome')
plt.figure(13)
sns.scatterplot(x="Glucose", y="Insulin", data=df,hue='Outcome')
plt.figure(14)
sns.scatterplot(x="SkinThickness", y="Insulin", data=df,hue='Outcome')
plt.figure(15)
sns.scatterplot(x="SkinThickness", y="BMI", data=df,hue='Outcome')
plt.figure(16)
sns.scatterplot(x="Glucose", y="Outcome", data=df,hue='Outcome')





X=df.iloc[:,0:8]
y=df['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
classifier=LogisticRegression(solver='lbfgs',C=100,max_iter=500)
#classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)

clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

y_test = np.array(list(y_test))
y_pred = np.array(y_pred)
wdf_new=pd.DataFrame({'Actual': y_test.flatten(),'Predicted':y_pred.flatten()})
wdf_new.to_csv('GuassianNB_Prediction_Results.csv',index=False)

correct = (y_test == y_pred).sum()
incorrect = (y_test != y_pred).sum()
accuracy = correct / (correct + incorrect) * 100
print('Accuracy',accuracy)


###Logisyic performed better


for name,series in df.iteritems():
    count,division=np.histogram(series,bins=np.linspace(series.min(),series.max(),5))
    print('Attribute Name : ',name)
    print('border bins'+name,division)
    print('Frequency : ',count)
    

plt.figure(18)
border_binsPregnancies=[ 0,4.25 , 8.5 , 12.75 ,17  ]
plt.hist(df['Pregnancies'],bins=border_binsPregnancies)
plt.xlabel('Pregnancies')
plt.ylabel('Frequency')

plt.figure(19)
border_binsGlucose=[  0,   49.75,  99.5,  149.25 ,199  ]
plt.hist(df['Glucose'],bins=border_binsGlucose)
plt.xlabel('Glucose')
plt.ylabel('Frequency')

plt.figure(20)
border_binsBloodPressure=[  0,   30.5 , 61,  91.5, 122 ]
plt.hist(df['BloodPressure'],bins=border_binsBloodPressure)
plt.xlabel('BloodPressure')
plt.ylabel('Frequency')

plt.figure(21)
border_binsSkinThickness=[ 0, 24.75, 49.5 , 74.25, 99 ]
plt.hist(df['SkinThickness'],bins=border_binsSkinThickness)
plt.xlabel('SkinThickness')
plt.ylabel('Frequency')

plt.figure(22)
border_binsInsulin=[  0, 211.5 ,423,  634.5, 846 ]
plt.hist(df['Insulin'],bins=border_binsPregnancies)
plt.xlabel('Insulin')
plt.ylabel('Frequency')

plt.figure(23)
border_binsBMI= [ 0,   16.775, 33.55 , 50.325 ,67.1  ]
plt.hist(df['BMI'],bins=border_binsPregnancies)
plt.xlabel('BMI')
plt.ylabel('Frequency')

plt.figure(24)
border_binsDiabetesPedigreeFunction =[0.078 , 0.6635 ,1.249 , 1.8345 ,2.42  ]
plt.hist(df['DiabetesPedigreeFunction'],bins=border_binsPregnancies)
plt.xlabel('DiabetesPedigreeFunction')
plt.ylabel('Frequency')

plt.figure(25)
border_binsAge= [21, 36,51, 66, 81]
plt.hist(df['Age'],bins=border_binsAge)
plt.xlabel('Age')
plt.ylabel('Frequency')
    


import numpy as np
diabetes_missing =df
diabetes_missing['BloodPressure']=df['BloodPressure'].replace(0, np.NaN)
print(diabetes_missing.isnull().sum())


from sklearn.impute import IterativeImputer

imp = IterativeImputer(max_iter=100, verbose=0)
imp.fit(diabetes_missing)
imputed_df = imp.transform(diabetes_missing)
imputed_df = pd.DataFrame(imputed_df, columns=diabetes_missing.columns)
imputed_df.to_csv('imputed.csv')
print(imputed_df.isnull().sum())

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
#imp = SimpleImputer(strategy='mean',verbose=0)
imp = IterativeImputer(max_iter=100, verbose=0)
logreg = LogisticRegression(solver='lbfgs',C=100,max_iter=500)
steps = [('imputation', imp),('logistic_regression', logreg)]
pipeline = Pipeline(steps)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                 test_size=0.2, random_state=0)
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(pipeline.score(X_test, y_test))










