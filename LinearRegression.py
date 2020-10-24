#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 15:23:31 2019

@author: Anusha
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


###Read the csv file
wdf=pd.read_csv("/Users/pavan/Desktop/Anusha/Courses/Data_Mining/Weather.csv",low_memory=False)

plt.scatter(wdf["MinTemp"],wdf["MaxTemp"])
plt.title('MinTemp vs MaxTemp')  
plt.xlabel("MinTemp")
plt.ylabel("MaxTemp")
plt.show()


##Correlation of Min and Max Temp
print(wdf[["MinTemp","MaxTemp"]].corr())


#replace all Nan with 0's for WTE
print(wdf["WTE"].fillna(0).head())


X=wdf["MinTemp"].values.reshape(-1,1) #Independent Variable
y=wdf["MaxTemp"].values.reshape(-1,1) # Dependent Variable


#Split Data into train (80%) and test set (20%).

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)

#Import Linear regression class and call the method

linear_reg=LinearRegression()
linear_reg.fit(Xtrain,ytrain)  #training data to algorithm

#Retrieving the intercept
print("Intercept " ,linear_reg.intercept_)

#retrieving the slope
print("Slope ",linear_reg.coef_)

## Adding two new columns

ypred=linear_reg.predict(Xtest)

wdf_new=pd.DataFrame({'Actual': ytest.flatten(),'Predicted':ypred.flatten()})
print(wdf_new)


#Calculate the absolutre difference

wdf_new["Error"]= abs(wdf_new['Actual']-wdf_new['Predicted'])

print(wdf_new.describe())


# mean and std of error

print("Mean",wdf_new['Error'].mean())

print("Std",wdf_new['Error'].std())

#Scatter plot for new variable

plt.scatter(Xtest, ytest)
plt.plot(Xtest, ypred, color='red', linewidth=2)
plt.xlabel("Min_temp")
plt.ylabel("Max_Temp")
plt.show()

##bin error

#compute bins

bins = [-30,-20,-10,0,10,20,30,40,50]
wdf_new['binned'] = pd.cut(wdf_new['Predicted'], bins)
#print(wdf_new)
df_m=wdf_new.groupby('binned').mean()
df_s=wdf_new.groupby('binned').std()
print('Mean of Error after binning',df_m)
print('std of the Error after binning',df_s["Error"])


##########################################################################

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score


wqdf=pd.read_csv("winequality.csv",low_memory=False)


wqdf.isnull().any()
wqdf = wqdf.fillna(method='ffill')

X = wqdf[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates','alcohol']].values
y = wqdf['quality'].values

#plt.figure(figsize=(15,10))
#plt.tight_layout()
#seabornInstance.distplot(wqdf['quality'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)


coeff_df = pd.DataFrame(regressor.coef_,columns=['Coefficient']) 
print(coeff_df)

y_pred = regressor.predict(X_test)


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R2 Score',r2_score(y_test, y_pred) )


#calculating correlation
sns.heatmap(wqdf.corr().astype(float), annot=True,cmap="BuGn")
plt.show()


##Removing columns based on correlation and coefficient values
Xnew = wqdf[['volatile acidity','citric acid','chlorides', 'density','sulphates','alcohol']].values
ynew = wqdf['quality'].values

X_train, X_test, y_train, y_test = train_test_split(Xnew, ynew, test_size=0.2, random_state=0)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df1 = df.head(25)
print(df.describe())

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))

















