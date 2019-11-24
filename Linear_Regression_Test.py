import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
#matplotlib inline


dataset=pd.read_csv('Weather.csv',low_memory=False)

dataset.describe()

dataset.plot(x='MinTemp', y='MaxTemp', style='o')  
plt.title('MinTemp vs MaxTemp')  
plt.xlabel('MinTemp')  
plt.ylabel('MaxTemp')  
plt.figure(figsize=(15,30))
plt.tight_layout()
sns.distplot(dataset['MaxTemp'])
#plt.show()

X = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm

(regressor.intercept_)
(regressor.coef_)

y_predict=regressor.predict(X_test)
df=pd.DataFrame({'Actual':y_test.flatten(),'Predict':y_predict.flatten()})
df1=df.head()

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_predict, color='red', linewidth=2)
plt.show()
