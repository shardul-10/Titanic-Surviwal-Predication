import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df= pd.read_csv(r"C:\Users\lenovo\Downloads\titanic_train.csv")

#print(df)

df.isnull().sum()# to see null values
#print(pp)

df['Age'].fillna(round(df['Age'].mean()),inplace=True)
#print(df['Age'].mean())

df.drop(columns=['Cabin'],inplace=True) # to remove column Cabin
#print(list(df.columns))

df.dropna(inplace=True)# it used to remove rows where column still has NaN

#print(df.info())
df.drop(columns=['PassengerId','Name','Ticket'],inplace=True)
#print(df.info())

from sklearn.preprocessing import OneHotEncoder

oge = OneHotEncoder
df['Sex']=pd.get_dummies(df['Sex'],dtype=int,drop_first=True)

#print(df)

df[['C','Q','S']]=pd.get_dummies(df['Embarked'],dtype=int)
print(df.head())

df.drop(columns=['Embarked'],inplace=True)


#train,test,split

df.iloc[:,1:]
df.iloc[:,0:1]
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(df.iloc[:,1:],
df.iloc[:,0:1],test_size=0.2,random_state=42)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train_scaled=scaler.fit_transform(X_train)
X_test_scaler=scaler.transform(X_test)

#print(X_test_scaler)

from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(X_train_scaled,Y_train)


y_pred = lr.predict(X_test_scaler)
#print(Y_test)
#print(y_pred)

#accuracy cheack

from sklearn.metrics import accuracy_score

aacc= accuracy_score(Y_test,y_pred)

#print(aacc)

import pickle
with open('logistic_model.pkl','wb') as file:
    pickle.dump(lr,file)
    
import os
#print("Saved in:",os.getcwd())    