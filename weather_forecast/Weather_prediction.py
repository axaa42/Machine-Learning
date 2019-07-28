import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns

#This project will predict temperature from its other features using linear regression.

data=pd.read_csv("weatherHistory.csv")


data=data.dropna() #This will drop all the rows that contrain nan values

#The following will seperate int and objects columns.So it is easier to turn them to catagorial and correaltion for each seperatly
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric = data.select_dtypes(include=numerics)


#apparent temp,humidity,visisbility will be used as features from numeric
#The following will drop the columns that have nothing in common with target from numeric
numeric=numeric.drop(["Loud Cover","Wind Bearing (degrees)","Wind Speed (km/h)","Pressure (millibars)"], axis=1)
#numeric=numeric.drop(["Loud Cover"], axis=1)


#The following is for object columns AKA to see which to turn into catagorical
objects=data.select_dtypes(include=["object"])


pd.options.mode.chained_assignment = None
objects['year'] = objects['Formatted Date'].astype(str).str[0:4]#This makes new column which will have year
objects['year']= objects['year'].astype(int)#Will convert that  column to int  
objects=objects.drop(["Formatted Date"], axis=1)#We will drop orignial as not needed

df=pd.concat([numeric,objects], axis=1)#We will join numeric and object back


df=df.drop(["Daily Summary"],axis=1) #We will drop this cuz we already have a column that is similer to this.plus it has lots of values for dummy

df = pd.get_dummies(df, columns=['Precip Type'],drop_first=True)#Convert the two to dummy
df = pd.get_dummies(df, columns=['Summary'],drop_first=True)

df=df.drop(["Apparent Temperature (C)"],axis=1)#TThis is leaking The data so we remove it

target=df["Temperature (C)"]#This is our target varaivle

features=df.drop(["Temperature (C)"],axis=1)#We drop the target from features

#features=features.drop(["year"],axis=1)



#print(features.columns)















#The rest is simply to know



X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.20,random_state=42)

reg=LinearRegression()
reg.fit(X_train, y_train)


# model evaluation for training set
y_train_predict = reg.predict(X_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))


a1=reg.coef_
a0=reg.intercept_
print("co-efficent of linear_regression =",a1)
print("\n")
print("intercept of linear_regression =",a0)
print("\n")

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print("\n")



# model evaluation for testing set
y_test_predict = reg.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))


print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))




