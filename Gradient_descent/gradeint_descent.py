from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

boston=load_boston()

describe=boston.DESCR #This describes the whole dataset.Note the feature and target are at different parts so we need to merge it

#The following will put the data into dataframe and the feature names of the columns.But we are missing the target, which is also added
data=pd.DataFrame(boston.data, columns=boston.feature_names)
data['MEDV'] = boston.target #This adds Target with the features.Not it is easy to work with the data

#The column with most corrlation with medv will be picked as the feature to predict
corra=data.corr()["MEDV"] #We can see Medv and lstat has highest corrlation

#plt.scatter(data["LSTAT"],data["MEDV"])

#The following is how to implment gradient descent
def gradient_descent(x,y):
    m_curr = b_curr = 0 #We will have intercept and slope at 0
    iterations = 10000 #We will run this many iterations to find the optimal m and b
    n = len(x) #Used for formula
    learning_rate = 0.002 #The learning rate will be low to avoid mistakes

    for i in range(iterations):#We will run this for loop for the specified iterations to get our values
        y_predicted = m_curr * x + b_curr #This is the formula for linear regression.Where we are trying to find optimal values
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)]) #This is the cost function for linear that we will optimize
        md = -(2/n)*sum(x*(y-y_predicted))#optimzing m
        bd = -(2/n)*sum(y-y_predicted)#optiizing b
        m_curr = m_curr - learning_rate * md#Updateing the m
        b_curr = b_curr - learning_rate * bd#updating the b
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))#Retrun the updated m and b aswell as cost and iteration for our own purposes


b=data["LSTAT"].values #We will use one feature so this is easy to follow
c=data["MEDV"].values#Target 


#print(data.corr()["MEDV"])
X_train, X_test, y_train, y_test = train_test_split(b, c,test_size=0.20,random_state=42) #This is the OLS function that we will campare with
x_train= X_train.reshape(-1, 1)#Data pre-preccessing for x
y_train= y_train.reshape(-1, 1)#data preprocessing for y

reg=LinearRegression()# We will use this to implement our data using OLS
reg.fit(x_train, y_train)#now we fit our data





# model evaluation for training set for ols
y_train_predict = reg.predict(x_train)


#Follwoing is intercept and slope from ols formula
a1=reg.coef_ 
a0=reg.intercept_

rmse = mean_squared_error(y_train, y_train_predict) #Loss function value for ols

gradient_descent(x_train,y_train)#This is our manuel gradient we made to compare with ols

print("co-efficent of linear_regression =",a1)
print("\n")
print("intercept of linear_regression =",a0)
print("\n")

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print("\n")

