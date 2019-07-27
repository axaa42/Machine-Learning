import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data=pd.read_csv("data.csv")

#The follwoing is a gradeint descent functionw.Watch youtube of codebasics to know how it works
def gradient_descent(x,y):
    m_curr = b_curr = 0 #We will start with some random values for m and b and gradeint will find way to get to globlal minimum
    #iterations = 100000 #THE AMOUNT OF ITME WE WILL RUN OUR FOR LOOP TO FIND THE VALUES
    n = len(x)#This is used as part of formula
    learning_rate = 0.002 #The amount of steps the gradeint will take to get there
    #Follwoing is formula which will test our values and update
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))
#gradient_descent(data["Height"],data["Weight"])
        
#We want to campare how the linear regression algorim and gradeint descent(our manuel one) campare
x_train, x_test, y_train, y_test = train_test_split(data["Height"].values,data["Weight"].values,test_size=0.10,random_state=42)
x_train= x_train.reshape(-1, 1) #We will reshake the data.Note the .values above means it is array
y_train= y_train.reshape(-1, 1)

reg=LinearRegression()
reg.fit(x_train, y_train)
gradient_descent(x_train,y_train)#Our algorim 

# model evaluation for training set
y_train_predict = reg.predict(x_train)
rmse = mean_squared_error(y_train, y_train_predict)#The skit learn algrim

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


#The follwoing is exploratery analysis on the gradeint descent.
        
'''

q = data["engine-size"].quantile(0.99)
#data=data[data["engine-size"] < 202]

q = data["price"].quantile(0.99)
#data=data[data["price"] < 25500]
corr=data.corr()["price"]
gradient_descent(data["engine-size"].values,data["price"].values)
'''
'''

X_train, X_test, y_train, y_test = train_test_split(data["engine-size"].values,data["price"].values,test_size=0.20,random_state=42)
x_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)

reg=LinearRegression()
reg.fit(x_train, y_train)

#gradient_descent(x_train,y_train)


# model evaluation for training set
y_train_predict = reg.predict(x_train)
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
'''
