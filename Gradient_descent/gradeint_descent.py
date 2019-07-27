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


def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.002

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))


b=data["LSTAT"].values
c=data["MEDV"].values


#print(data.corr()["MEDV"])
X_train, X_test, y_train, y_test = train_test_split(b, c,test_size=0.20,random_state=42)
x_train= X_train.reshape(-1, 1)
y_train= y_train.reshape(-1, 1)

reg=LinearRegression()
reg.fit(x_train, y_train)#now we fit our data

#gradient_descent(X_train,y_train)
x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x_train,y_train)

# model evaluation for training set
y_train_predict = reg.predict(x_train)








a1=reg.coef_
a0=reg.intercept_

rmse = mean_squared_error(y_train, y_train_predict)
print("co-efficent of linear_regression =",a1)
print("\n")
print("intercept of linear_regression =",a0)
print("\n")

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print("\n")

