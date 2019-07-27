from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import metrics
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt



data=load_boston()

#print(data)
feature_names=data.data#This is the data for the features

names_of_features=data.feature_names#This is the column names of the features

target_names=data.target#This is the target data that we want to predict


#k fold works by running 10 iterations on the dataset.It will randmoly take a part as test set and
#rest as training and run the iterations
#Kfold will split out the 10 iteration results(scroing is up to u) and you can taKe mean as it gives more accurate result

mean={}
#The following code find what is the best k value for knn.Through k fold cross validation
for x in range(1,50):#Amount of k nreaset neigboht tests
    neigh = KNeighborsRegressor(n_neighbors=x)#testing
    #The neigh is the algorim we using.the second and target are feature and target.the cv is amount of iteration.scoring is squared cuz thats only one we can do for regression(popular)
    cross=cross_val_score(neigh,feature_names, target_names, cv=10,scoring="neg_mean_squared_error")
    sqr=cross
    mean[x]=cross.mean()# take mean of each iteration of cross validation which is 10
print(mean)
plt.plot(*zip(*mean.items()))#We will plot the points.we will use x as the k and y and error rate
plt.show()
#From the graph we can see that after 19 it goes flat.So it makes senese just to use 19 as our K.





'''
#The following uses tradional training and testing
X_train, X_test, y_train, y_test = train_test_split(feature_names, target_names,test_size=0.20,random_state=42)
#Following is a grid search which find the best k
for x in range(1,20):
    neigh = KNeighborsRegressor(n_neighbors=x)#Testing differnt k values randing from 1-20
    neigh.fit(X_train, y_train)#now we fit our data
    prediction=neigh.predict(X_test)#This will predict the values tesing values
    two_features_mse = mean_squared_error(y_test, prediction)#Now we will compare prediction with the actual
    best_k[x]=two_features_mse
#two_features_rmse = two_features_mse ** (1/2)
print(best_k)
#print(prediction) 

'''