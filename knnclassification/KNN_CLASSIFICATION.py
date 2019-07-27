import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import metrics
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score

#This project looks will classify stars from its features using KNN CLASSIFIER.

df=pd.read_csv("pulsar_stars.csv")#Reading data


features=df.drop("target_class",axis=1)#wE WILL DROP THE TARGET_CLASS AS WE ONLY NEED FEATURES
target=df["target_class"]#wE MAKE TARGET A SEPERATE THINGS.SO WE CAN SEPEARTLY USE IT

#Below we want to test best accuracy model using cross valudation.We will use 1 to 100 neigbours
best_n={}
for x in range(1,11):
    neigh = KNeighborsClassifier(n_neighbors=x)#testing
    #The neigh is the algorim we using.the second and target are feature and target.the cv is amount of iteration.scoring is squared cuz thats only one we can do for regression(popular)
    cross=cross_val_score(neigh,features, target, cv=10,scoring="accuracy")#Give us another way to test our model.by iterating 10 times and we take average
    best_n[x]=cross.mean()
print(best_n)
import operator
best_n=max(best_n.items(), key=operator.itemgetter(1))[0]#This will find the best niegbour
#NOTE THE RESULT WE GOT WAS THAT 9 WAS BEST NEIGBOUR.AND IN GRID SEARCH THEY TOLD US THIS


#The follwoing is using KNN through tradeional train and test
'''
model = KNeighborsClassifier(n_neighbors=3)#wE WILL ONLY USE 3 NEIGBOURS.NOTE CLASSIFICATION WORKS BY VOTING MAHORITY

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.25,random_state=42)#Train and test are split 25% for test and 75% for traing
# Train the model using the training sets
model.fit(X_train,y_train)#Fitting the model
predicted= model.predict(X_test)#Predicting it

accuracy=metrics.accuracy_score(y_test, predicted) #finding accuracy

print("Accuracy of model: ",accuracy)#THe accuracy of model
'''


#The follwoing will show how to do grid search for knn
'''
#Watch this link to understand below code. https://www.youtube.com/watch?v=CgmvAMiVKFE
#You're Welcome, Future self.

#This is grid search. Grid search is a way for the function to give you the best hyperparameter for your model.

k_range = list(range(1,10))#We want the model to give us the best neigbour out of 10.Grid search will look at which will give us best accuracy
#weight_options = ["uniform", "distance"]#This is another hyperparameter we can put.Grid search allowes us to test various hypermpareters 

param_grid = dict(n_neighbors = k_range)#Parameters are accepted as dictionary thats why we doing this
#print (param_grid) 
knn = KNeighborsClassifier()#This is knn neigbour classififier
#The code below is train and test
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.25,random_state=42)#Train and test are split 25% for test and 75% for traing

#The following is the grid search functiion.1)is the algorimts we want.2)is the parameters we want to test.3)is the cross validation or how many times it will test it
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')#4)we want accuracy as scoring
grid.fit(X_train,y_train)#Note we could have put the whole dataset features and target.But i just chose this cuz i am stupid

#print(grid.grid_scores_)

print(grid.grid_scores_[0].parameters)
print(grid.grid_scores_[0].cv_validation_scores)
print(grid.grid_scores_[0].mean_validation_score)


print (grid.best_score_)#This will give use the best accuracy score we got in the testing
print (grid.best_params_)#This will tell us which neigbour got the highest accuracy score
print (grid.best_estimator_)#This will us the infomration for which specific paramenter we should put for our hyperparameter in the KNN function
'''

#Implementing grid

'''
#The parameters under are based on what the grid search told us to put as metrics. Which i have below
model=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=9, p=2,
           weights='uniform')
X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.25,random_state=42)#Train and test are split 25% for test and 75% for traing
# Train the model using the training sets
model.fit(X_train,y_train)#Fitting the model
predicted= model.predict(X_test)#Predicting it

accuracy=metrics.accuracy_score(y_test, predicted) #finding accuracy

print("Accuracy of model: ",accuracy)#THe accuracy of model
'''