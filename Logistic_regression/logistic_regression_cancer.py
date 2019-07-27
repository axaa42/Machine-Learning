from sklearn.datasets import load_breast_cancer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
from sklearn.metrics import confusion_matrix

#In this project we will use Logistic Regression to classify wheather a tumor is benigh or malignment by its features through logsitic regression
data=load_breast_cancer()


X_train,X_test,y_train,y_test=train_test_split(data["data"],data["target"],test_size=0.25,random_state=0) # Splitting the data for test and train

# instantiate the model (using the default parameters)
logreg = LogisticRegression()

# fit the model with data
logreg.fit(X_train,y_train)

#
y_pred=logreg.predict(X_test) #Using the test set to predict its values

#The following code compares the prediction and actual list to see how much it mathced manually
counter=0

for index,value in enumerate(y_pred):
    if value==y_test[index]:
        counter+=1


    
    
#print(y_test)
#print(y_pred)


#The follwoing shows the confsuion matrix. IT will show how accurate our model is in prediction correct and wrong values through specficity and sensistivity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn+fp)
sensitivity= tp/(tp+fn)
print("Specificity: ",specificity)
print("Sensitivity: ",sensitivity)
#Good at predicting cancers that malign.but not at beingn.
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #This prints out the accuracy of our model
print("Precision:",metrics.precision_score(y_test, y_pred))#The precision of our model
print("Recall:",metrics.recall_score(y_test, y_pred))#The recall

