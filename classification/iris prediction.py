from random import *
import numpy as np
import random
import sklearn
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn import datasets as ds
from sklearn.datasets import (load_iris,make_classification)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import seaborn as sns
from mlxtend import plotting as pl
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA


#from sklearn.model_selection import train_test_split
iris = sklearn.datasets.load_iris()

def p(*priI_Inted):
    for I_I in priI_Inted:
        print(I_I)



class datapoint:
    dt = 0.0
    lab = 0.0

#x_train,x_test,y_train,y_test
def Split(Data,Labels,testRatio,valRatio):
    N = len(Data)
    
    testSize = int(testRatio*N)
    valSize = int(valRatio*N)
    allData = []

   

    for i in range(0,N):
        x = datapoint()
        x.dt = np.array([Data[i][0],Data[i][1],Data[i][2],Data[i][3]],dtype=float)
        x.lab = Labels[i]
        allData.append(x)


    random.seed(7)
    random.shuffle(allData)
    
    
    trainSize = N - (testSize+valSize)



    #p(trainSize)
    x_train = np.zeros((trainSize,4),dtype=float)
    y_train = np.zeros((trainSize),dtype=int)
    
    x_val = np.zeros((valSize,4),dtype=float)
    y_val = np.zeros((valSize),dtype=int)

    x_test = np.zeros((testSize,4),dtype=float)
    y_test = np.zeros((testSize),dtype=int)
    
    i = 0
    c = 0
    while i < testSize:
        x_test[c] = allData[i].dt
        y_test[c] = allData[i].lab
        i = i + 1
        c = c + 1

    c = 0
    while i < testSize + valSize:
        x_val[c] = allData[i].dt
        y_val[c] = allData[i].lab
        i = i + 1
        c = c + 1

    c = 0
    while i < N:
        x_train[c] = allData[i].dt
        y_train[c] = allData[i].lab
        i = i + 1
        c = c + 1
    return x_train,x_test,x_val,y_val,y_train,y_test


def calculate_accuracy(predicted_y , y):
    sum = 0
    l = len(y)
    for i in range(0,l):
        if predicted_y[i] == y[i]: 
            sum += 1

    return float(100*sum/l)
    

Data = iris.data
Labels = iris.target



x_train,x_test,x_val,y_val,y_train,y_test = Split(Data,Labels,0.30,0.30)


clf = GaussianNB()

clf.fit(x_train,y_train)


predicted_y = clf.predict(x_test)

accuracy = calculate_accuracy(predicted_y,y_test)


p(accuracy)


    
pca = PCA(n_components=2)


xx = pca.fit_transform(x_train)
clf.fit(xx,y_train)
pl.plot_decision_regions(xx, y_train, clf=clf)
plt.show()