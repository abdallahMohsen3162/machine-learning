import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


#fast print
def p(*priI_Inted):
    for I_I in priI_Inted:
        print(I_I)


#weight =(xt*x)^-1 * (xt*y)
def least_square_method(X_train,y_train):
    xtranspose = np.transpose(X_train)
    xtranspose_x = np.matmul(xtranspose,X_train)
    xtranspose_x = np.linalg.inv(xtranspose_x)
    xtranspose_y = np.matmul(xtranspose,y_train)
    return np.matmul(xtranspose_y,xtranspose_x)



def cost(ypred,y_test):
    sum =  0
    for i in range(0,len(ypred)):
        sum += (ypred[i] - y_test[i])**2
    return sum/(2*len(ypred))


def MSE(ypred,y_test):
    sum =  0
    for i in range(0,len(ypred)):
        sum += (ypred[i] - y_test[i])**2
    return sum/len(ypred)



def loss(y,ypred):
    return np.divide(np.sum((y-ypred)**2, axis=0), len(y)) 



