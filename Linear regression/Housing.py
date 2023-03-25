import numpy as np
import pandas as pd
from lib import p,cost,descend,MSE,loss,gd
from lib import least_square_method
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import random
from sklearn.preprocessing import StandardScaler

alldata = pd.read_csv("USA_Housing.csv")


y = alldata[0:5000]["Price"]

x = alldata.iloc[0:5000, 0:5]

##normalize data
mn = x.iloc[0:5000,0].min()
mx =x.iloc[0:5000,0].max()
x.iloc[: ,0] = (x.iloc[:,0] - mn)/(mx - mn)



mn = x.iloc[0:5000,1].min()
mx =x.iloc[0:5000,1].max()
x.iloc[: ,1] = (x.iloc[:,1] - mn)/(mx - mn)

mn = x.iloc[0:5000,2].min()
mx =x.iloc[0:5000,2].max()
x.iloc[: ,2] = (x.iloc[:,2] - mn)/(mx - mn)

mn = x.iloc[0:5000,3].min()
mx =x.iloc[0:5000,3].max()
x.iloc[: ,3] = (x.iloc[:,3] - mn)/(mx - mn)

mn = x.iloc[0:5000,3].min()
mx =x.iloc[0:5000,3].max()
x.iloc[: ,3] = (x.iloc[:,3] - mn)/(mx - mn)

mn = x.iloc[0:5000,4].min()
mx =x.iloc[0:5000,4].max()
x.iloc[: ,4] = (x.iloc[:,4] - mn)/(mx - mn)

mn = y.iloc[0:5000].min()
mx =y.iloc[0:5000].max()
y.iloc[:5000] = (y.iloc[:5000] - mn)/(mx - mn)





x = np.array(x)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30,shuffle=False)

weights = least_square_method(X_train,y_train)





# p(r2_score(y_test,ypred))

ww = gd(X_train,y_train,weights,0.001,1000)

p(np.matmul(X_test,ww))

# for _ in range(50):
#     w = descend(X_train,y_train,w,0.01)
#     ypred = np.matmul(X_test,w)
#     ls = loss(y_test,ypred)
#     p(f'{_} loss is {ls}, paramters w:{w} ')
    

