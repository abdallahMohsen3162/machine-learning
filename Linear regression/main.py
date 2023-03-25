import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import random
from sklearn.metrics import r2_score
from lib import cost,least_square_method,p,loss

data = np.zeros((1000,3))

target = np.zeros(1000)

#Random Data init
i = 0
random.seed(2)
while i<1000:
    lmt = 1000
    data[i][0] = random.uniform(0,lmt)          
    data[i][1] = random.uniform(0,lmt)          
    data[i][2] = random.uniform(0,lmt)          
    target[i] = (5 * data[i][0]) + (3 * data[i][1]) + (1.5 * data[i][2]) + 6
    i += 1

#Train Test Split

X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.33,shuffle=False)

built_in_lstsq = np.linalg.lstsq(X_train,y_train,rcond=None)
theta = least_square_method(X_train,y_train)
pred = np.matmul(X_test,theta)


#compare
p(f"lstsq from skratch: {theta}")
p(f"lstsq Built in: {built_in_lstsq[0]}")

p("**************************")

#Accuracy
score = r2_score(y_test,pred)
p(f"Score: {score}")














# l_model = linear_model.LinearRegression()

# l_model.fit(X_train, y_train)

# ypred = l_model.predict(X_test)



# weight'(best weight to fit) =(xt*x)^-1*(xt*y)

# y pred = x_test * weight'(best weight to fit)







