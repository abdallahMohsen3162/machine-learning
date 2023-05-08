import math
import numpy as np

def k_fold(x, y, k = 10):
  n = x.shape[0]
  st = 0
  en = k
  data =[]
  for _ in range(k):
    start = math.floor((st / 100) * n)
    end = math.floor((en / 100) * n)
    # print(start , end)
    x_train = np.concatenate((x[0:start],x[end:n]))
    y_train = np.concatenate((y[0:start],y[end:n]))

    x_test = x[start:end]
    y_test = y[start:end]
    data.append((x_train,y_train,x_test,y_test))
    # one_iter(x_train,y_train,x_test,y_test)

    st += k
    en += k
  return data