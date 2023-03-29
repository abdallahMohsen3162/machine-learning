'''
Name: Mahmoud Wael , ID: 20200505
Name: Abdallah Mohsen , ID: 20200304
Name: Ahmed Ali , ID: 20200030
'''

import numpy as np
import pandas as pd
# from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

# Importing the dataset
data = pd.read_csv('data_banknote_authentication.csv')

# Shuffling the dataset
data = shuffle(data)

# Selecting the 'Variance' and 'Skewness' columns
features = data[['Variance', 'Skewness']]

# Standardizing the dataset
Standardized_Data = (features - features.mean()) / features.std()

# Scattering plot the data
plt.scatter(Standardized_Data.iloc[:, 0] , Standardized_Data.iloc[:, 1], c = data['Class'])
plt.xlabel('Variance')
plt.ylabel('Skewness')
plt.title('Scatter Plot of the Data')

# Splitting the data into labels and features
Y_labels = data['Class'].values
X_features = Standardized_Data[['Variance', 'Skewness']].values
X_train, X_test, Y_train, Y_test = train_test_split(X_features, Y_labels, test_size = 0.3)

# Implementing the Perceptron algorithm
class Perceptron:

    def __init__(self, learning_rate = 0.1, no_iterations = 1000):
        self.learning_rate = learning_rate
        self.no_iterations = no_iterations

    def fit(self, X, Y):
        no_samples, no_features = X.shape
        self.weights = np.zeros(no_features)
        self.bias = 0

        for i in range(self.no_iterations):
            for x, y in zip(X, Y):
                linear_output = np.dot(x, self.weights) + self.bias
                y_predicted = np.where(linear_output >= 0, 1, -1)
                weight_update = self.learning_rate * (y - y_predicted)
                self.weights += weight_update * x
                self.bias += weight_update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        predictions = np.where(linear_output >= 0, 1, -1)
        return predictions

    def Calculate_Accuracy(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        accuracy = np.sum(Y_test == Y_predict) / len(Y_test)
        return accuracy

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
P_Accuracy = perceptron.Calculate_Accuracy(X_test, Y_test)
print("The Perceptron model accuracy:", P_Accuracy)

# Plotting the decision boundary of the Perceptron model
# fig = plt.subplots()
# plot_decision_regions(X_train, Y_train, clf = perceptron)
# plt.title('Perceptron Decision Boundary')
# plt.show()

# Implementing the Adaline algorithm
class Adaline:

    def __init__(self, learning_rate=0.0001, no_iterations=1000):
        self.learning_rate = learning_rate
        self.no_iterations = no_iterations

    def fit(self, X, Y):
        no_samples, no_features = X.shape
        self.weights = np.zeros(no_features)
        self.bias = 0
        self.costs = []

        for i in range(self.no_iterations):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = Y - output
            self.weights += self.learning_rate * X.T.dot(errors)
            self.bias += self.learning_rate * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.costs.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.weights) + self.bias

    def activation(self, X):
        return X

    def predict(self, X):
        return np.where(self.activation(self.net_input(X)) >= 0, 1, -1)

    def Calculate_Accuracy(self, X_test, Y_test):
        Y_predict = self.predict(X_test)
        accuracy = np.sum(Y_test == Y_predict) / len(Y_test)
        return accuracy

adaline = Adaline()
adaline.fit(X_train, Y_train)
A_Accuracy = adaline.Calculate_Accuracy(X_test, Y_test)
print("The Adaline model accuracy:", A_Accuracy)

# Plotting the decision boundary of the Adaline model
# fig = plt.subplots()
# plot_decision_regions(X_train, Y_train, clf = adaline)
# plt.title('Adaline Decision Boundary')
# plt.show()