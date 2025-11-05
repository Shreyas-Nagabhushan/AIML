import numpy as np 
from sklearn import datasets
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

class LinearRegression: 
    
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr 
        self.n_iters = n_iters
        self.weights = None 
        self.bias = None 

    def fit(self, x_train, y_train):
        num_train, num_features = np.shape(x_train) 
        self.weights = np.zeros(num_features) 
        self.bias = 0 

        for _ in range(self.n_iters):
            y_predicted = np.dot(x_train, self.weights) + self.bias 

            dw = (1/num_train) * np.dot(np.transpose(x_train), (y_predicted - y_train)) 
            db = (1/num_train) * np.sum(y_predicted - y_train) 

            self.weights = self.weights - self.lr * dw 
            self.bias -= self.lr * db 
 

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

if __name__ == "__main__":
    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    regressor = LinearRegression(lr=0.01, n_iters=1000)
    regressor.fit(X, y)
    predictions = regressor.predict(X)

    print("Predictions:", predictions)

    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    error = mse(y, predictions)
    print("Mean Squared Error:", error)

    plt.scatter(X, y, color='blue', marker='o', s=30)
    plt.plot(X, predictions, color='red')
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.title("Linear Regression Fit")
    plt.savefig("linear_regression_plot.png")

