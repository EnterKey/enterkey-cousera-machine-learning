from numpy import loadtxt, zeros, e, array, log, ones, where, argmax, mean
from scipy.io import loadmat

def feature_normalize(input):
    mu = input.mean(axis=0)
    sd = input.std(axis=0)
    input_norm = (input - mu) / sd

    return input_norm

def cost_function(theta, x, y):
    m = y.size
    h = sigmoid(x.dot(theta))
    J = (1.0 / m) * ((-y.T.dot(log(h))) - ((1.0 - y.T).dot(log(1.0 - h))))
    return - 1 * J.sum()

def sigmoid(X):
    return 1.0 / (1.0 + e ** (-1.0 * X))

def add_column(input_matrix) :
    m, n = input_matrix.shape
    result = ones(shape=(m, n + 1))
    result[:, 1:n + 1] = input_matrix
    return result

def get_accuracy(y_predict, y_real):
    m = y_predict.size
    y_predict.shape = (m, 1)
    result = "accuracy : {: .2%}".format(mean(y_predict == y_real))
    print result

def predict(theta1, theta2, x):
    x = add_column(x)
    h1 = sigmoid(x.dot(theta1.T))

    X_next = add_column(h1)
    h2 = sigmoid(X_next.dot(theta2.T))

    column_idxs = argmax(h2, axis=1)
    p = column_idxs + 1
    return p

data = loadmat('ex3data1.mat')
x = data['X']
y = data['y']

weight = loadmat('ex3weights.mat')
theta1 = weight['Theta1']
theta2 = weight['Theta2']

y_predict = predict(theta1, theta2, x)
get_accuracy(y_predict, y)