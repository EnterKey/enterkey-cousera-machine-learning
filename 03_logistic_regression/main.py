from numpy import loadtxt, zeros, e, array, log, ones, where

def feature_normalize(input):
    mu = input.mean(axis=0)
    sd = input.std(axis=0)
    input_norm = (input - mu) / sd

    return input_norm

def sigmoid(X):
    return 1.0 / (1.0 + e ** (-1.0 * X))

def cost_function(theta, x, y):
    m = y.size
    h = sigmoid(x.dot(theta))
    J = (1.0 / m) * ((-y.T.dot(log(h))) - ((1.0 - y.T).dot(log(1.0 - h))))
    return - 1 * J.sum()

def gradient_descent(x, y, theta, alpha = 0.01, num_iters = 1):
    m = x.shape[0]
    theta_size = theta.size
    for i in range(num_iters):

        predictions = x.dot(theta)

        for it in range(theta_size):
            temp = x[:, it]
            temp.shape = (m, 1)

            errors_x1 = (predictions - y) * temp

            theta[it][0] = theta[it][0] - alpha * (1.0 / m) * errors_x1.sum()

        cost_function(theta, x, y)

    return theta

def predict(theta, x):
    m, n = x.shape
    p = zeros(shape=(m, 1))

    h = sigmoid(x.dot(theta))

    for it in range(h.shape[0]):
        if h[it] >= 0.5:
            p[it, 0] = 1
        else:
            p[it, 0] = 0

    return p

data = loadtxt('ex2data1.txt', delimiter=',')
x = data[:, :-1]
y = data[:, -1]
m, n = x.shape
y.shape = (m, 1)

x_norm = feature_normalize(x)
#theta0, theta1, thetaM
x = ones(shape=(m, 3))
x[:, 1:3] = x_norm

theta = zeros(shape=(n+1, 1))

theta = gradient_descent(x, y, theta)
print theta

p = predict(array(theta), x)
print 'Accuracy: %f' % ((y[where(p == y)].size / float(y.size)) * 100.0)
