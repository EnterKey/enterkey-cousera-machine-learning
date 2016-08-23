from numpy import loadtxt, zeros, ones, mean, std

def feature_normalize(input):
    mu = input.mean(axis=0)
    sd = input.std(axis=0)
    input_norm = (input - mu) / sd
    return input_norm

def cost_function(x, y, theta):
    m = x.shape[0]
    predictions = x.dot(theta)
    sqErrors = (predictions - y)
    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J

def gradient_descent(x, y, theta):
    num_iters = 100
    alpha = 0.01
    theta_size = theta.size
    m = x.shape[0]

    J_history = zeros(shape=(num_iters, 1))

    for i in range(num_iters):
        predictions = x.dot(theta)

        for j in range(theta_size):
            temp = x[:, j]
            temp.shape = (m, 1)

            errors_x1 = (predictions - y) * temp

            theta[j][0] = theta[j][0] - alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = cost_function(x, y, theta)

    return theta, J_history

data = loadtxt('ex1data2.txt', delimiter=',')
x = data[:, :-1]
y = data[:, -1]

m, n = x.shape
y.shape = (m, 1)

x_norm = feature_normalize(x)
#theta0, theta1, thetaM
x = ones(shape=(m, 3))
x[:, 1:3] = x_norm

y = feature_normalize(y)

theta = zeros(shape=(n+1, 1))

theta, j_history = gradient_descent(x, y, theta)
print theta, j_history