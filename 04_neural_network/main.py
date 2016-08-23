from scipy.io import loadmat

data = loadmat('ex3data1.mat')
print data['X'].shape
print data['y'].shape