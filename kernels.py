import numpy as np

def kernel(kernel_type, gamma = 0.2, c = 1):
    ## x : np.array([....])
    def linear(x, z):
        return np.matmul(x,z.T)

    def poly(x, z):
        return (np.matmul(x,z.T) + 1)**gamma
    
    def rbf(x, z):
        tmp = x - z
        return np.exp(-gamma*np.matmul(tmp, tmp.T))
    
    def sigmoid(x, z):
        return np.tanh(gamma*np.matmul(x, z.T) + c)

    def spline(x, z):
        tmpAbs = np.abs(x-z)
        tmpMin = np.vectorize(lambda x, z: np.min([x, z]))(x, z)
        tmpMin2 = np.multiply(tmpMin, tmpMin)
        length = len(x)
        result = 1
        for i in range(length):
            result *= (1 + x[i]*z[i] + tmpAbs[i] * tmpMin2[i] / 2 + tmpMin[i] * tmpMin2[i] / 3)
        return result

    if (kernel_type == 'spline'):
        return spline
    elif (kernel_type == 'sigmoid'):
        return sigmoid
    elif (kernel_type == 'rbf'):
        return rbf
    elif (kernel_type == 'poly'):
        return poly
    elif (kernel_type == 'linear'):
        return linear
    else:
        return rbf

