import numpy as np
import SVRs as svrs
import kernels as ker
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

### when imported differnt svrs, using this to match
def svrAPI(x, y, kernel, C, eta, maxIter, para):
    # return curve a b
    use = 'linear' 
    use = 'quadratic' 
    use = 'huber' 
    if (use == 'huber'):
        svr = svrs.huberSVR
    elif (use == 'quadratic'):
        svr = svrs.epsilonQuadraticSVR
    else:
        svr = svrs.epsilonLinearSVR
    return svr(x, y, kernel, C, eta, maxIter, para)


def dataSet(dataType='cos', noiseAmp=0.2, k=0, setNum=0):
    setNum = 128 if (setNum == 0) else setNum
    x_min = -0.5
    x_max = +0.5
    x = np.array([np.arange(x_min, x_max, (x_max - x_min) / setNum)])
    y = noiseAmp * (np.random.rand(1,x.shape[1])-0.5)[0,:]

    D3 = False
    if (dataType == 'log'):
        k = 12 if(k ==0) else k
        y += np.vectorize(lambda x: np.log(1+np.exp(-k*x)))(x[0,:])
    elif (dataType == 'cos'):
        k = 12 * 3.14 if(k ==0) else k
        y += np.vectorize(lambda x: np.cos(k*x))(x[0,:])
    elif (dataType == 'sinc'):
        k = 12 * 3.14 if(k ==0) else k
        y += np.vectorize(lambda x: np.sin(k*x) / x if(x != 0) else k)(x[0,:])
    elif (dataType == 'linear'):
        k = 8 if(k ==0) else k
        y +=  (lambda x: k*x)(x[0,:])  
    elif (dataType == 'sinc3D'):
        D3 = True
        x = np.array([np.arange(x_min, x_max, (x_max - x_min) / int(np.sqrt(setNum)))])
        x1, x2 = np.meshgrid(x,x)
        x = np.array(np.c_[x1.ravel(), x2.ravel()].T)
        k = 8 if(k ==0) else k
        y = noiseAmp * (np.random.rand(1,x.shape[1])-0.5)[0,:]
        y1 = np.vectorize(lambda x: np.sin(k*x) / x if(x != 0) else k)(x[0,:])
        y2 = np.vectorize(lambda x: np.sin(k*x) / x if(x != 0) else k)(x[1,:])
        y += y1 * y2
    else:
        print('currently you can choose ')
        
    return x, y, x1.shape if(D3) else 0


def testSVR(C, maxIter, epsilon, dataSet, kernel, D3 = False):
    x, y, gridShape = dataSet()
    start = datetime.now()
    f, a, _ = svrAPI(x, y, kernel, C, 0.0001, maxIter, epsilon)
    print('time for train:', datetime.now() - start)

    def plot2D(f, a, x, y):
        plt.figure()
        plt.scatter(x[0,:], y, c="#000000", alpha=0.5, label='ori curve')
        z = np.zeros(x.shape[1])
        start = datetime.now()
        for i in range(x.shape[1]):
            z[i] = f(x[:,i])
            if (a[i] > 0.001 or a[i] < -0.001):
                plt.scatter(x[0,i], y[i], c="#00ff00", marker="<")
        print('time for predict:', datetime.now() - start)

        plt.plot(x[0,:], z, c="#00ff00", label='svr fit')
        plt.scatter(0, 0, c="#00ff00", marker="<", label="support vector, all color")
        plt.scatter(0, 0, c="#ffffff", marker="x")
        plt.title('support vector regression curve')
        plt.xlabel('y')
        plt.ylabel('x')
        plt.legend(loc='upper right')
        plt.show()

    def plot3D(f, a, x, y):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(x[0,:], x[0,:], y, c="#000000", alpha=0.5, label='ori curve')

        z = np.zeros(x.shape[1])
        start = datetime.now()
        for i in range(x.shape[1]):
            z[i] = f(x[:,i])
            if (a[i] > 0.001 or a[i] < -0.001):
                ax.scatter(x[0,i], x[1,i], y[i], c="#00ff00", marker="<")
        print('time for predict:', datetime.now() - start)

        # setNum = 10000
        # xT = np.array([np.arange(x_min, x_max, (x_max - x_min) / int(np.sqrt(setNum)))])
        # x1, x2 = np.meshgrid(xT,xT)
        # x = np.array(np.c_[x1.ravel(), x2.ravel()].T)

        ax.plot_surface(x[0,:].reshape(gridShape), x[1,:].reshape(gridShape), z.reshape(gridShape),rstride=1,cstride=1,cmap=plt.cm.spring)
        # ax.plot(x[0,:], x[1,:], z, c="#00ff00", label='svr fit')
        ax.scatter(0, 0, 0, c="#00ff00", marker="<", label="support vector, all color")
        ax.scatter(0, 0, 0, c="#ffffff", marker="x")
        plt.title('support vector regression curve')
        ax.set_xlabel('x_1')
        ax.set_ylabel('x_2')
        ax.set_zlabel('f(x_1,x_2)')
        plt.legend(loc='upper right')
        plt.show()

    if (D3):
        plot3D(f,a,x,y)
    else:
        plot2D(f,a,x,y)


## gamma ++ -> overfit; gamma -- -> lessfit --------------------------------------
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='cos', k=12*3.14, noiseAmp=0.4)), ker.kernel('rbf', 64))
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='cos', k=12*3.14, noiseAmp=0.4)), ker.kernel('rbf', 16))
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='cos', k=3*3.14, noiseAmp=0.4)), ker.kernel('rbf', 64))
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='cos', k=3*3.14, noiseAmp=0.4)), ker.kernel('rbf', 16))

## test for log, sinc,  poly, spline -----------------------
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='log', noiseAmp=0.4)), ker.kernel('rbf', 16))
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='linear', noiseAmp=0.4)), ker.kernel('rbf', 16))
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='sinc', k=8*3.14, noiseAmp=0.4)), ker.kernel('rbf', 16))
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='cos', k=3*3.14, noiseAmp=0.4)), ker.kernel('ploy', 4))
# testSVR(100, 4000, 0.1, (lambda : dataSet(dataType='cos', k=3*3.14, noiseAmp=0.4)), ker.kernel('spline', 4)) # too much time

## test for C  C++ -> fit more ---------------------------------
# testSVR(1, 8000, 0.1, (lambda : dataSet(dataType='sinc', k=8*3.14, noiseAmp=0.4)), ker.kernel('rbf', 32))
# testSVR(10, 8000, 0.1, (lambda : dataSet(dataType='sinc', k=8*3.14, noiseAmp=0.4)), ker.kernel('rbf', 32))
# testSVR(100, 8000, 0.1, (lambda : dataSet(dataType='sinc', k=8*3.14, noiseAmp=0.4)), ker.kernel('rbf', 32))

## test for epsilon
# testSVR(10, 8000, 10, (lambda : dataSet(dataType='sinc', k=8*3.14, noiseAmp=0.4)), ker.kernel('rbf', 32))
# testSVR(10, 8000, 1, (lambda : dataSet(dataType='sinc', k=8*3.14, noiseAmp=0.4)), ker.kernel('rbf', 32))
# testSVR(10, 8000, 0.1, (lambda : dataSet(dataType='sinc', k=8*3.14, noiseAmp=0.4)), ker.kernel('rbf', 32))

## test for 3D --------------------------
# testSVR(10, 4000, 1, (lambda : dataSet(dataType='sinc3D', k=8*3.14, noiseAmp=0.4, setNum=4000,)), ker.kernel('rbf', 32), D3=True)
