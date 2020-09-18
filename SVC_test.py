import threading
import numpy as np
import SVCs as svc
import kernels as ker
from datetime import datetime
import matplotlib.pyplot as plt

## rand_linear(1,-1), rand_inverse(1,-1), rand_circle(1,-1),
## rand_quarterLinear(1,-1), rand_quarterInverse(1,-1), rand_circle3c(0,1,2) 
## iris(0,1,2), wine(0,1,2), cancer(1,-1) 
def dataSet(dataType, xDim=2, setNum=0):
    if (dataType[0:4] == 'rand'):
        setNum = 128 if (setNum == 0) else setNum
        x = np.random.rand(2, setNum)
        y = np.zeros(setNum)
        if (dataType == 'rand_linear'):
            for i in range(setNum):
                y[i] = (lambda x: 1 if (x[0] + x[1] > 1) else -1)(x[:, i])
        elif (dataType == 'rand_inverse'):
            for i in range(setNum):
                y[i] = (lambda x: 1 if (x[0] * x[1] > 0.28) else -1)(x[:, i])
        elif (dataType == 'rand_quarterLinear'):
            for i in range(setNum):
                y[i] = (lambda x: 1 if (np.abs((x[0]-0.5)) + np.abs((x[1]-0.5)) < 0.48) else -1)(x[:, i])
        elif (dataType == 'rand_quarterInverse'):
            for i in range(setNum):
                y[i] = (lambda x: 1 if (np.abs((x[0]-0.5) * (x[1]-0.5)) < 0.04) else -1)(x[:, i])
        elif (dataType == 'rand_circle'):
            for i in range(setNum):
                y[i] = (lambda x: 1 if (np.abs((x[0]-0.5) ** 2 + (x[1]-0.5) ** 2) < 0.064) else -1)(x[:, i])
        elif (dataType == 'rand_circle3c'):
            for i in range(setNum):
                r2 = (x[0, i]-0.5) ** 2 + (x[1, i]-0.5) ** 2
                if (r2 < 0.04):
                    y[i] = 0
                elif (r2 < 0.16):
                    y[i] = 1
                else:
                    y[i] = 2
        else:
            print('currently you can choose linear, inverse, quarterLinear, quarterInverse, circle, circle3c')
    else: 
        if (dataType == 'iris'):
            filepath='dataset/iris.csv'
        elif (dataType == 'wine'):
            filepath='dataset/wine_data.csv'
        elif (dataType == 'cancer'):
            filepath='dataset/breast_cancer.csv'
        else:
            print('currently you can choose iris, wine, cancer')
        data = np.loadtxt(filepath,dtype=float,delimiter=',',skiprows=1)
        setNum = data.shape[0] if (setNum == 0) else setNum
        xDim = data.shape[1] - 1 if (xDim == 0) else xDim
        y = data[0:setNum, -1]
        x = data[0:setNum, 0:xDim]
        # normalizing x ---------------
        xMax = np.tile(np.max(x, axis=0), (setNum, 1))
        xMin = np.tile(np.min(x, axis=0), (setNum, 1))
        x = (x-xMin) / (xMax-xMin)
        x = x.T

    if (dataType == 'cancer'):
        y = 2*(y-0.5)

    return x, y

### when imported differnt svcs, using this to match
def svcAPI(x, y, kernel, C, eta, maxIter, svcType):
    if (svcType == 'basic'):
        classifier, plane, a, b = svc.basicSVM(x, y, kernel, C, eta, maxIter)
        return  classifier, plane, a, b
    elif (svcType == 're_basic'):
        classifier, plane, a, b = svc.reBasicSVM(x, y, kernel, C, eta, maxIter)
        return  classifier, plane, a, b
    elif (svcType == 're_basic_kCache'):
        classifier, plane, a, b = svc.reBasicSVM_kCache(x, y, kernel, C, eta, maxIter)
        return  classifier, plane, a, b
    elif (svcType == 'adaBoost'):
        classifier, plane = svc.adaBoostSVM(x, y, kernel, C, eta, maxIter, adaIter = 4)
        return  classifier, plane, 0, 0
    elif (svcType == 'logistic'):
        classifier, plane = svc.logisticSVMn(x, y, kernel, C, eta, maxIter, plane_min = 0, plane_max = 3, splineNum = 5)
        return  classifier, plane, 0, 0
    else:
        classifier, plane, a, b = svc.reBasicSVM(x, y, kernel, C, eta, maxIter)
        return  classifier, plane, a, b

## used for multi classify problem
def svcMulti(x, y, kernel, C, eta, maxIter, svcType):
    planeSet = []
    threadList = []

    def svcOnce(x, y, targetLabel, kernel, C, eta, maxIter, planeSet):
        newLabel = np.vectorize(lambda y: 1 if (y == targetLabel) else -1)(y)
        _, f, _, _ = svcAPI(x, newLabel, kernel, C, eta, maxIter, svcType)
        planeSet.append({'plane': f, 'label': targetLabel})

    # # not using thread
    # for s in list(set(y)):
    #     print('---------- train for label %d ---------' % s)
    #     svcOnce(x, y, s, kernel, C, eta, maxIter, planeSet)

    # using thread
    for s in list(set(y)):
        print('---------- train for label %d ---------' % s)
        task = threading.Thread(target=svcOnce, args=(x, y, s, kernel, C, eta, maxIter, planeSet))
        threadList.append(task)
        task.start()
    for i in range(len(threadList)):
        threadList[i].join()

    classifier = (lambda x: planeSet[int(np.argmax([planeSet[i]['plane'](x) for i in range(len(planeSet))]))]['label'])
    return classifier

def testSVC(C, maxIter, kernel, dataSet, svcType='re_basic', multi=False, showSV=False, showLine=False):
    x, y = dataSet()
    start = datetime.now()
    if(not multi):
        classifier, plane, a, b = svcAPI(x, y, kernel, C, 0.0001, maxIter, svcType)
    else:
        classifier = svcMulti(x, y, kernel, C, 0.0001, maxIter, svcType)
    print('time for train:', datetime.now() - start)

    # should be used before plotSV as an addition
    def plotLine(plane, b):
        poi0 = [0, -b / (plane(np.array([0, 1])) - b) ]
        poi1 = [-b / (plane(np.array([1, 0])) - b), 0]
        poi2 = [0, (1-b) / (plane(np.array([0, 1])) - b) ]
        poi3 = [(1-b) / (plane(np.array([1, 0])) - b), 0]
        poi4 = [0, (-1-b) / (plane(np.array([0, 1])) - b) ]
        poi5 = [(-1-b) / (plane(np.array([1, 0])) - b), 0]
        plt.plot([poi0[0], poi1[0]], [poi0[1], poi1[1]], c="#000000", label='f(x) = 0')
        plt.plot([poi2[0], poi3[0]], [poi2[1], poi3[1]], c="#0000ff", label='f(x) = 1')
        plt.plot([poi4[0], poi5[0]], [poi4[1], poi5[1]], c="#ff0000", label='f(x) = -1')

    # must have plt.figure() before and plt.show() after
    def plotSV(classifier, x, y, a, C):
        errors = 0
        setNum = x.shape[1]
        o = np.zeros(setNum)
        for i in range(setNum):
            o[i] = classifier(x[:, i])
            if (y[i] != o[i]):
                errors += 1
            c = "#000000"
            marker = 'o'
            if (y[i] == 1 and o[i] == 1):
                c = "#0000ff"
            elif (y[i] == -1 and o[i] == -1):
                c = "#ff0000"
            elif (y[i] == 1 and o[i] == -1):
                c = "#00ffff"
            elif (y[i] == -1 and o[i] == 1):
                c = "#00ff00"
            if (a[i] > 0 and a[i] < C):
                marker = '<'
            elif (a[i] == C):
                marker = 'x'
            plt.scatter(x[0, i], x[1, i], c=c, marker=marker)
        plt.scatter(0, 0, c="#ff0000", marker="o", label="sort right for -1")
        plt.scatter(0, 0, c="#0000ff", marker="o", label="sort right for 1")
        plt.scatter(0, 0, c="#00ff00", marker="o", label="sort wrong for -1")
        plt.scatter(0, 0, c="#00ffff", marker="o", label="sort wrong for 1")
        plt.scatter(0, 0, c="#000000", marker="<", label="support vector")
        plt.scatter(0, 0, c="#000000", marker="x", label="inside margin vector")
        plt.title('errorRate = %f' % (errors / setNum))
        plt.legend(loc='upper right')

    # must have plt.figure() before and plt.show() after
    def plotArea(x_min, x_max, y_min, y_max, testSet, testLabel, classifier):
        x_step = (x_max - x_min) / 100
        y_step = (y_max - y_min) / 100
        x_max += x_step
        y_max += y_step
        xx, yy = np.meshgrid(np.arange(x_min, x_max, x_step),
                        np.arange(y_min, y_max, y_step))
        inputVec = np.array(np.c_[xx.ravel(), yy.ravel()].T)
        setNum = inputVec.shape[1]
        labels = np.zeros(setNum)
        
        start = datetime.now()
        for i in range(setNum):
            labels[i] = classifier(inputVec[:,i])
        print('time for predict:', datetime.now() - start)

        errorRate = 0
        for i in range(len(testLabel)):
            outLabel = classifier(testSet[:,i])
            if (outLabel != testLabel[i]):
                errorRate += 1
        errorRate /= len(testLabel)

        from matplotlib.colors import ListedColormap
        plt.contourf(xx, yy, labels.reshape(xx.shape), alpha=0.2, cmap=ListedColormap(['r', 'y', 'b']))
        plt.scatter(testSet[0, :], testSet[1, :], c = testLabel, cmap=ListedColormap(['r', 'y', 'b']))
        plt.xlabel('_')
        plt.ylabel('_')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.title('errorRate = %f' % errorRate)

    plt.figure()
    if ((not multi) and (showSV or showLine)):    
        plt.subplot(1,2,1)
        if (showLine):
            plotLine(plane, b)
        plotSV(classifier, x, y, a, C)
        plt.subplot(1,2,2)
    
    plotArea(0,1,0,1,x,y,classifier)
    plt.show()

## get linear spline curve parameters fit for logistic loss function
def logisticSpline(x_min, x_max, n):
    # logistic: ln(1+exp(-yf)) = ln(1+exp(-x))
    # linear spline: \Sigma c_k (d_k - x)_+
    sampleX = np.arange(x_min, x_max, (x_max - x_min) / n)
    sampleY = np.vectorize(lambda x: np.log(1+np.exp(-x)))(sampleX)
    d = sampleX
    c = sampleY
    for k in range(n)[::-1]:
        if (k == 0):
            c[k] = 0
        elif (k == n-1):
            c[k] = c[k-1] / (d[k] - d[k-1])
        else:
            tmp = (lambda x: np.sum([c[m]*np.max([d[m] - x, 0]) for m in range(k+1,n)]))(d[k-1])
            c[k] = (sampleY[k-1] - tmp) / (d[k] - d[k-1])
    
    spline = (lambda x : np.sum([c[k]*np.max([d[k] - x, 0]) for k in range(len(d))]))
    x = np.arange(x_min, x_max, (x_max - x_min) / 100)
    y = np.vectorize(lambda x: np.log(1+np.exp(-x)))(x)
    z = np.vectorize(spline)(x)
    plt.figure()
    plt.plot(x, y, c="#000000", label='logistic')
    plt.plot(x, z, c="#00ff00", label='spline')
    plt.title('logistic regression carve')
    plt.xlabel('y')
    plt.ylabel('x')
    plt.legend(loc='upper right')
    plt.show()
    return c, d

# print(logisticSpline(-4,4,4))

## solid data test to compare the capability of basic/ re_basic/ re_basic_kCache --------------------
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('cancer', setNum=96)), showSV=True, svcType='basic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('cancer', setNum=96)), showSV=True, svcType='re_basic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('cancer', setNum=96)), showSV=True, svcType='re_basic_kCache')


## basic to test if this svc works ---------------------------
# testSVC(100, 4000, ker.kernel('linear'), (lambda : dataSet('rand_linear')), showLine=True)
# testSVC(100, 4000, ker.kernel('linear'), (lambda : dataSet('rand_inverse')), showLine=True)
# testSVC(100, 4000, ker.kernel('rbf', 1), (lambda : dataSet('rand_inverse')), showSV=True)
# testSVC(100, 4000, ker.kernel('rbf', 1), (lambda : dataSet('rand_circle')), showSV=True)
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_quarterLinear')), showSV=True)
# testSVC(100, 4000, ker.kernel('poly', 5), (lambda : dataSet('rand_quarterLinear')), showSV=True)
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_quarterInverse')), showSV=True)
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('cancer')), showSV=True)
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('cancer', setNum=128)), showSV=True)
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_circle3c')), multi=True)
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('iris')), multi=True)
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('wine')), multi=True)

## adaBoost to test if this svc works ------------------------
# testSVC(100, 4000, ker.kernel('linear'), (lambda : dataSet('rand_linear')), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('linear'), (lambda : dataSet('rand_inverse')), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 1), (lambda : dataSet('rand_inverse')), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 1), (lambda : dataSet('rand_circle')), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_quarterLinear')), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('poly', 5), (lambda : dataSet('rand_quarterLinear')), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_quarterInverse')), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('cancer', setNum=128)), svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_circle3c')), multi=True, svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('iris')), multi=True, svcType='adaBoost')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('wine')), multi=True, svcType='adaBoost')


## logistic to test if this svc works ------------------------
testSVC(100, 4000, ker.kernel('linear'), (lambda : dataSet('rand_linear')), svcType='logistic')
# testSVC(100, 4000, ker.kernel('linear'), (lambda : dataSet('rand_inverse')), svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 1), (lambda : dataSet('rand_inverse')), svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 1), (lambda : dataSet('rand_circle')), svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_quarterLinear')), svcType='logistic')
# testSVC(100, 4000, ker.kernel('poly', 5), (lambda : dataSet('rand_quarterLinear')), svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_quarterInverse')), svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('cancer', setNum=96)), svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('rand_circle3c')), multi=True, svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('iris')), multi=True, svcType='logistic')
# testSVC(100, 4000, ker.kernel('rbf', 10), (lambda : dataSet('wine')), multi=True, svcType='logistic')
