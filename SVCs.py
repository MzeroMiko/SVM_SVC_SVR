import numpy as np

def basicSVM(x, y, kernel, C, eta, maxIter):
    # x : ndarray inputDim * setNum
    # y : ndarray setNum
    # a : ndarray setNum
    # eMap[i] = wx[i] + b - y[i]

    def smoUpdateAB(yI, yJ, aI, aJ, aC, KII, KIJ, KJJ, EI, EJ):
        ## get L H ---------------------------
        if (yI != yJ):
            L = np.max((0, aJ - aI))
            H = np.min(( aC,  aJ - aI + aC))
        else:
            L = np.max((0, aJ + aI -  aC))
            H = np.min(( aC, aJ + aI))
        if (L >= H):
            # print("L >= H", L, H, yI * yJ)
            return 0, 0, 0
        ## get dAI dAJ ------------------------
        dK = (KII + KJJ - 2 * KIJ)
        if (dK == 0):
            # print("KII + KJJ - 2 * KIJ == 0")
            return 0, 0, 0
        aJNew = np.max((L, np.min((H, aJ + yJ * (EI - EJ) / dK))))
        dAJ = aJNew - aJ
        dAI = dAJ if (yI != yJ) else -dAJ
        aINew = aI + dAI
        ## get dB --------------------------
        if (aINew > 0 and aINew < aC):
            dB = -EI - dAI * yI * KII - dAJ * yJ * KIJ
        elif (aJNew > 0 and aJNew < aC):
            dB = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
        else:
            dBI = -EI - dAI * yI * KII - dAJ * yJ * KIJ
            dBJ = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
            dB = (dBI + dBJ) / 2
        return dAI, dAJ, dB

    ## init ----------------------------------------
    setNum = np.shape(x)[1]
    b = 0
    a = np.zeros(setNum)
    eMap = np.zeros(setNum) - y # cuz y is int, eMap is float
    ## choose a_1, a_2 ------------------------------
    iter = 0
    scanEntireSet = True
    while (iter < maxIter):
        changed = False
        for i in range(setNum):
            ## choose a_1 ----------------------------
            tmp = y[i] * eMap[i]
            if ((a[i] > 0 and a[i] < C) and (tmp > eta or tmp < -eta)):
                pass
            elif (scanEntireSet):
                if ((a[i] == 0) and (tmp < 0)) or ((a[i] == C) and (tmp > 0)):
                    pass
                else:
                    continue
            else:
                continue
            ## choose a_2 --------------------------- 
            # sort from big to small
            jRank = list(np.argsort(-np.abs(eMap - eMap[i])))
            for j in jRank:
                if (j == i):
                    continue
                ## calc dAI dAJ dB -----------------------
                dAI, dAJ, dB = smoUpdateAB(y[i], y[j], a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j])
                if (dAI == 0):
                    continue
                changed = True
                iter += 1
                ## update a b ----------------------------
                a[i] += dAI
                a[j] += dAJ
                b += dB
                ## update E ---------------------------
                dAYI = dAI * y[i]
                dAYJ = dAJ * y[j]
                for k in range(setNum):
                    eMap[k] += dB + dAYI * kernel(x[:,i], x[:,k]) + dAYJ * kernel(x[:,j], x[:,k])
                break
        
        if ((not scanEntireSet) and (not changed)):
            scanEntireSet = True
        elif (scanEntireSet and changed):
            scanEntireSet = False
        elif (scanEntireSet and (not changed)):
            break
    
    print('iteration %d finish' % iter)

    svAIndex = list(np.nonzero(a)[0])
    aY = np.multiply(a, y)
    plane = (lambda input: np.sum([aY[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
    classifier = (lambda input: 1 if (plane(input) > 0) else -1)
    return classifier, plane, a, b


def reBasicSVM(x, y, kernel, C, eta, maxIter):
    # x : ndarray inputDim * setNum
    # y : ndarray setNum
    # a : ndarray setNum
    # eMap[i] = wx[i] + b - y[i]

    def smoUpdateAB(yI, yJ, aI, aJ, aC, KII, KIJ, KJJ, EI, EJ):
        ## get L H ---------------------------
        if (yI != yJ):
            L = np.max((0, aJ - aI))
            H = np.min(( aC,  aJ - aI + aC))
        else:
            L = np.max((0, aJ + aI -  aC))
            H = np.min(( aC, aJ + aI))
        if (L >= H):
            # print("L >= H", L, H, yI * yJ)
            return 0, 0, 0
        ## get dAI dAJ ------------------------
        dK = (KII + KJJ - 2 * KIJ)
        if (dK == 0):
            # print("KII + KJJ - 2 * KIJ == 0")
            return 0, 0, 0
        aJNew = np.max((L, np.min((H, aJ + yJ * (EI - EJ) / dK))))
        dAJ = aJNew - aJ
        dAI = dAJ if (yI != yJ) else -dAJ
        aINew = aI + dAI
        ## get dB --------------------------
        if (aINew > 0 and aINew < aC):
            dB = -EI - dAI * yI * KII - dAJ * yJ * KIJ
        elif (aJNew > 0 and aJNew < aC):
            dB = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
        else:
            dBI = -EI - dAI * yI * KII - dAJ * yJ * KIJ
            dBJ = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
            dB = (dBI + dBJ) / 2
        return dAI, dAJ, dB

    def smoSelectI(a, aC, y, eMap, eta):
        '''
        target:
            find ((a[i] > 0 and a[i] < C) and (tmp > eta or tmp < -eta))
            or ((a[i] == 0) and (tmp < 0)) or ((a[i] == C) and (tmp > 0))
        '''
        tmp = y * eMap
        setRank = list(np.argsort(tmp)) # sort from negative to positive
        iNegAll, iNegSV = -1, -1
        iPosAll, iPosSV = -1, -1
        # search Negative ------------------------
        for i in setRank:
            # in tmp < -eta
            if (tmp[i] > -eta):
                break
            # find (a[i] == 0) or (a[i] > 0 and a[i] < C)
            if (a[i] < aC):
                if (iNegAll == -1):
                    iNegAll = i 
                # find (a[i] > 0 and a[i] < C)
                if (a[i] > 0):
                    iNegSV = i 
                    break
        # search Positive ------------------------
        for i in setRank[::-1]:
            if (tmp[i] < eta):
                break
            if (a[i] > 0):
                if (iPosAll == -1):
                    iPosAll = i
                if (a[i] < aC):
                    iPosSV = i
                    break   
        ## get posI ------------------------------
        searchEnd = False
        scanEntireSet = False
        if (iPosSV != -1 and iNegSV != -1):
            posI = iPosSV if (tmp[iPosSV] > -tmp[iNegSV]) else iNegSV
        elif (iPosSV != -1):
            posI = iPosSV
        elif (iNegSV != -1):
            posI = iNegSV
        else:
            scanEntireSet = True
            if (iPosAll != -1 and iNegAll != -1):
                posI = iPosAll if (tmp[iPosAll] > -tmp[iNegAll]) else iNegAll
            elif (iPosAll != -1):
                posI = iPosAll
            elif (iNegAll != -1):
                posI = iNegAll
            else:
                searchEnd = True
                posI = -1
        ## get posJ Rank -----------------------
        jRank = list(np.argsort(eMap))
        jRank = jRank[::-1] if (eMap[posI] < 0) else jRank
        return posI, searchEnd, scanEntireSet, jRank

    ## init ----------------------------------------
    setNum = np.shape(x)[1]
    b = 0
    a = np.zeros(setNum)
    eMap = np.zeros(setNum) - y # cuz y is int, eMap is float
    ## choose a_1, a_2 ------------------------------
    iter = 0
    changed = True
    while (iter < maxIter and changed):
        changed = False
        i, searchEnd, scanEntireSet, jRank = smoSelectI(a, C, y, eMap, eta)
        if (searchEnd):
            break
        for j in jRank:
            if (j == i):
                continue
            ## calc dAI dAJ dB -----------------------
            dAI, dAJ, dB = smoUpdateAB(y[i], y[j], a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j])
            if (dAI == 0):
                continue
            changed = True
            iter += 1
            ## update a b ----------------------------
            a[i] += dAI
            a[j] += dAJ
            b += dB
            ## update E ---------------------------
            dAYI = dAI * y[i]
            dAYJ = dAJ * y[j]
            for k in range(setNum):
                eMap[k] += dB + dAYI * kernel(x[:,i], x[:,k]) + dAYJ * kernel(x[:,j], x[:,k])
            break
        
        if (scanEntireSet and (not changed)):
            break
    
    print('iteration %d finish' % iter)

    svAIndex = list(np.nonzero(a)[0])
    aY = np.multiply(a, y)
    plane = (lambda input: np.sum([aY[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
    classifier = (lambda input: 1 if (plane(input) > 0) else -1)
    return classifier, plane, a, b


def reBasicSVM_kCache(x, y, kernel, C, eta, maxIter):
    # x : ndarray inputDim * setNum
    # y : ndarray setNum
    # a : ndarray setNum
    # eMap[i] = wx[i] + b - y[i]

    def smoUpdateAB(yI, yJ, aI, aJ, aC, KII, KIJ, KJJ, EI, EJ):
        ## get L H ---------------------------
        if (yI != yJ):
            L = np.max((0, aJ - aI))
            H = np.min(( aC,  aJ - aI + aC))
        else:
            L = np.max((0, aJ + aI -  aC))
            H = np.min(( aC, aJ + aI))
        if (L >= H):
            # print("L >= H", L, H, yI * yJ)
            return 0, 0, 0
        ## get dAI dAJ ------------------------
        dK = (KII + KJJ - 2 * KIJ)
        if (dK == 0):
            # print("KII + KJJ - 2 * KIJ == 0")
            return 0, 0, 0
        aJNew = np.max((L, np.min((H, aJ + yJ * (EI - EJ) / dK))))
        dAJ = aJNew - aJ
        dAI = dAJ if (yI != yJ) else -dAJ
        aINew = aI + dAI
        ## get dB --------------------------
        if (aINew > 0 and aINew < aC):
            dB = -EI - dAI * yI * KII - dAJ * yJ * KIJ
        elif (aJNew > 0 and aJNew < aC):
            dB = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
        else:
            dBI = -EI - dAI * yI * KII - dAJ * yJ * KIJ
            dBJ = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
            dB = (dBI + dBJ) / 2
        return dAI, dAJ, dB

    def smoSelectI(a, aC, y, eMap, eta):
        '''
        target:
            find ((a[i] > 0 and a[i] < C) and (tmp > eta or tmp < -eta))
            or ((a[i] == 0) and (tmp < 0)) or ((a[i] == C) and (tmp > 0))
        '''
        tmp = y * eMap
        setRank = list(np.argsort(tmp)) # sort from negative to positive
        iNegAll, iNegSV = -1, -1
        iPosAll, iPosSV = -1, -1
        # search Negative ------------------------
        for i in setRank:
            # in tmp < -eta
            if (tmp[i] > -eta):
                break
            # find (a[i] == 0) or (a[i] > 0 and a[i] < C)
            if (a[i] < aC):
                if (iNegAll == -1):
                    iNegAll = i 
                # find (a[i] > 0 and a[i] < C)
                if (a[i] > 0):
                    iNegSV = i 
                    break
        # search Positive ------------------------
        for i in setRank[::-1]:
            # in tmp > eta
            if (tmp[i] < eta):
                break
            # find (a[i] == C) or (a[i] > 0 and a[i] < C)
            if (a[i] > 0):
                if (iPosAll == -1):
                    iPosAll = i
                # find (a[i] > 0 and a[i] < C)
                if (a[i] < aC):
                    iPosSV = i
                    break   

        ## get posI ------------------------------
        searchEnd = False
        scanEntireSet = False
        if (iPosSV != -1 and iNegSV != -1):
            posI = iPosSV if (tmp[iPosSV] > -tmp[iNegSV]) else iNegSV
        elif (iPosSV != -1):
            posI = iPosSV
        elif (iNegSV != -1):
            posI = iNegSV
        else:
            scanEntireSet = True
            if (iPosAll != -1 and iNegAll != -1):
                posI = iPosAll if (tmp[iPosAll] > -tmp[iNegAll]) else iNegAll
            elif (iPosAll != -1):
                posI = iPosAll
            elif (iNegAll != -1):
                posI = iNegAll
            else:
                searchEnd = True
                posI = -1
        ## get posJ Rank -----------------------
        jRank = list(np.argsort(eMap))
        jRank = jRank[::-1] if (eMap[posI] < 0) else jRank
        return posI, searchEnd, scanEntireSet, jRank

    ## init ----------------------------------------
    setNum = np.shape(x)[1]
    b = 0
    a = np.zeros(setNum)
    eMap = np.zeros(setNum) - y # cuz y is int, eMap is float
    kList = []
    keep = 4 
    ## choose a_1, a_2 ------------------------------
    iter = 0
    changed = True
    while (iter < maxIter and changed):
        changed = False
        i, searchEnd, scanEntireSet, jRank = smoSelectI(a, C, y, eMap, eta)
        if (searchEnd):
            break
        for j in jRank:
            if (j == i):
                continue
            # print(i, j)
            ## calc dAI dAJ dB -----------------------
            KII, KIJ, KJJ = 0, 0, 0
            for kl in kList:
                if (KII != 0 and KIJ != 0 and KJJ != 0):
                    break
                if (kl[0] == i and kl[1] == i):
                    KII = kl[2]
                    kl[3] = keep
                elif (kl[0] == i and kl[1] == j) or (kl[0] == j and kl[1] == i):
                    KIJ = kl[2]
                    kl[3] = keep
                elif (kl[0] == j and kl[1] == j):
                    KJJ = kl[2]
                    kl[3] = keep
                elif (kl[3] == 0):
                    kList.remove(kl)
                else:
                    kl[3] -= 1

            if (KII == 0):
                KII = kernel(x[:,i], x[:,i])
                if (KII != 0):
                    kList.append([i,i,KII, keep])
            if (KIJ == 0):
                KIJ = kernel(x[:,i], x[:,j])
                if (KIJ != 0):
                    kList.append([i,j,KIJ, keep])
            if (KJJ == 0):
                KJJ = kernel(x[:,j], x[:,j])
                if (KJJ != 0):
                    kList.append([j,j,KJJ, keep])
   
            dAI, dAJ, dB = smoUpdateAB(y[i], y[j], a[i], a[j], C, KII, KIJ, KJJ, eMap[i], eMap[j])
            if (dAI == 0):
                continue
            changed = True
            iter += 1
            ## update a b ----------------------------
            a[i] += dAI
            a[j] += dAJ
            b += dB
            ## update E ---------------------------
            dAYI = dAI * y[i]
            dAYJ = dAJ * y[j]
            KI = np.zeros(setNum)
            KJ = np.zeros(setNum)
            for kl in kList:
                if (kl[0] == i):
                    KI[kl[1]] = kl[2]
                    kl[3] = keep
                elif (kl[1] == i):
                    KI[kl[0]] = kl[2]
                    kl[3] = keep
                elif (kl[0] == j):
                    KJ[kl[1]] = kl[2]
                    kl[3] = keep
                elif (kl[1] == j):
                    KJ[kl[0]] = kl[2]
                    kl[3] = keep
                elif (kl[3] == 0):
                    kList.remove(kl)
                else:
                    kl[3] -= 1

            for k in range(setNum):
                if (KI[k] == 0):
                    KI[k] = kernel(x[:,i], x[:,k])
                    if (KI[k] != 0):
                        kList.append([i,k,KI[k],keep])
                if (KJ[k] == 0):
                    KJ[k] = kernel(x[:,j], x[:,k])
                    if (KJ[k] != 0):
                        kList.append([j,k,KJ[k],keep])

            eMap += dB + np.multiply(dAYI, KI) + np.multiply(dAYJ, KJ)
            break
        
        if (scanEntireSet and (not changed)):
            break
    
    print('iteration %d finish' % iter)

    svAIndex = list(np.nonzero(a)[0])
    aY = np.multiply(a, y)
    plane = (lambda input: np.sum([aY[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
    classifier = (lambda input: 1 if (plane(input) > 0) else -1)
    return classifier, plane, a, b


def adaBoostSVM(x, y, kernel, C, eta, maxSVMIter, adaIter):
    
    ## smo update used for svm 
    def smoUpdateAB(yI, yJ, aI, aJ, aCI, aCJ, KII, KIJ, KJJ, EI, EJ):
        ## get L H ---------------------------
        if (yI != yJ):
            L = np.max((0, aJ - aI))
            H = np.min(( aCJ,  aJ - aI + aCI))
        else:
            L = np.max((0, aJ + aI -  aCI))
            H = np.min(( aCJ, aJ + aI))
        if (L >= H):
            # print("L >= H", L, H, yI * yJ)
            return 0, 0, 0
        ## get dAI dAJ ------------------------
        dK = (KII + KJJ - 2 * KIJ)
        if (dK == 0):
            # print("KII + KJJ - 2 * KIJ == 0")
            return 0, 0, 0
        aJNew = np.max((L, np.min((H, aJ + yJ * (EI - EJ) / dK))))
        dAJ = aJNew - aJ
        dAI = dAJ if (yI != yJ) else -dAJ
        aINew = aI + dAI
        ## get dB --------------------------
        if (aINew > 0 and aINew < aCI):
            dB = -EI - dAI * yI * KII - dAJ * yJ * KIJ
        elif (aJNew > 0 and aJNew < aCJ):
            dB = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
        else:
            dBI = -EI - dAI * yI * KII - dAJ * yJ * KIJ
            dBJ = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
            dB = (dBI + dBJ) / 2
        return dAI, dAJ, dB

    def smoSelectI(a, aC, y, eMap, eta):
        '''
        target:
            find ((a[i] > 0 and a[i] < C[i]) and (tmp > eta or tmp < -eta))
            or ((a[i] == 0) and (tmp < 0)) or ((a[i] == C[i]) and (tmp > 0))
        '''
        tmp = y * eMap
        setRank = list(np.argsort(tmp)) # sort from negative to positive
        iNegAll, iNegSV = -1, -1
        iPosAll, iPosSV = -1, -1
        # search Negative ------------------------
        for i in setRank:
            # in tmp < -eta
            if (tmp[i] > -eta):
                break
            # find (a[i] == 0) or (a[i] > 0 and a[i] < C)
            if (a[i] < aC[i]):
                if (iNegAll == -1):
                    iNegAll = i 
                # find (a[i] > 0 and a[i] < C)
                if (a[i] > 0):
                    iNegSV = i 
                    break
        # search Positive ------------------------
        for i in setRank[::-1]:
            if (tmp[i] < eta):
                break
            if (a[i] > 0):
                if (iPosAll == -1):
                    iPosAll = i
                if (a[i] < aC[i]):
                    iPosSV = i
                    break   

        ## get posI ------------------------------
        searchEnd = False
        scanEntireSet = False
        if (iPosSV != -1 and iNegSV != -1):
            posI = iPosSV if (tmp[iPosSV] > -tmp[iNegSV]) else iNegSV
        elif (iPosSV != -1):
            posI = iPosSV
        elif (iNegSV != -1):
            posI = iNegSV
        else:
            scanEntireSet = True
            if (iPosAll != -1 and iNegAll != -1):
                posI = iPosAll if (tmp[iPosAll] > -tmp[iNegAll]) else iNegAll
            elif (iPosAll != -1):
                posI = iPosAll
            elif (iNegAll != -1):
                posI = iNegAll
            else:
                searchEnd = True
                posI = -1
        ## get posJ Rank -----------------------
        jRank = list(np.argsort(eMap))
        jRank = jRank[::-1] if (eMap[posI] < 0) else jRank
        return posI, searchEnd, scanEntireSet, jRank

    ## svm used for get classifier in single ada iteration
    def svm(x, y, kernel, C, eta, maxIter):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : ndarray setNum
        # C : C[i] = C^{ori} * c[i]
        # eMap[i] = wx[i] + b - y[i]
        ## init ----------------------------------------
        setNum = np.shape(x)[1]
        b = 0
        a = np.zeros(setNum)
        eMap = np.zeros(setNum) - y # cuz y is int, eMap is float
        ## choose a_1, a_2 ------------------------------
        iter = 0
        changed = True
        while (iter < maxIter and changed):
            changed = False
            i, searchEnd, scanEntireSet, jRank = smoSelectI(a, C, y, eMap, eta)
            if (searchEnd):
                break
            for j in jRank:
                if (j == i):
                    continue
                ## calc dAI dAJ dB ---------------------------
                dAI, dAJ, dB = smoUpdateAB(y[i], y[j], a[i], a[j], C[i], C[j], kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j])
                if (dAI == 0):
                    continue
                changed = True
                iter += 1
                ## update a b ----------------------------
                a[i] += dAI
                a[j] += dAJ
                b += dB
                ## update E ---------------------------
                dAYI = dAI * y[i]
                dAYJ = dAJ * y[j]
                for k in range(setNum):
                    eMap[k] += dB + dAYI * kernel(x[:,i], x[:,k]) + dAYJ * kernel(x[:,j], x[:,k])
                break
            
            if (scanEntireSet and (not changed)):
                break

        print('iteration %d finish' % iter)

        svAIndex = list(np.nonzero(a)[0])
        aY = np.multiply(a, y)
        plane = (lambda input: np.sum([aY[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        classifier = (lambda input: 1 if (plane(input) > 0) else -1)
        return classifier, plane, a, b

    sgnSet = []
    setNum = np.shape(x)[1]
    c = np.ones(setNum)
    iter = 0
    d = 1
    while (iter < adaIter):
        ## update sgnSet -------------
        iter += 1
        sgnPhi, _,_,_ = svm(x, y, kernel, C * c, eta, maxSVMIter)
        sgnSet.append({'sgnFunc':sgnPhi, 'weight': d})
        ## update c -------------------
        cPositive = 0
        cNegative = 0
        for i in range(setNum):
            tmp = y[i] * sgnPhi(x[:,i])
            c[i] = c[i] * np.exp(-d*tmp)
            if (tmp > 0):
                cPositive += c[i]
            else:
                cNegative += c[i]
        ## already finish(all are right)
        if (cNegative == 0):
            break
        ## update d -------------------
        d = np.log(cPositive / cNegative) / 2
    
    plane = (lambda x: np.sum([sgnSet[i]['weight'] * sgnSet[i]['sgnFunc'](x) for i in range(iter)])) 

    return (lambda x: 1 if(plane(x) > 0) else -1), plane


## wrong in smo algorithm
def logisticSVMn(x, y, kernel, C, eta, maxIter, plane_min, plane_max, splineNum):

    ## smo update used for svm 
    def smoUpdateAB(yI, yJ, aI, aJ, aCI, aCJ, KII, KIJ, KJJ, EI, EJ):
        ## yI: y_{i_1}, aI: a_{k_1, i_1}, aCI: C * c_{k_1}
        ## KIJ: K_{i_1, i_2}, EI: E_{k_1, i_1}
        ## get L H --------------------------------
        if (yI != yJ):
            L = np.max((0, aJ - aI))
            H = np.min((aCJ,  aJ - aI + aCI))
        else:
            L = np.max((0, aJ + aI -  aCI))
            H = np.min((aCJ, aJ + aI))
        if (L >= H):
            # print("L >= H", L, H, yI * yJ)
            return 0, 0, 0
        ## get dAI dAJ --------------------------- 
        dK = (KII + KJJ - 2 * KIJ)
        if (dK == 0):
            # print("KII + KJJ - 2 * KIJ == 0")
            return 0, 0, 0
        aJNew = np.max((L, np.min((H, aJ + yJ * (EI - EJ) / dK))))
        dAJ = aJNew - aJ
        dAI = dAJ if (yI != yJ) else -dAJ
        aINew = aI + dAI
        ## get dB -------------------------------
        if (aINew > 0 and aINew < aCI):
            dB = -EI - dAI * yI * KII - dAJ * yJ * KIJ
        elif (aJNew > 0 and aJNew < aCJ):
            dB = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
        else:
            dBI = -EI - dAI * yI * KII - dAJ * yJ * KIJ
            dBJ = -EJ - dAI * yI * KIJ - dAJ * yJ * KJJ
            dB = (dBI + dBJ) / 2
        return dAI, dAJ, dB

    def smoSelectI(a, aC, y, eMap, eta):
        '''
        target:
            find ((a[k, i] > 0 and a[k, i] < C[k]) and (tmp > eta or tmp < -eta))
            or ((a[k, i] == 0) and (tmp < 0)) or ((a[k, i] == C) and (tmp > 0))
        '''
        eMapRows = eMap.shape[0]
        eMapCols = eMap.shape[1]
        tmp = np.multiply(np.tile(y.reshape(1,-1), (eMapRows, 1)), eMap)
        setRank = list(np.argsort(tmp.reshape(1,-1)[0])) # sort from negative to positive
        iNegAll, kNegAll, iNegSV, kNegSV = -1, -1, -1, -1
        iPosAll, kPosAll, iPosSV, kPosSV = -1, -1, -1, -1
        # search Negative ------------------------
        for ki in setRank:
            k = int(np.floor(ki / eMapCols)) 
            i = ki % eMapCols 
            # in tmp < -eta
            if (tmp[k, i] > -eta):
                break
            # find (a[i] == 0) or (a[i] > 0 and a[i] < C)
            if (a[k,i] < aC[k]):
                if (iNegAll == -1):
                    iNegAll = i
                    kNegAll = k 
                # find (a[i] > 0 and a[i] < C)
                if (a[k,i] > 0):
                    iNegSV = i
                    kNegSV = k 
                    break
        # search Positive ------------------------
        for ki in setRank[::-1]:
            k = int(np.floor(ki / eMapCols)) 
            i = ki % eMapCols 
            if (tmp[k, i] < eta):
                break
            if (a[k,i] > 0):
                if (iPosAll == -1):
                    iPosAll = i
                    kPosAll = k
                if (a[k,i] < aC[k]):
                    iPosSV = i
                    kPosSV = k
                    break   
        ## get posI posK --------------------------
        searchEnd = False
        scanEntireSet = False
        if (iPosSV != -1 and iNegSV != -1):
            if (tmp[kPosSV, iPosSV] > -tmp[kNegSV, iNegSV]):
                posI = iPosSV
                posK = kPosSV
            else:
                posI = iNegSV
                posK = kNegSV
        elif (iPosSV != -1):
            posI = iPosSV
            posK = kPosSV
        elif (iNegSV != -1):
            posI = iNegSV
            posK = kNegSV
        else:
            scanEntireSet = True
            if (iPosAll != -1 and iNegAll != -1):
                if (tmp[kPosAll,iPosAll] > -tmp[kNegAll,iNegAll]):
                    posI = iPosAll  
                    posK = kPosAll  
                else:
                    posI = iNegAll
                    posK = kNegAll
            elif (iPosAll != -1):
                posI = iPosAll
                posK = kPosAll
            elif (iNegAll != -1):
                posI = iNegAll
                posK = kNegAll
            else:
                searchEnd = True
                posI = -1
                posK = -1
        ## get posJ Rank -----------------------
        qjRank = list(np.argsort(eMap.reshape(1,-1)[0]))
        qjRank = qjRank[::-1] if (eMap[posK, posI] < 0) else qjRank
        # print( iNegAll, kNegAll, iNegSV, kNegSV, iPosAll, kPosAll, iPosSV, kPosSV, posI, posK)
        return posI, posK, searchEnd, scanEntireSet, qjRank

    ## svm used for get classifier in single ada iteration
    def svm(x, y, kernel, C, d, eta, maxIter):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : ndarray splineNum * setNum
        # C : ndarray C[k] = C^{ori} * c[k]
        # d : ndarray splineNum
        # eMap: ndarray splineNum * setNum eMap[k,i] = wx[i] + b - y[i]d[k]
        ## init ----------------------------------------
        setNum = np.shape(x)[1]
        splineNum = np.size(d)
        b = 0
        a = np.zeros((splineNum, setNum))
        eMap = - np.matmul(d.reshape(splineNum,1), y.reshape(1,setNum))
        ## choose a_1, a_2 ------------------------------
        iter = 0
        changed = True
        while (iter < maxIter and changed):
            changed = False
            i, k, searchEnd, scanEntireSet, qjRank = smoSelectI(a, C, y, eMap, eta)
            if (searchEnd):
                break
            for qj in qjRank:
                q = int(np.floor(qj / setNum))
                j = qj % setNum
                if(j == i):
                    continue
                ## calc dAI dAJ dB -----------------------
                dAI, dAJ, dB = smoUpdateAB(y[i], y[j], a[k,i], a[q,j], C[k], C[q], kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[k,i], eMap[q,j])
                if (dAI == 0):
                    continue
                changed = True
                iter += 1
                ## update a b ----------------------------
                a[k,i] += dAI
                a[q,j] += dAJ
                b += dB
                ## update E ---------------------------
                dAYI = dAI * y[i]
                dAYJ = dAJ * y[j]
                for p in range(setNum):
                    eMap[:, p] += dB + dAYI * kernel(x[:,i], x[:,p]) + dAYJ * kernel(x[:,j], x[:,p])
                break
                            
            if (scanEntireSet and (not changed)):
                break
        
        print('iteration %d finish' % iter)
        
        svAIndex = list(np.nonzero(a)[0])
        aW = np.matmul(np.ones(a.shape[0]), a)
        aY = np.multiply(aW, y)
        plane = (lambda input: np.sum([aY[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        classifier = (lambda input: 1 if (plane(input) > 0) else -1)
        return classifier, plane, aW, b

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
        
        # spline = (lambda x : np.sum([c[k]*np.max([d[k] - x, 0]) for k in range(len(d))]))
        # x = np.arange(x_min, x_max, (x_max - x_min) / 100)
        # y = np.vectorize(lambda x: np.log(1+np.exp(-x)))(x)
        # z = np.vectorize(spline)(x)
        # plt.figure()
        # plt.plot(x, y, c="#000000", label='logistic')
        # plt.plot(x, z, c="#00ff00", label='spline')
        # plt.title('logistic regression carve')
        # plt.xlabel('y')
        # plt.ylabel('x')
        # plt.legend(loc='upper right')
        # plt.show()
        return c, d

    c, d = logisticSpline(plane_min, plane_max, splineNum)
    print(c,d)
    classifier, plane, aW, b = svm(x, y, kernel, C * c , d, eta, maxIter)    
    return classifier, plane
