import numpy as np

def epsilonLinearSVR(x, y, kernel, C, eta, maxIter, epsilon):
    
    def smoUpdateAB(aI, aJ, aC, KII, KIJ, KJJ, EI, EJ, epsilon):
        ## get L H ---------------------------
        L = np.max((-aC, aJ + aI - aC))
        H = np.min(( aC, aJ + aI + aC))
        if (L >= H):
            # print("L >= H", L, H, yI * yJ)
            return 0, 0, 0
        ## get dAI dAJ ------------------------
        dK = (KII + KJJ - 2 * KIJ)
        if (dK == 0):
            # print("dK == 0")
            return 0, 0, 0
        if (aI > 0 and aJ < 0): 
            m = 2
        elif (aI < 0 and aJ > 0):
            m = -2
        else:
            m = 0
        aJNew = np.max((L, np.min((H, aJ + (EI - EJ + m * epsilon ) / dK))))
        dAJ = aJNew - aJ
        dAI = -dAJ
        aINew = aI + dAI
        ## get dB --------------------------
        if (aINew < 0):
            dBI = epsilon - EI - dAI * KII - dAJ * KIJ
        elif (aINew > 0):
            dBI = -epsilon - EI - dAI * KII - dAJ * KIJ
        else:
            dBI = -EI - dAI * KII - dAJ * KIJ
        if (aJNew < 0):
            dBJ = epsilon -EJ - dAI * KIJ - dAJ * KJJ
        elif (aJNew > 0):
            dBJ = -epsilon - EJ - dAI * KIJ - dAJ * KJJ
        else:
            dBJ = -EJ - dAI * KIJ - dAJ * KJJ
        if (aINew > -aC and aINew < aC and aINew != 0):
            dB = dBI
        elif (aJNew > -aC and aJNew < aC and aJNew != 0):
            dB = dBJ
        else:
            dB = (dBI + dBJ) / 2

        return dAI, dAJ, dB

    def smoSelectI(a, aC, eMap, epsilon, eta):
        '''
        target:
            tmpNeg = eMap[i] - epsilon
            tmpPos = eMap[i] + epsilon
            find ((a[i] < 0 and a[i] > -C) and (tmpNeg > eta or tmpNeg < -eta)):
            or   ((a[i] > 0 and a[i] < C) and (tmpPos > eta or tmpPos < -eta)):
            or   ((a[i] == 0) and ((tmpNeg > 0) or (tmpPos < 0))):
            or   ((a[i] == -C) and (tmpNeg < 0)) or ((a[i] == C) and (tmpPos > 0)):
        '''
        tmpNeg = eMap - epsilon
        tmpPos = eMap + epsilon
        setRank = list(np.argsort(eMap)) # sort from negative to positive
        iNegAllTmpNeg, iNegSVTmpNeg, iNegAllTmpPos, iNegSVTmpPos = -1, -1, -1, -1
        iPosAllTmpNeg, iPosSVTmpNeg, iPosAllTmpPos, iPosSVTmpPos = -1, -1, -1, -1
        # search tmpNeg \ tmpPos < -eta ------------------
        for i in setRank:            
            if (tmpNeg[i] > -eta):
                break
            # tmpNeg < 0 @ a[i] == -C, a[i] < 0 and a[i] > -C
            if (a[i] >= -aC and a[i] < 0):
                if (iNegAllTmpNeg == -1):
                    iNegAllTmpNeg = i
                if(a[i] != -aC):
                    iNegSVTmpNeg = i
                    break
        for i in setRank:
            if (tmpPos[i] > -eta):
                break
            # tmpPos < 0 @ a[i] == 0, a[i] > 0 and a[i] < C
            if (a[i] >= 0 and a[i] < aC):
                if (iNegAllTmpPos == -1):
                    iNegAllTmpPos = i
                if(a[i] != 0):
                    iNegSVTmpPos = i
                    break     

        # search tmpNeg \ tmpPos > eta ------------------------
        for i in setRank[::-1]:
            # tmpNeg > 0 @ a[i] == 0, a[i] < 0 and a[i] > -C
            if (tmpNeg[i] < eta):
                break
            if (a[i] <= 0 and a[i] > -aC):
                if (iPosAllTmpNeg == -1):
                    iPosAllTmpNeg = i
                if (a[i] != 0):
                    iPosSVTmpNeg = i
                    break   
        for i in setRank[::-1]:
            if (tmpPos[i] < eta):
                break
            # tmpPos > 0 @ a[i] == C, a[i] > 0 and a[i] < C
            if (a[i] > 0 and a[i] <= aC):
                if (iPosAllTmpPos == -1):
                    iPosAllTmpPos = i
                if (a[i] != aC):
                    iPosSVTmpPos = i
                    break   
        
        ## get posI ------------------------------
        searchEnd = False
        scanEntireSet = False
        # in SV group
        posI, maxTmp = -1, -1
        if (iPosSVTmpNeg != -1):
            posI = iPosSVTmpNeg
            maxTmp = tmpNeg[iPosSVTmpNeg]
        if (iPosSVTmpPos != -1):
            tmpMaxTmp = tmpPos[iPosSVTmpPos]
            if (maxTmp < tmpMaxTmp):
                posI = iPosSVTmpPos
                maxTmp = tmpMaxTmp
        if (iNegSVTmpNeg != -1):
            tmpMaxTmp = -tmpNeg[iNegSVTmpNeg]
            if (maxTmp < tmpMaxTmp):
                posI = iNegSVTmpNeg
                maxTmp = tmpMaxTmp
        if (iNegSVTmpPos != -1):
            tmpMaxTmp = -tmpPos[iNegSVTmpPos]
            if (maxTmp < tmpMaxTmp):
                posI = iNegSVTmpPos
                maxTmp = tmpMaxTmp
        
        # in All group
        if (posI == -1):
            scanEntireSet = True
            if (iPosAllTmpNeg != -1):
                posI = iPosAllTmpNeg
                maxTmp = tmpNeg[iPosAllTmpNeg]
            if (iPosAllTmpPos != -1):
                tmpMaxTmp = tmpPos[iPosAllTmpPos]
                if (maxTmp < tmpMaxTmp):
                    posI = iPosAllTmpPos
                    maxTmp = tmpMaxTmp
            if (iNegAllTmpNeg != -1):
                tmpMaxTmp = -tmpNeg[iNegAllTmpNeg]
                if (maxTmp < tmpMaxTmp):
                    posI = iNegAllTmpNeg
                    maxTmp = tmpMaxTmp
            if (iNegAllTmpPos != -1):
                tmpMaxTmp = -tmpPos[iNegAllTmpPos]
                if (maxTmp < tmpMaxTmp):
                    posI = iNegAllTmpPos
                    maxTmp = tmpMaxTmp

        if (posI == -1):
            searchEnd = True

        ## get posJ Rank -----------------------
        jRank = setRank # (EI - EJ + m * epsilon ) / dK
        jRank = jRank[::-1] if (eMap[posI] < 0) else jRank
        return posI, searchEnd, scanEntireSet, jRank

    def svm(x, y, kernel, C, eta, maxIter, epsilon):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : (alpha* - alpha) ndarray setNum 
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
            i, searchEnd, scanEntireSet, jRank = smoSelectI(a, C, eMap, epsilon, eta)
            if(searchEnd):
                break
            for j in jRank:
                if (j == i):
                    continue
                ## calc dAI dAJ dB ---------------------------
                dAI, dAJ, dB = smoUpdateAB(a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j], epsilon)
                if (dAI == 0):
                    continue
                changed = True
                iter += 1
                ## update a b ----------------------------
                a[i] += dAI
                a[j] += dAJ
                b += dB
                ## update E ---------------------------
                for k in range(setNum):
                    eMap[k] += dB + dAI * kernel(x[:,i], x[:,k]) + dAJ * kernel(x[:,j], x[:,k])
                break
            
            if (scanEntireSet and (not changed)):
                break
        
        print('iteration %d finish' % iter)
        svAIndex = list(np.nonzero(a)[0])
        f = (lambda input: np.sum([a[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        return f, a, b

    return svm(x, y, kernel, C, eta, maxIter, epsilon)

def epsilonQuadraticSVR(x, y, kernel, C, eta, maxIter, epsilon):

    def smoUpdateAB(aI, aJ, aC, KII, KIJ, KJJ, EI, EJ, epsilon):
        ## get L H ---------------------------
        ## get dAI dAJ ------------------------
        dK = (KII + KJJ - 2 * KIJ + 1 / aC)
        if (dK == 0):
            # print("dK == 0")
            return 0, 0, 0
        if (aI > 0 and aJ < 0): 
            m = 2
        elif (aI < 0 and aJ > 0):
            m = -2
        else:
            m = 0
        dAJ = (EI - EJ + (aI - aJ) / 2 / aC + m * epsilon ) / dK
        aJNew = aJ + dAJ
        dAI = -dAJ
        aINew = aI + dAI
        ## get dB --------------------------
        if (aINew < 0):
            dBI = epsilon - aINew / 2 / aC - EI - dAI * KII - dAJ * KIJ
        elif (aINew > 0):
            dBI = -epsilon - aINew / 2 / aC - EI - dAI * KII - dAJ * KIJ
        else:
            dBI = - EI - dAI * KII - dAJ * KIJ
        if (aJNew < 0):
            dBJ = epsilon - aJNew / 2 / aC - EJ - dAI * KIJ - dAJ * KJJ
        elif (aJNew > 0):
            dBJ = -epsilon - aJNew / 2 / aC - EJ - dAI * KIJ - dAJ * KJJ
        else:
            dBJ = - EJ - dAI * KIJ - dAJ * KJJ
        if (aINew != 0):
            dB = dBI
        elif (aJNew != 0):
            dB = dBJ
        else:    
            dB = (dBI + dBJ) / 2

        return dAI, dAJ, dB

    def smoSelectI(a, aC, eMap, epsilon, eta):
        '''
        target:
            tmpNeg = eMap[i] - epsilon
            tmpPos = eMap[i] + epsilon
            tmpNeg2 = tmpNeg + a[i] / 2 / C
            tmpPos2 = tmpPos + a[i] / 2 / C
            find ((a[i] < 0) and (tmpNeg2 > eta or tmpNeg2 < -eta)):
            or ((a[i] > 0) and (tmpPos2 > eta or tmpPos2 < -eta)):
            or ((a[i] == 0) and ((tmpNeg > 0) or (tmpPos < 0))):
        '''
        tmpNeg = eMap - epsilon
        tmpPos = eMap + epsilon
        tmpNeg2 = tmpNeg + a / 2 / C
        tmpPos2 = tmpPos + a / 2 / C
        setRank = list(np.argsort(eMap)) # sort from negative to positive
        setRank2 = list(np.argsort(eMap + a / 2 / C)) # sort from negative to positive
        iNegAllTmpNeg, iNegSVTmpNeg, iNegAllTmpPos, iNegSVTmpPos = -1, -1, -1, -1
        iPosAllTmpNeg, iPosSVTmpNeg, iPosAllTmpPos, iPosSVTmpPos = -1, -1, -1, -1
        # search tmpNeg2 \ tmpPos2 < -eta ------------------
        for i in setRank2:            
            if (tmpNeg2[i] > -eta):
                break
            # tmpNeg2 < 0 @ a[i] < 0
            if(a[i] < 0):
                iNegSVTmpNeg = i
                break
        for i in setRank2:
            if (tmpPos2[i] > -eta):
                break
            # tmpPos2 < 0 @ a[i] > 0
            if(a[i] > 0):
                iNegSVTmpPos = i
                break     

        # search tmpNeg2 \ tmpPos2 > eta -------------------
        for i in setRank2[::-1]:
            # tmpNeg2 > 0 @ a[i] < 0
            if (tmpNeg2[i] < eta):
                break
            if (a[i] < 0):
                iPosSVTmpNeg = i
                break   
        for i in setRank2[::-1]:
            if (tmpPos2[i] < eta):
                break
            # tmpPos2 > 0 @ a[i] > 0
            if (a[i] > 0):
                iPosSVTmpPos = i
                break   
        
        # search tmpPos < -eta ------------------
        for i in setRank:            
            if (tmpPos[i] > -eta):
                break
            # tmpPos < 0 @ a[i] == 0
            if (a[i] == 0):
                iNegAllTmpPos = i
                break
        # search tmpNeg > eta ------------------------
        for i in setRank[::-1]:
            # tmpNeg > 0 @ a[i] == 0
            if (tmpNeg[i] < eta):
                break
            if (a[i] == 0):
                iPosAllTmpNeg = i
                break 

        ## get posI ------------------------------
        searchEnd = False
        scanEntireSet = False
        # in SV group
        posI, maxTmp = -1, -1
        if (iPosSVTmpNeg != -1):
            posI = iPosSVTmpNeg
            maxTmp = tmpNeg[iPosSVTmpNeg]
        if (iPosSVTmpPos != -1):
            tmpMaxTmp = tmpPos[iPosSVTmpPos]
            if (maxTmp < tmpMaxTmp):
                posI = iPosSVTmpPos
                maxTmp = tmpMaxTmp
        if (iNegSVTmpNeg != -1):
            tmpMaxTmp = -tmpNeg[iNegSVTmpNeg]
            if (maxTmp < tmpMaxTmp):
                posI = iNegSVTmpNeg
                maxTmp = tmpMaxTmp
        if (iNegSVTmpPos != -1):
            tmpMaxTmp = -tmpPos[iNegSVTmpPos]
            if (maxTmp < tmpMaxTmp):
                posI = iNegSVTmpPos
                maxTmp = tmpMaxTmp
        
        # in All group
        if (posI == -1):
            scanEntireSet = True
            if (iPosAllTmpNeg != -1):
                posI = iPosAllTmpNeg
                maxTmp = tmpNeg[iPosAllTmpNeg]
            if (iPosAllTmpPos != -1):
                tmpMaxTmp = tmpPos[iPosAllTmpPos]
                if (maxTmp < tmpMaxTmp):
                    posI = iPosAllTmpPos
                    maxTmp = tmpMaxTmp
            if (iNegAllTmpNeg != -1):
                tmpMaxTmp = -tmpNeg[iNegAllTmpNeg]
                if (maxTmp < tmpMaxTmp):
                    posI = iNegAllTmpNeg
                    maxTmp = tmpMaxTmp
            if (iNegAllTmpPos != -1):
                tmpMaxTmp = -tmpPos[iNegAllTmpPos]
                if (maxTmp < tmpMaxTmp):
                    posI = iNegAllTmpPos
                    maxTmp = tmpMaxTmp

        if (posI == -1):
            searchEnd = True

        ## get posJ Rank -----------------------
        jRank = setRank2 # (EI - EJ + (aI - aJ) / 2 / aC + m * epsilon ) / dK
        jRank = jRank[::-1] if (eMap[posI] + a[posI] / 2 / C < 0) else jRank
        return posI, searchEnd, scanEntireSet, jRank

    def svm(x, y, kernel, C, eta, maxIter, epsilon):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : (alpha* - alpha) ndarray setNum 
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
            i, searchEnd, scanEntireSet, jRank = smoSelectI(a, C, eMap, epsilon, eta)
            if (searchEnd):
                break
            for j in jRank:
                if (j == i):
                    continue
                ## calc dAI dAJ dB ---------------------------
                dAI, dAJ, dB = smoUpdateAB(a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j], epsilon)
                if (dAI == 0):
                    continue
                changed = True
                iter += 1
                ## update a b ----------------------------
                a[i] += dAI
                a[j] += dAJ
                b += dB
                ## update E ---------------------------
                for k in range(setNum):
                    eMap[k] += dB + dAI * kernel(x[:,i], x[:,k]) + dAJ * kernel(x[:,j], x[:,k])
                break
            
            if (scanEntireSet and (not changed)):
                break
        
        print('iteration %d finish' % iter)
        svAIndex = list(np.nonzero(a)[0])
        f = (lambda input: np.sum([a[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        return f, a, b

    return svm(x, y, kernel, C, eta, maxIter, epsilon)

def huberSVR(x, y, kernel, C, eta, maxIter, uc):

    def smoUpdateAB(aI, aJ, aC, KII, KIJ, KJJ, EI, EJ, uc):
        ## get L H ---------------------------
        L = np.max((-aC, aJ + aI - aC))
        H = np.min(( aC, aJ + aI + aC))
        if (L >= H):
            # print("L >= H", L, H, yI * yJ)
            return 0, 0, 0
        ## get dAI dAJ ------------------------
        dK = (KII + KJJ - 2 * KIJ + 2*uc / aC)
        if (dK == 0):
            # print("dK == 0")
            return 0, 0, 0
        aJNew = np.max((L, np.min((H, aJ + (EI - EJ + uc * (aI - aJ) / aC ) / dK))))
        dAJ = aJNew - aJ
        dAI = -dAJ
        aINew = aI + dAI
        ## get dB --------------------------
        if (aINew != 0):
            dBI = -uc * aINew / aC - EI - dAI * KII - dAJ * KIJ
        else:
            dBI = -EI - dAI * KII - dAJ * KIJ
        if (aJNew != 0):
            dBJ = -uc * aJNew / aC - EJ - dAI * KIJ - dAJ * KJJ
        else:
            dBJ = -EJ - dAI * KIJ - dAJ * KJJ
        if (aINew > -aC and aINew < aC ):
            dB = dBI
        elif (aJNew > -aC and aJNew < aC ):
            dB = dBJ
        else:
            dB = (dBI + dBJ) / 2

        return dAI, dAJ, dB

    def smoSelectI(a, aC, eMap, uc, eta):
        '''
        target:
            tmp = eMap[i] + uc * a[i] / C
            find ((a[i] < 0 and a[i] > -C) and (tmp > eta or tmp < -eta)):
            or ((a[i] > 0 and a[i] < C) and (tmp > eta or tmp < -eta)):
            or ((a[i] == 0) and (eMap[i] > eta or eMap[i] < -eta)):
            or ((a[i] == -C) and (eMap[i] < uc)) or ((a[i] == C) and (eMap[i] > -uc)):
        '''
        tmp = eMap + uc * a / aC
        setRank = list(np.argsort(eMap)) # sort from negative to positive
        setRank2 = list(np.argsort(tmp)) # sort from negative to positive
        iNegAll, iNegSV, iNegAllUc = -1, -1, -1
        iPosAll, iPosSV, iPosAllUc = -1, -1, -1
        # search tmp < -eta ------------------
        for i in setRank2:            
            if (tmp[i] > -eta):
                break
            # tmp < 0 @ a[i] != -C, C, 0
            if (a[i] != -aC and a[i] != 0 and a[i] != aC):
                iNegSV = i
                break
        # search tmp > eta ------------------------
        for i in setRank2[::-1]:
            if (tmp[i] < eta):
                break
            # tmp > 0 @ a[i] != -C,0,C
            if (a[i] != -aC and a[i] != 0 and a[i] != aC):
                iPosSV = i
                break   
        
        # search eMap < -eta ------------------
        for i in setRank:            
            if (eMap[i] > -eta):
                break
            # eMap < 0 @ a[i] == 0
            if (a[i] == 0):
                iNegAll = i
                break
        # search eMap < -eta ------------------
        for i in setRank[::-1]:            
            if (eMap[i] < eta):
                break
            # eMap > 0 @ a[i] == 0
            if (a[i] == 0):
                iPosAll = i
                break
        # search eMap - uc < -eta ------------------
        for i in setRank:            
            if (eMap[i] -uc > -eta):
                break
            # eMap - uc < 0 @ a[i] == -C
            if (a[i] == -aC):
                iNegAllUc = i
                break
        # search eMap + uc > eta ------------------
        for i in setRank[::-1]:            
            if (eMap[i] + uc < eta):
                break
            # eMap + uc > 0 @ a[i] == C
            if (a[i] == aC):
                iPosAllUc = i
                break

        ## get posI ------------------------------
        searchEnd = False
        scanEntireSet = False
        # in SV group
        posI, maxTmp = -1, -1
        if (iPosSV != -1):
            posI = iPosSV
            maxTmp = tmp[iPosSV]
        if (iNegSV != -1):
            tmpMaxTmp = -tmp[iNegSV]
            if (maxTmp < tmpMaxTmp):
                posI = iNegSV
                maxTmp = tmpMaxTmp
        
        # in All group
        if (posI == -1):
            scanEntireSet = True
            if (iPosAll != -1):
                posI = iPosAll
                maxTmp = eMap[iPosAll]
            if (iPosAllUc != -1):
                tmpMaxTmp = eMap[iPosAllUc] + uc
                if (maxTmp < tmpMaxTmp):
                    posI = iPosAllUc
                    maxTmp = tmpMaxTmp
            if (iNegAll != -1):
                tmpMaxTmp = -eMap[iNegAll]
                if (maxTmp < tmpMaxTmp):
                    posI = iNegAll
                    maxTmp = tmpMaxTmp
            if (iNegAllUc != -1):
                tmpMaxTmp = -eMap[iNegAllUc] + uc
                if (maxTmp < tmpMaxTmp):
                    posI = iNegAllUc
                    maxTmp = tmpMaxTmp

        if (posI == -1):
            searchEnd = True

        ## get posJ Rank -----------------------
        
        jRank = setRank2 # (EI - EJ + uc * (aI - aJ) / aC ) / dK
        jRank = jRank[::-1] if (tmp[posI] < 0) else jRank
        return posI, searchEnd, scanEntireSet, jRank

    def svm(x, y, kernel, C, eta, maxIter, uc):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : (alpha* - alpha) ndarray setNum 
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
            i, searchEnd, scanEntireSet, jRank = smoSelectI(a, C, eMap, uc, eta)
            if(searchEnd):
                break
            for j in jRank:
                if (j == i):
                    continue
                ## calc dAI dAJ dB ---------------------------
                dAI, dAJ, dB = smoUpdateAB(a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j], uc)
                if (dAI == 0):
                    continue
                changed = True
                iter += 1
                ## update a b ----------------------------
                a[i] += dAI
                a[j] += dAJ
                b += dB
                ## update E ---------------------------
                for k in range(setNum):
                    eMap[k] += dB + dAI * kernel(x[:,i], x[:,k]) + dAJ * kernel(x[:,j], x[:,k])
                break
            
            if (scanEntireSet and (not changed)):
                break
        
        print('iteration %d finish' % iter)
        svAIndex = list(np.nonzero(a)[0])
        f = (lambda input: np.sum([a[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        return f, a, b

    return svm(x, y, kernel, C, eta, maxIter, uc)

'''
def waste():
    def svm_epLi(x, y, kernel, C, eta, maxIter, epsilon):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : (alpha* - alpha) ndarray setNum 
        # eMap[i] = wx[i] + b - y[i]
        ## init ----------------------------------------
        setNum = np.shape(x)[1]
        b = 0
        a = np.zeros(setNum)
        eMap = np.zeros(setNum) - y # cuz y is int, eMap is float
        ## choose a_1, a_2 ------------------------------
        iter = 0
        scanEntireSet = True
        while (iter < maxIter):
            iter += 1
            changed = False
            for i in range(setNum):
                ## choose a_1 ----------------
                tmpNeg = eMap[i] - epsilon
                tmpPos = eMap[i] + epsilon
                if ((a[i] < 0 and a[i] > -C) and (tmpNeg > eta or tmpNeg < -eta)):
                    pass
                elif ((a[i] > 0 and a[i] < C) and (tmpPos > eta or tmpPos < -eta)):
                    pass
                elif (scanEntireSet):
                    if ((a[i] == 0) and ((tmpNeg > 0) or (tmpPos < 0))):
                        pass
                    elif ((a[i] == -C) and (tmpNeg < 0)) or ((a[i] == C) and (tmpPos > 0)):
                        pass
                    else: 
                        continue
                else:
                    continue
                ## choose a_2 ----------------- 
                # sort from big to small
                jRank = list(np.argsort(-np.abs(eMap - eMap[i])))
                for j in jRank:
                    if (j == i):
                        continue
                    ## calc dAI dAJ dB ---------------------------
                    dAI, dAJ, dB = smoUpdateAB(a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j], epsilon)
                    if (dAI == 0):
                        continue
                    changed = True
                    ## update a b ----------------------------
                    a[i] += dAI
                    a[j] += dAJ
                    b += dB
                    ## update E ---------------------------
                    for k in range(setNum):
                        eMap[k] += dB + dAI * kernel(x[:,i], x[:,k]) + dAJ * kernel(x[:,j], x[:,k])
                    break
            
            if ((not scanEntireSet) and (not changed)):
                scanEntireSet = True
            elif (scanEntireSet and changed):
                scanEntireSet = False
            elif (scanEntireSet and (not changed)):
                break
        
        print('iteration %d finish' % iter)
        svAIndex = list(np.nonzero(a)[0])
        f = (lambda input: np.sum([a[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        return f, a, b

    def svm_epQu(x, y, kernel, C, eta, maxIter, epsilon):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : (alpha* - alpha) ndarray setNum 
        # eMap[i] = wx[i] + b - y[i]
        ## init ----------------------------------------
        setNum = np.shape(x)[1]
        b = 0
        a = np.zeros(setNum)
        eMap = np.zeros(setNum) - y # cuz y is int, eMap is float
        ## choose a_1, a_2 ------------------------------
        iter = 0
        scanEntireSet = True
        while (iter < maxIter):
            iter += 1
            changed = False
            for i in range(setNum):
                ## choose a_1 ----------------
                tmpNeg = eMap[i] - epsilon
                tmpPos = eMap[i] + epsilon
                tmpNeg2 = tmpNeg + a[i] / 2 / C
                tmpPos2 = tmpPos + a[i] / 2 / C
                if ((a[i] < 0) and (tmpNeg2 > eta or tmpNeg2 < -eta)):
                    pass
                elif ((a[i] > 0) and (tmpPos2 > eta or tmpPos2 < -eta)):
                    pass
                elif (scanEntireSet):
                    if ((a[i] == 0) and ((tmpNeg > 0) or (tmpPos < 0))):
                        pass
                    else: 
                        continue
                else:
                    continue
                ## choose a_2 ----------------- 
                # sort from big to small
                jRank = list(np.argsort(-np.abs(eMap - eMap[i])))
                for j in jRank:
                    if (j == i):
                        continue
                    ## calc dAI dAJ dB ---------------------------
                    dAI, dAJ, dB = smoUpdateAB(a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j], epsilon)
                    if (dAI == 0):
                        continue
                    changed = True
                    ## update a b ----------------------------
                    a[i] += dAI
                    a[j] += dAJ
                    b += dB
                    ## update E ---------------------------
                    for k in range(setNum):
                        eMap[k] += dB + dAI * kernel(x[:,i], x[:,k]) + dAJ * kernel(x[:,j], x[:,k])
                    break
            
            if ((not scanEntireSet) and (not changed)):
                scanEntireSet = True
            elif (scanEntireSet and changed):
                scanEntireSet = False
            elif (scanEntireSet and (not changed)):
                break
        
        print('iteration %d finish' % iter)
        svAIndex = list(np.nonzero(a)[0])
        f = (lambda input: np.sum([a[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        return f, a, b

    def svm_huber(x, y, kernel, C, eta, maxIter, uc):
        # x : ndarray inputDim * setNum
        # y : ndarray setNum
        # a : (alpha* - alpha) ndarray setNum 
        # eMap[i] = wx[i] + b - y[i]
        ## init ----------------------------------------
        setNum = np.shape(x)[1]
        b = 0
        a = np.zeros(setNum)
        eMap = np.zeros(setNum) - y # cuz y is int, eMap is float
        ## choose a_1, a_2 ------------------------------
        iter = 0
        scanEntireSet = True
        while (iter < maxIter):
            iter += 1
            changed = False
            for i in range(setNum):
                ## choose a_1 ----------------
                tmp = eMap[i] + uc * a[i] / C
                if ((a[i] < 0 and a[i] > -C) and (tmp > eta or tmp < -eta)):
                    pass
                elif ((a[i] > 0 and a[i] < C) and (tmp > eta or tmp < -eta)):
                    pass
                elif (scanEntireSet):
                    if ((a[i] == 0) and (eMap[i] > eta or eMap[i] < -eta)):
                        pass
                    elif ((a[i] == -C) and (eMap[i] < uc)) or ((a[i] == C) and (eMap[i] > -uc)):
                        pass
                    else: 
                        continue
                else:
                    continue
                ## choose a_2 ----------------- 
                # sort from big to small
                jRank = list(np.argsort(-np.abs(eMap - eMap[i])))
                for j in jRank:
                    if (j == i):
                        continue
                    ## calc dAI dAJ dB ---------------------------
                    dAI, dAJ, dB = smoUpdateAB(a[i], a[j], C, kernel(x[:,i], x[:,i]), kernel(x[:,i], x[:,j]), kernel(x[:,j], x[:,j]), eMap[i], eMap[j], uc)
                    if (dAI == 0):
                        continue
                    changed = True
                    ## update a b ----------------------------
                    a[i] += dAI
                    a[j] += dAJ
                    b += dB
                    ## update E ---------------------------
                    for k in range(setNum):
                        eMap[k] += dB + dAI * kernel(x[:,i], x[:,k]) + dAJ * kernel(x[:,j], x[:,k])
                    break
            
            if ((not scanEntireSet) and (not changed)):
                scanEntireSet = True
            elif (scanEntireSet and changed):
                scanEntireSet = False
            elif (scanEntireSet and (not changed)):
                break
        
        print('iteration %d finish' % iter)
        svAIndex = list(np.nonzero(a)[0])
        f = (lambda input: np.sum([a[i] * kernel(x[:, i], input) for i in svAIndex]) + b)
        return f, a, b
'''