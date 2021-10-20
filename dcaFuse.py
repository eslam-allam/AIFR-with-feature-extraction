import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power, svd
from sklearn import preprocessing




def Diag_Bx(PhibX):
    """
    Diagolalize the between-class scatter matrix (Sb) for Y
    """

    artSbX = np.dot(PhibX.T, PhibX)
    eigVals, eigVecs = LA.eig(artSbX)

    # Ignore zero eigenvalues
    maxEigVal = max(eigVals)
    zeroEigIndx = np.squeeze(np.argwhere(eigVals / maxEigVal < 1e-6), axis=1)
    eigVals = np.delete(eigVals, zeroEigIndx)
    eigVecs = np.delete(eigVecs, zeroEigIndx, axis=1)

    # Sort in descending order
    idx = np.flip(np.argsort(eigVals))
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:, idx]

    # Calculate the actual eigenvectors for the between-class scatter matrix (Sbx)
    SbxEigVecs = np.dot(PhibX, eigVecs)

    # Normalize to unit length to create orthonormal eigenvectors for Sbx:
    cx = len(eigVals)  # Rank of SbX
    for i in range(cx):
        SbxEigVecs[:, i] = SbxEigVecs[:, i] / LA.norm(SbxEigVecs[:, i])

    # Unitize the between-class scatter matrix (Sbx) for X
    SbxEigVals = np.diag(eigVals)  # SbxEigVals is a (cx x cx) diagonal matrix
    Wbx = np.dot(SbxEigVecs, fractional_matrix_power(SbxEigVals, -0.5))  # Wbx is a (p x cx) matrix which unitizes Sbx

    return cx, Wbx


def dcaFuse(X, Y, L):

    """
    X (p*n)
    Y (q*n)
    L (n)
    """

    # X = np.array([[1,2,3,4],
    #               [4,5,6,7],
    #               ])
    # Y = np.array([[4,4,4,4],
    #               [1,1,1,1]]
    #              )
    # L = np.array([0,1,2,2])


    p, n = X.shape
    q = Y.shape[0]

    # Normalize
    X = preprocessing.normalize(X)
    Y = preprocessing.normalize(Y)


    classes = np.unique(L)
    c = len(classes)
    nSample = np.zeros([c])

    cellX = []
    cellY = []

    for i in range(c):
        idx = np.squeeze(np.argwhere(L == classes[i]), axis=1)
        cellX.append(X[:,idx])
        cellY.append(Y[:,idx])
        nSample[i] = len(idx)

    meanX = np.mean(X, axis=1)
    meanY = np.mean(Y, axis=1)

    classMeanX = np.zeros([p,c])
    classMeanY = np.zeros([q,c])

    for i in range(c):
        classMeanX[:,i] = np.mean(cellX[i], axis=1)
        classMeanY[:,i] = np.mean(cellY[i], axis=1)


    PhibX = np.zeros([p,c])
    PhibY = np.zeros([q,c])

    for i in range(c):
        PhibX[:,i] = np.sqrt(nSample[i]) * (classMeanX[:,i] - meanX)
        PhibY[:,i] = np.sqrt(nSample[i]) * (classMeanY[:,i] - meanY)

    del L, idx, cellX, cellY, meanX, meanY, classMeanX, classMeanY

    """
        Diagolalize the between-class scatter matrix (Sb) for X and Y
    """

    cx, Wbx = Diag_Bx(PhibX)
    cy, Wby = Diag_Bx(PhibY)

    """
        Project data in a space, where the between-class scatter matrices are 
        identity and the classes are separated
    """

    r = min(cx, cy)
    Wbx = Wbx[:, :r]
    Wby = Wby[:, :r]

    Xp = np.dot(Wbx.T, X)
    Yp = np.dot(Wby.T, Y)

    """
        Unitize the between-set covariance matrix (Sxy)
        Note that Syx == Sxy'
    """

    Sxy = np.dot(Xp, Yp.T)      # Between-set covariance matrix
    Wcx, S_val, Wcy = svd(Sxy)
    S = np.diag(S_val)


    Wcx = np.dot(Wcx, fractional_matrix_power(S, -0.5))
    Wcy = np.dot(Wcy, fractional_matrix_power(S, -0.5))

    Xs = np.dot(Wcx.T, Xp)
    Ys = np.dot(Wcy.T, Yp)

    Ax = np.dot(Wcx.T, Wbx.T)
    Ay = np.dot(Wcy.T, Wby.T)

    return Xs, Ys, Ax, Ay