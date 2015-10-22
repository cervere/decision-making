import numpy as np
from exmodel import *

Vmin       =   0.0
Vmax       =  20.0
Vh         =  16.0
Vc         =   3.0

numOfCues = 4
base = 1
popPerCueCTX = base * 16
popPerCueSTR = base * 4
popPerCueGPI = base * 4
popPerCueSTN = base * 4
popPerCueTHL = base * 4

strToCtx = (1.0*popPerCueSTR)/popPerCueCTX
gpiToStr = (1.0)/popPerCueSTR
thlToGpi = (1.0)/popPerCueGPI
ctxToThl = (1.0*popPerCueCTX)/popPerCueTHL
thlToCtx = (1.0*popPerCueTHL)/popPerCueCTX
stnToCtx = (1.0*popPerCueSTN)/popPerCueCTX

tau = 0.01
CTX_tau, CTX_rest, CTX_noise = tau, -3.0 , 0.010
STR_tau, STR_rest, STR_noise = tau,  0.0 , 0.001
GPI_tau, GPI_rest, GPI_noise = tau,  10.0, 0.030 
THL_tau, THL_rest, THL_noise = tau, -40.0, 0.001
STN_tau, STN_rest, STN_noise = tau, -10.0, 0.001

def getNormal(size):    
    mu = size/2.0
    sig = mu/3.0
    data = 0.5 + np.arange(size)
    return np.exp(-(data - mu)**2 / (2 * sig**2))

def get2DNormal(x, y):
    Xm, Ym = (x-1)/2.0, (y-1)/2.0
    diag = np.sqrt(x**2 + y**2) 
    sig = diag/2.0/3.0 
    data = np.zeros((x,y))
    for i in range(x):
        for j in range(y):
            data[i,j] = np.exp(-np.abs(i - Xm) * np.abs(j - Ym) / (2 * sig**2))
    return data

def OneToOneWeights(sourcesize, targetsize):
    if targetsize > sourcesize:
        return np.transpose(OneToOneWeights(targetsize, sourcesize))
    each = sourcesize / targetsize
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(targetsize):
        start = i * each
        kernel[i, start:start+each] = 1
    return kernel

def AscToAscWeights(sourceshape, targetshape):
    (p, q) = sourceshape
    sourcesize = p * q 
    (m, n) = targetshape
    targetsize = m * n
    r = p/m
    c = q/n
    each = sourcesize / targetsize
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(m):
        for j in range(n):
            cell = np.zeros((p, q))
            cell[r*i:r*i+r,c*j:c*j+c] = 1
            kernel[i*n+j] = np.reshape(cell, (1, cell.size)) 
    return kernel

def OneToAllWeights(sourcesize, targetsize):
    kernel = np.ones((targetsize, sourcesize))
    return kernel

def CogToAssWeights(sourcesize, targetshape):
    # COG structure is an nx1 array
    # target.shape = m x n
    (m, n) = targetshape
    targetsize = m * n
    each = sourcesize / n 
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(m):
        start = i * each
        kernel[i*n:i*n+n, start:start+each] = 1
    return kernel

def MotToAssWeights(sourcesize, targetshape):
    # MOT structure is a 1xn array
    # target.shape = m x n
    (m, n) = targetshape
    targetsize = m * n
    each = sourcesize / m 
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(m):
        for j in range(n):
            start = j * each
            kernel[i*m+j , start:start+each] = 1 
    return kernel

def AssToCogWeights(sourceshape, targetsize):
    # COG structure is a nx1 array
    # source.shape = m x n
    (m, n) = sourceshape
    sourcesize = m * n
    eachsrc = m / numOfCues
    eachtar = targetsize / numOfCues
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(numOfCues):
        ass = np.zeros((m, n))
        ass[i*eachsrc:i*eachsrc+eachsrc, :] = 1 
        kernel[i*eachtar:i*eachtar+eachtar, :] = np.reshape(ass,(1,ass.size))
    return kernel

def AssToMotWeights(sourceshape, targetsize):
    # Mot structure is a 1xn array
    # source.shape = m x n
    (m, n) = sourceshape
    sourcesize = m * n
    eachsrc = n / numOfCues
    eachtar = targetsize / numOfCues
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(targetsize):
        ass = np.zeros((m, n))
        ass[:,i*eachsrc:i*eachsrc+eachsrc] = 1 
        kernel[i*eachtar:i*eachtar+eachtar, :] = np.reshape(ass,(1,ass.size))
    return kernel

def limitWeights(weights, Wmin = 0.25, Wmax = 0.75):
    N = np.random.normal(0.5, 0.005, weights.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return np.multiply((Wmin + (Wmax - Wmin)*N), weights)

def getConnection(source, target, kernel, gain, clipWeights):
    if clipWeights:
        kernel = limitWeights(kernel)
    return Connection(source, target, kernel, gain)

def OneToOne(source, target, gain=1.0, clipWeights=False):
    kernel = OneToOneWeights(source.size, target.size)
    return getConnection(source, target, kernel, gain, clipWeights)

def AscToAsc(source, target, gain=1.0, clipWeights=False):
    spop = int(np.sqrt(source.size))
    tpop = int(np.sqrt(target.size))
    kernel = AscToAscWeights((spop,spop), (tpop,tpop))
    return getConnection(source, target, kernel, gain, clipWeights)

def OneToAll(source, target, gain=1.0, clipWeights=False):
    kernel = OneToAllWeights(source.size, target.size)
    return getConnection(source, target, kernel, gain, clipWeights)

def CogToAss(source, target, gain=1.0, clipWeights=False):
    tpop = int(np.sqrt(target.size))
    kernel = CogToAssWeights(source.size, (tpop,tpop))
    return getConnection(source, target, kernel, gain, clipWeights)

def MotToAss(source, target, gain=1.0, clipWeights=False):
    tpop = int(np.sqrt(target.size))
    kernel = MotToAssWeights(source.size, (tpop,tpop))
    return getConnection(source, target, kernel, gain, clipWeights)

def AssToCog(source, target, gain=1.0, clipWeights=False):
    spop = int(np.sqrt(source.size))
    kernel = AssToCogWeights((spop,spop), target.size)
    return getConnection(source, target, kernel, gain, clipWeights)

def AssToMot(source, target, gain=1.0, clipWeights=False):
    spop = int(np.sqrt(source.size))
    kernel = AssToMotWeights((spop,spop), target.size)
    return getConnection(source, target, kernel, gain, clipWeights)

