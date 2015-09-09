from dana import *

# Default Time resolution
dt = 1.0*millisecond

Vmin       =   0.0
Vmax       =  20.0
Vh         =  16.0
Vc         =   3.0

popPerCueSTR = 1 
ctxTostr = 1
strToctx = 1.0/ctxTostr
popPerCueCTX = ctxTostr * popPerCueSTR 

tau = 0.01
CTX_tau, CTX_rest, CTX_noise = tau, -3.0 , 0.010
STR_tau, STR_rest, STR_noise = tau,  0.0 , 0.001
GPI_tau, GPI_rest, GPI_noise = tau,  10.0, 0.030 
THL_tau, THL_rest, THL_noise = tau, -40.0, 0.001
STN_tau, STN_rest, STN_noise = tau, -10.0, 0.001

def getCTXGain():
    if popPerCueCTX == 1:
        return 7
    elif popPerCueCTX == 4:
        return 4.45
    elif popPerCueCTX == 16:
        return 2.05
    else:
        return 1.775

ctxg = 7 
#getCTXGain()

def getGain():
   for i in 1+np.arange(10):
       j = 2 * i
       if 2**j == popPerCueCTX:
           return 0.5/(3.0**i)
   return 1

stcxg = 1 
#getGain()

def identity(x):
    if x < 0.0: return 0.0
    return x

def clamp(x, min=0, max=100) :
    return np.maximum(np.minimum(x,max),min)

def sigmoid(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

def unoise(Z, level):
    Z = (1+np.random.uniform(-level/2,level/2,Z.shape))*Z
    return np.maximum(Z,0.0)

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

class Structure:
    'Common class for all structures'

    def __init__(self, structure, pop=4):
        if structure == 'CTX':
            self._cog = Group(pop, 'dV/dt = (-V + Isyn + (ctxg * Iext) - CTX_rest)/CTX_tau ; U = clamp(V + CTX_noise); Isyn ; Iext ')
            self._mot = Group(pop, 'dV/dt = (-V + Isyn + (ctxg * Iext) - CTX_rest)/CTX_tau ; U = clamp(V + CTX_noise); Isyn ; Iext ')
        elif structure == 'STR':
            self._cog = Group(pop, 'dV/dt = (-V + Isyn + Iext - STR_rest)/STR_tau ; U = sigmoid(V + STR_noise); Isyn ; Iext ')
            self._mot = Group(pop, 'dV/dt = (-V + Isyn + Iext - STR_rest)/STR_tau ; U = sigmoid(V + STR_noise); Isyn ; Iext ')
        elif structure == 'GPI':
            self._cog = Group(pop, 'dV/dt = (-V + Isyn + Iext - GPI_rest)/GPI_tau ; U = clamp(V + GPI_noise); Isyn ; Iext ')
            self._mot = Group(pop, 'dV/dt = (-V + Isyn + Iext - GPI_rest)/GPI_tau ; U = clamp(V + GPI_noise); Isyn ; Iext ')
        elif structure == 'THL':
            self._cog = Group(pop, 'dV/dt = (-V + Isyn + Iext - THL_rest)/THL_tau ; U = clamp(V + THL_noise); Isyn ; Iext ')
            self._mot = Group(pop, 'dV/dt = (-V + Isyn + Iext - THL_rest)/THL_tau ; U = clamp(V + THL_noise); Isyn ; Iext ')
        elif structure == 'STN':
            self._cog = Group(pop, 'dV/dt = (-V + Isyn + Iext - STN_rest)/STN_tau ; U = clamp(V + STN_noise); Isyn ; Iext ')
            self._mot = Group(pop, 'dV/dt = (-V + Isyn + Iext - STN_rest)/STN_tau ; U = clamp(V + STN_noise); Isyn ; Iext ')


    @property
    def mot(self):
        """ The motor group """
        return self._mot

    @property
    def cog(self):
        """ The cognitive group """
        return self._cog

    def evaluate(self, dt):
        self._mot.evaluate(dt)
        self._cog.evaluate(dt)

    def reset(self):
        self._mot.reset()
        self._cog.reset()

class AssociativeStructure(Structure):
    'Class for associative structures (CTX, STR)'

    def __init__(self, structure, pop=4):
	Structure.__init__(self, structure, pop)
        if structure == 'CTX':
            self._ass = Group((pop,pop), 'dV/dt = (-V + Isyn + (ctxg * Iext) - CTX_rest)/CTX_tau ; U = clamp(V + CTX_noise); Isyn ; Iext ')
        elif structure == 'STR':
            self._ass = Group((pop,pop), 'dV/dt = (-V + Isyn + Iext - STR_rest)/STR_tau ; U = sigmoid(V + STR_noise); Isyn ; Iext ')

    @property
    def ass(self):
        """ The associative group """
        return self._ass

    def evaluate(self, dt):
        Structure.evaluate(self, dt)
        self._ass.evaluate(dt)

    def reset(self):
        Structure.reset(self)
        self._ass.reset()

def OneToOneWeights(sourcesize, targetsize):
    each = sourcesize / targetsize
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(targetsize):
        start = i * each
        kernel[i, start:start+each] = 1
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
    eachcog = m / targetsize
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(targetsize):
        start = i * eachcog * n 
        kernel[i, start:start+(eachcog*n)] = 1 
    return kernel

def AssToMotWeights(sourceshape, targetsize):
    # Mot structure is a 1xn array
    # source.shape = m x n
    (m, n) = sourceshape
    sourcesize = m * n
    eachmot = n / targetsize
    kernel = np.zeros((targetsize, sourcesize))
    for i in range(targetsize):
        ass = np.zeros((m, n))
        start = i * eachmot * m 
        ass[:,i] = 1 
        kernel[i, :] = np.reshape(ass,(1,ass.size))
    return kernel

def limitWeights(weights, Wmin = 0.25, Wmax = 0.75):
    N = np.random.normal(0.5, 0.005, weights.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return np.multiply((Wmin + (Wmax - Wmin)*N), weights)

def getConnection(source, target, kernel, clipWeights=False):
    if clipWeights:
        kernel = limitWeights(kernel)
    return DenseConnection(source, target, kernel)

def OneToOne(source, target, gain=1.0, clipWeights=False):
    if target.size > source.size:
        kernel = np.transpose(OneToOneWeights(target.size, source.size))
    else:
        kernel = OneToOneWeights(source.size, target.size)
    return getConnection(source, target, gain * kernel, clipWeights)

def OneToAll(source, target, gain=1.0, clipWeights=False):
    kernel = OneToAllWeights(source.size, target.size)
    return getConnection(source, target, gain * kernel, clipWeights)

def CogToAss(source, target, gain=1.0, clipWeights=False):
    kernel = CogToAssWeights(source.size, target.shape)
    return getConnection(source, target, gain * kernel, clipWeights)

def MotToAss(source, target, gain=1.0, clipWeights=False):
    kernel = MotToAssWeights(source.size, target.shape)
    return getConnection(source, target, gain * kernel, clipWeights)

def AssToCog(source, target, gain=1.0, clipWeights=False):
    kernel = AssToCogWeights(source.shape, target.size)
    return getConnection(source, target, gain * kernel, clipWeights)

def AssToMot(source, target, gain=1.0, clipWeights=False):
    kernel = AssToMotWeights(source.shape, target.size)
    return getConnection(source, target, gain * kernel, clipWeights)
