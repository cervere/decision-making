# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
# -----------------------------------------------------------------------------
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp
from libc.stdlib cimport rand, srand, RAND_MAX


# ---------------------------------------------------------------- Function ---
cdef class Function:
    cdef double call(self, double x) except *:
        return x


# --- Identity ---
cdef class Identity(Function):

    cdef double call(self, double x) except *:
        if x < 0.0: return 0.0
        return x


# --- Clamp ---
cdef class Clamp(Function):
    cdef public double min, max

    def __init__(self, double min=0, double max=1e9):
        self.min = min
        self.max = max

    cdef double call(self, double x) except *:
        if x < self.min: return self.min
        if x > self.max: return self.max
        return x


# --- Noise ---
cdef class UniformNoise(Function):
    cdef public double amount

    def __init__(self, double amount):
        self.amount = amount

    cdef double call(self, double x) except *:
        return x * (1 + self.amount*(rand()/float(RAND_MAX) - 0.5))


# --- Sigmoid ---
cdef class Sigmoid(Function):
    cdef public double Vmin, Vmax, Vh, Vc

    def __init__(self, Vmin=0.0, Vmax=20.0, Vh=16., Vc=3.0):
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Vh = Vh
        self.Vc = Vc

    cdef double call(self, double V) except *:
        return self.Vmin + (self.Vmax-self.Vmin)/(1.0+exp((self.Vh-V)/self.Vc))



# ------------------------------------------------------------------- Group ---
# Python group type (dtype)
dtype = [("V",  float),
         ("U",  float),
         ("Isyn",  float),
         ("Iext", float)]

# C group type (ctype)
cdef packed struct ctype:
    np.float64_t V
    np.float64_t U
    np.float64_t Isyn
    np.float64_t Iext


cdef class Group:
    """  """

    cdef double      _tau
    cdef double      _rest
    cdef double      _noise
    cdef double      _delta
    cdef ctype[:]    _units
    cdef Function    _activation
    cdef Function    _unoise
    cdef int         _history_index
    cdef double[:,:] _history
    cdef int         _numOfCues

    def __init__(self, units, tau=0.01, rest=0.0, noise=0.0, activation = Identity()):
        self._tau = tau
        self._rest = rest
        self._unoise = UniformNoise(noise)
        self._units = units
        self._delta = 0
        self._activation = activation
        self._history_index = 0
        self._numOfCues = 4
        self._history = np.zeros((10000, self._numOfCues))#num of cues is 4

    property history:
        """ Activity history (firing rate) """
        def __get__(self):
            return np.asarray(self._history)

    property delta:
        """ Difference of activity between the first two maximum activites """
        def __get__(self):
            return self._delta

    property tau:
        """ Membrane time constant """
        def __get__(self):
            return self._tau
        def __set__(self, value):
            self._tau = value

    property rest:
        """ Membrane resting potential """
        def __get__(self):
            return self._rest
        def __set__(self, value):
            self._rest = value

    property V:
        """ Firing rate """
        def __get__(self):
            return np.asarray(self._units)["V"]

    property U:
        """ Membrane potential """
        def __get__(self):
            return np.asarray(self._units)["U"]

    property Isyn:
        """ Input current from external synapses """
        def __get__(self):
            return np.asarray(self._units)["Isyn"]

    property Iext:
        """ Input current from external sources """
        def __get__(self):
            return np.asarray(self._units)["Iext"]
        def __set__(self, value):
            np.asarray(self._units)["Iext"] = value

    @cython.boundscheck(False)
    def evaluate(self, double dt):
        """ Compute activities (Forward Euler method) """

        cdef int i
#        cdef int m,n
        cdef noise
        cdef ctype * unit
        cdef double max1=0, max2=0
        cdef double[:] meanAct
#        m = self._units.shape[0]
#        n = self._units.shape[1]
        #for i in range(m):
            #for j in range(n):
        for i in range(len(self._units)):
            if 1:
                unit = & self._units[i]
                # Compute white noise
                #noise = self._noise*(rand()/float(RAND_MAX)) - self._noise/2.0
                # Update membrane potential
                unit.V += dt/self._tau*(-unit.V + unit.Isyn + unit.Iext - self._rest )
                # Update firing rate
                unit.U = self._unoise.call(self._activation.call(unit.V))
                #unit.U = self._unoise(self._activation.call(unit.V), self._noise)
                # Store firing rate activity
                #self._history[self._history_index,i] = unit.U

                # Here we record the max activities to store their difference
                # This is used later to decide if a motor decision has been made
                #if unit.U > max1:   max1 = unit.U
                #elif unit.U > max2: max2 = unit.U
        # Store firing rate activity
        #meanActivity(np.asarray(self._units)["U"], 4, self._history[self._history_index,:])
        population = np.asarray(self._units)["U"]
        meanAct = np.reshape(population, (self._numOfCues, population.size/self._numOfCues)).mean(axis=1)
        for i in range(meanAct.shape[0]):
            #print "%d, %d" % (i,meanAct[i])
            self._history[self._history_index,i] = meanAct[i]
            if meanAct[i] > max1:   max1 = meanAct[i]
            elif meanAct[i] > max2: max2 = meanAct[i]

        self._delta = max1 - max2
        self._history_index +=1


    def reset(self):
        """ Reset all activities and history index """

        cdef int i
        #cdef int m,n
        self._history_index = 0
        #m = self._units.shape[0]
        #n = self._units.shape[1]
        #for i in range(m):
        #    for j in range(n):
        for i in range(len(self._units)):
            if 1:
                self._units[i].V = 0
                self._units[i].U = 0
                self._units[i].Isyn = 0
                self._units[i].Iext = 0



    def __getitem__(self, key):
        return np.asarray(self._units)[key]


    def __setitem__(self, key, value):
        np.asarray(self._units)[key] = value


def meanActivity(np.ndarray[double, ndim=2] population, int numOfCues, double[:] out):
    me = np.reshape(population, (numOfCues, population.size/numOfCues)).mean(axis=1)
    for i in range(len(out)):
        out[i] = me[i]

# --------------------------------------------------------------- Structure ---
cdef class Structure:
    cdef Group _mot
    cdef Group _cog

    def __init__(self, pop=16, tau=0.01, rest=0, noise=0, activation=Identity()):
        self._mot = Group(np.zeros(pop,dtype=dtype), tau=tau, rest=rest,
                           noise=noise, activation=activation)
        self._cog = Group(np.zeros(pop,dtype=dtype), tau=tau, rest=rest,
                         noise=noise, activation=activation)

    property mot:
        """ The motor group """
        def __get__(self):
            return self._mot

    property cog:
        """ The cognitive group """
        def __get__(self):
            return self._cog

    def evaluate(self, double dt):
        self._mot.evaluate(dt)
        self._cog.evaluate(dt)

    def reset(self):
        self._mot.reset()
        self._cog.reset()



# ---------------------------------------------------- AssociativeStructure ---
cdef class AssociativeStructure(Structure):
    cdef public Group _ass

    def __init__(self, pop=16, tau=0.01, rest=0, noise=0, activation=Identity()):
        Structure.__init__(self, pop, tau, rest, noise, activation)
        self._ass = Group(np.zeros(pop*pop,dtype=dtype), tau=tau, rest=rest,
                          noise=noise, activation=activation)

    def evaluate(self, double dt):
        Structure.evaluate(self, dt)
        self._ass.evaluate(dt)

    def reset(self):
        Structure.reset(self)
        self._ass.reset()

    property ass:
        """ The associative group """
        def __get__(self):
            return self._ass


# ------------------------------------------------------------- Connections ---
cdef class Connection:
    cdef double[:] _source
    cdef double[:] _target
    cdef double[:,:] _weights
    cdef long[:,:] _effweights
    cdef double    _gain
    #cdef int _m,_n
    #cdef int _p,_q

    def __init__(self, source, target, weights, gain):
        self._gain = gain
        self._source = source
        self._target = target
        self._weights = weights
        cdef int i,j,k
        cdef double[:] con
        cdef long[:] act
        effw = []
        for i in range(self._weights.shape[0]):
            con = np.asarray(self._weights[i])
            act = np.nonzero(con)[0] 
            effw.append(act)
        self._effweights = np.asarray(effw) 
        #self._m = self._target.shape[0]
        #self._n = self._target.shape[1]
        #self._p = self._source.shape[0]
        #self._q = self._source.shape[1]

    def flush(self):
        cdef int i
        for i in range(self._target.shape[0]):
            #for j in range(self._target.shape[1]):
            if 1:
                self._target[i] = 0.0

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def propagate(self):
        cdef int i,j
        cdef double[:] con
        cdef long[:] act
        for i in range(self._weights.shape[0]):
            #con = self._weights[i]
            #act = np.nonzero(con)[0] 
            #for j in act:
            for j in range(self._weights.shape[1]):
            #for j in self._weights[i]:
                self._target[i] += self._source[j] * self._weights[i,j] * self._gain

    property gain:
        """Gain of the connection"""
        def __get__(self):
            return self._gain
        def __set__(self, value):
            self._gain = value

    property source:
        """Source of the connection """
        def __get__(self):
            return np.asarray(self._source)

    property target:
        """Target of the connection (numpy array)"""
        def __get__(self):
            return np.asarray(self._target)

    property weights:
        """Weights matrix (numpy array)"""
        def __get__(self):
            return np.asarray(self._weights)
        def __set__(self, weights):
            self._weights = weights


