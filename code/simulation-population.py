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
import matplotlib.pyplot as plt
from model import *

# --- Parameter
ms         = 0.001
settling   = 500*ms
trial      = 2500*ms
dt         = 1*ms

debug      = True
threshold  = 40
alpha_c    = 0.025
alpha_LTP  = 0.004
alpha_LTD  = 0.002
Wmin, Wmax = 0.25, 0.75
tau        = 0.01
clamp      = Clamp(min=0, max=1000)
sigmoid    = Sigmoid(Vmin=0, Vmax=20, Vh=16, Vc=3)

CTX = AssociativeStructure(
                 tau=tau, rest=- 3.0, noise=0.010, activation=clamp )
STR = AssociativeStructure(
                 tau=tau, rest=  0.0, noise=0.001, activation=sigmoid )
STN = Structure( tau=tau, rest=-10.0, noise=0.001, activation=clamp )
GPI = Structure( tau=tau, rest=+10.0, noise=0.030, activation=clamp )
THL = Structure( tau=tau, rest=-40.0, noise=0.001, activation=clamp )
structures = (CTX, STR, STN, GPI, THL)

def get_gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) )

num_cues = 4
neurons_per_cue = 250
mu_cues = np.zeros(num_cues)

stim_norm = np.zeros((num_cues,neurons_per_cue)) #For computational simplicity
stim_cog = np.zeros((num_cues,neurons_per_cue)) #For computational simplicity

for i in range(num_cues):
    mu_cues[i] = (neurons_per_cue/2) + (i * neurons_per_cue)

print mu_cues
sigma = (neurons_per_cue/2)/3 #In a normal distribution, 3 * sigma covers 99.7% of population


for i in range(num_cues):
    x = []
    stim = np.zeros(neurons_per_cue)
    mu = mu_cues[i]
    for j in range(neurons_per_cue):
        index = (mu - neurons_per_cue/2) + j
        stim[j] = get_gaussian(index , mu, sigma)
    stim_norm[i] = stim

def weights(shape):
    Wmin, Wmax = 0.25, 0.75
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)


# These weights will change (learning)
W = weights(4)

connections = [
    OneToOne( CTX.cog.V, STR.cog.Isyn, W,            gain=+1.0 ),
    OneToOne( CTX.mot.V, STR.mot.Isyn, weights(4),   gain=+1.0 ),
    OneToOne( CTX.ass.V, STR.ass.Isyn, weights(4*4), gain=+1.0 ),
    CogToAss( CTX.cog.V, STR.ass.Isyn, weights(4),   gain=+0.2 ),
    MotToAss( CTX.mot.V, STR.ass.Isyn, weights(4),   gain=+0.2 ),
    OneToOne( CTX.cog.V, STN.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( CTX.mot.V, STN.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( STR.cog.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ),
    OneToOne( STR.mot.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ),
    AssToCog( STR.ass.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ),
    AssToMot( STR.ass.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ),
    OneToAll( STN.cog.V, GPI.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToAll( STN.mot.V, GPI.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( GPI.cog.V, THL.cog.Isyn, np.ones(4),   gain=-0.5 ),
    OneToOne( GPI.mot.V, THL.mot.Isyn, np.ones(4),   gain=-0.5 ),
    OneToOne( THL.cog.V, CTX.cog.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( THL.mot.V, CTX.mot.Isyn, np.ones(4),   gain=+1.0 ),
    OneToOne( CTX.cog.V, THL.cog.Isyn, np.ones(4),   gain=+0.4 ),
    OneToOne( CTX.mot.V, THL.mot.Isyn, np.ones(4),   gain=+0.4 ),
]


cues_mot = np.array([0,1,2,3])
cues_cog = np.array([0,1,2,3])
cues_value = np.ones(4) * 0.5
cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0

def set_trial():
    global cues_mot, cues_cog, cues_values, cues_reward, stim_cog

    np.random.shuffle(cues_cog)
    np.random.shuffle(cues_mot)
    c1,c2 = cues_cog[:2]
    m1,m2 = cues_mot[:2]
    v = 7
    noise = 0.01
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0
    CTX.mot.Iext[m1]  = v + np.random.normal(0,v*noise)
    CTX.mot.Iext[m2]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c1]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c2]  = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c1*4+m1] = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c2*4+m2] = v + np.random.normal(0,v*noise)
    stim_cog = np.zeros((num_cues,neurons_per_cue)) #For computational simplicity
    stim_cog[c1] = stim_norm[c1]
    stim_cog[c2] = stim_norm[c2]



def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)


def iterate(dt):
    global connections, structures

    # Flush connections
    for connection in connections:
        connection.flush()

    # Propagate activities
    for connection in connections:
        connection.propagate()

    # Compute new activities
    for structure in structures:
        structure.evaluate(dt)


def reset():
    global cues_values, structures
    for structure in structures:
        structure.reset()


def learn(time, debug=True):
    # A motor decision has been made
    c1, c2 = cues_cog[:2]
    m1, m2 = cues_mot[:2]
    mot_choice = np.argmax(CTX.mot.V)
    cog_choice = np.argmax(CTX.cog.V)

    # The actual cognitive choice may differ from the cognitive choice
    # Only the motor decision can designate the chosen cue
    if mot_choice == m1:
        choice = c1
    else:
        choice = c2

    if choice == min(c1,c2):
        P.append(1)
    else:
        P.append(0)

    # Compute reward
    reward = np.random.uniform(0,1) < cues_reward[choice]
    R.append(reward)

    # Compute prediction error
    #error = cues_reward[choice] - cues_value[choice]
    error = reward - cues_value[choice]

    # Update cues values
    cues_value[choice] += error* alpha_c

    # Learn
    lrate = alpha_LTP if error > 0 else alpha_LTD
    dw = error * lrate * STR.cog.V[choice]
    W[choice] = W[choice] + dw * (W[choice]-Wmin)*(Wmax-W[choice])

    if not debug: return

    # Just for displaying ordered cue
    oc1,oc2 = min(c1,c2), max(c1,c2)
    if choice == oc1:
        print "Choice:          [%d] / %d  (good)" % (oc1,oc2)
    else:
        print "Choice:           %d / [%d] (bad)" % (oc1,oc2)
    print "Reward (%3d%%) :   %d" % (int(100*cues_reward[choice]),reward)
    print "Mean performance: %.3f" % np.array(P)[-20:].mean()
    print "Mean reward:      %.3f" % np.array(R).mean()
    print "Response time:    %d ms" % (time)


def run_trial():
    # Settling phase (500ms)
    i0 = 0
    i1 = i0+int(settling/dt)
    for i in xrange(i0,i1):
        iterate(dt)

    # Trial setup
    set_trial()

    # Learning phase (2500ms)
    i0 = int(settling/dt)
    i1 = i0+int(trial/dt)
    for i in xrange(i0,i1):
        iterate(dt)
        # Test if a decision has been made
        if CTX.mot.delta > threshold:
            break
    # Debug information
    if debug:
        if i >= (i1-1):
            print "! Failed trial"
        print
    return i

P, R = [], []
# 120 trials
for j in range(120):
    reset()

    dec_time = run_trial()
    learn(time=dec_time-int(settling/dt), debug=debug)

colors = ['r', 'g', 'b', 'c']

f, axarr = plt.subplots(3,2)
for j in range(3):
    reset()
    i = run_trial()
    c1, c2 = cues_cog[:2]
    m1, m2 = cues_mot[:2]
    mot_choice = np.argmax(CTX.mot.V)

    # The actual cognitive choice may differ from the cognitive choice
    # Only the motor decision can designate the chosen cue
    if mot_choice == m1:
        choice = c1
    else:
        choice = c2

    resp_cog = np.zeros((num_cues,neurons_per_cue)) #For computational simplicity
    resp_cog[choice] = stim_norm[choice]

    for i in range(num_cues):
        axarr[j, 0].plot(range(i * neurons_per_cue,i * neurons_per_cue + neurons_per_cue), stim_cog[i], linewidth=2, color=colors[i])
        axarr[j, 1].plot(range(i * neurons_per_cue,i * neurons_per_cue + neurons_per_cue), resp_cog[i], linewidth=2, color=colors[i])
plt.setp([a.get_yticklabels() for a in axarr[:, 0]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)
plt.setp(axarr[0, 0].get_yticklabels(), visible=True)
axarr[0, 0].set_title('Presented')
axarr[0, 1].set_title('Selected')
plt.savefig("cues-population.pdf")
plt.show()

