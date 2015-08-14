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
from display import *

# --- Parameter
ms         = 0.001
settling   = 500*ms
trial      = 2500*ms
dt         = 1*ms

getWeights = True
getWeights = False
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

def weights(shape):
    Wmin, Wmax = 0.25, 0.75
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)


# These weights will change (learning)
if getWeights:
    W = weights(4)
else:
    W = np.load("cog-weights.npy")

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
num_trials = 120 

def set_trial():
    global cues_mot, cues_cog, cues_values, cues_reward

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

def first_stimulus(cues=[],salience=0):
    global cues_mot, cues_cog, cues_values, cues_reward
    global dc1
    np.random.shuffle(cues_mot)
    m1 = cues_mot[0]
    if np.size(cues) == 0:
        np.random.shuffle(cues_cog)
        c1,c2 = cues_cog[:2]
        c1 = max(c1,c2) 
    else:
        c1 = cues[0]
    dc1 = c1
    v = 7
    noise = 0.01
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0
    CTX.mot.Iext[m1]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c1]  = v + np.random.normal(0,v*noise) + salience
    CTX.ass.Iext[c1*4+m1] = v + np.random.normal(0,v*noise)

def second_stimulus(cues=[]):
    global dc2
    m2 = cues_mot[1]
    if np.size(cues) == 0:
        c1,c2 = cues_cog[:2]
        c2 = min(c1,c2) 
    else:
        c2 = cues[1]
    dc2 = c2
    v = 7
    noise = 0.01
    CTX.mot.Iext[m2]  = v + np.random.normal(0,v*noise)
    CTX.cog.Iext[c2]  = v + np.random.normal(0,v*noise)
    CTX.ass.Iext[c2*4+m2] = v + np.random.normal(0,v*noise)

def stop_trial():
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0

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


def update_and_learn(time, learn=True, debug=True):
    # A motor decision has been made
    cc1, cc2 = cues_cog[:2]
    if getWeights:
        c1 = cc1
        c2 = cc2
    else:
        c1 = max(cc1,cc2)
        c2 = min(cc1,cc2)
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

    if learn:
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

def run_session(stim=[], delay=0, salience=0):
    # 120 trials
    for j in range(num_trials):
        reset()

        # Settling phase (500ms)
        i0 = 0 
        i1 = i0+int(settling/dt)
        for i in xrange(i0,i1):
            iterate(dt)

        # Trial setup
        if getWeights:
            set_trial()
        else:
            first_stimulus(stim,salience)
        # Learning phase (2500ms)
        i0 = int(settling/dt)
        i1 = i0+int(trial/dt)
        for i in xrange(i0,i1):
            if not getWeights:
                if i == i0 + delay:
                    print "introducing second stimulus"
                    second_stimulus()
            iterate(dt)
            # Test if a decision has been made
            if CTX.mot.delta > threshold:
                update_and_learn(time=i-i0, learn=getWeights, debug=debug)
                break

        # Debug information
        if debug:
            if i >= (i1-1):
                print "! Failed trial"
            print

if getWeights:
    P, R = [], []
    run_session()
    print "Done with learning. Saving weights to cog-weights.npy"
    np.save("cog-weights.npy", W)
    exit()

possible_cues = ([1,0],[2,0],[3,0],[2,1],[3,1],[3,2])

# Introducing less rewarding stimulus with a salience
saliences = (0,0.5,1,1.5,2,2.5)
perf_for_salience = np.zeros((np.size(saliences),np.shape(possible_cues)[0])) 
delay = 0
for d in range(np.size(saliences)):
    salience = saliences[d]
    perf = np.zeros(np.shape(possible_cues)[0])
    for c in range(np.shape(possible_cues)[0]):
        stim = possible_cues[c]
        P, R = [], []
        run_session(stim, delay, salience)
        perf[c] = np.array(P).mean()
    perf_for_salience[d] = perf 
print perf_for_salience
np.save("perf_for_salience.npy",perf_for_salience)
plot_lines(1, 1, np.transpose(perf_for_salience)[:3], saliences, possible_cues[:3])
plt.ylabel('Performance')
plot_lines(1, 2, np.transpose(perf_for_salience)[3:5], saliences, possible_cues[3:5])
plt.xlabel('Salience')
plot_lines(1, 3, np.transpose(perf_for_salience)[5], saliences, possible_cues[5])
plt.ylabel('Performance')
plt.xlabel('Salience')
fig = plt.figure(1)
fig.suptitle("Performance - less rewarding stimulus first - with salience")

plt.savefig("performances-with-saliences.pdf")

# Introducing more rewarding stimulus with a delay
delays = (0,10,20,30,40,50,60,70)
perf_for_delay = np.zeros((np.size(delays),np.shape(possible_cues)[0])) 
salience = 0
for d in range(np.size(delays)):
    delay = delays[d]
    perf = np.zeros(np.shape(possible_cues)[0])
    for c in range(np.shape(possible_cues)[0]):
        stim = possible_cues[c]
        P, R = [], []
        run_session(stim, delay, salience)
        perf[c] = np.array(P).mean()
    perf_for_delay[d] = perf 
print perf_for_delay
np.save("perf_for_delay.npy",perf_for_delay)

plot_lines(2, 1, np.transpose(perf_for_delay)[:3], delays, possible_cues[:3])
plt.ylabel('Performance')
plot_lines(2, 2, np.transpose(perf_for_delay)[3:5], delays, possible_cues[3:5])
plt.xlabel('Delay (ms)')
plot_lines(2, 3, np.transpose(perf_for_delay)[5], delays, possible_cues[5])
plt.xlabel('Delay (ms)')
plt.ylabel('Performance')
fig = plt.figure(2)
fig.suptitle("Performance - more rewarding stimulus later - with delay")

plt.savefig("performances-with-delays.pdf")
plt.show()

exit()





