#!/usr/bin/python

import numpy as np
from model import *
from display import *
import matplotlib.pyplot as plt
import threading
from time import gmtime, strftime
# --- Parameter
ms         = 0.001
settling   = 500*ms
trial      = 2500*ms
dt         = 1*ms

both=False
#debug      = True
debug      = False
mot_learning = False
#mot_learning = True
threshold  = 40
alpha_c    = 0.025
alpha_m    = alpha_c            #change_motor_learning
alpha_LTP  = 0.004
alpha_LTD  = 0.002
alpha_LTP_m  = alpha_LTP
alpha_LTD_m  = alpha_LTD
Wmin, Wmax = 0.25, 0.75
WM_min, WM_max = 0.25, 0.75   #change_motor_learning
tau        = 0.01
clamp      = Clamp(min=0, max=1000)
sigmoid    = Sigmoid(Vmin=0, Vmax=20, Vh=16, Vc=3)
cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0

def weights(shape):
    Wmin, Wmax = 0.25, 0.75
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)

W = weights(4)
WM = weights(4)

def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)

def get_structures():
    CTX = AssociativeStructure(
                     tau=tau, rest=- 3.0, noise=0.010, activation=clamp )
    STR = AssociativeStructure(
                     tau=tau, rest=  0.0, noise=0.001, activation=sigmoid )
    STN = Structure( tau=tau, rest=-10.0, noise=0.001, activation=clamp )
    GPI = Structure( tau=tau, rest=+10.0, noise=0.030, activation=clamp )
    THL = Structure( tau=tau, rest=-40.0, noise=0.001, activation=clamp )
    structures = (CTX, STR, STN, GPI, THL)
    return structures

class Session (threading.Thread):
    def __init__(self, thread_count, num_sessions, trials_per_session):
        threading.Thread.__init__(self)
        self._thread_count = thread_count
        self._structures = get_structures()
        (self._CTX, self._STR, self._STN, self._GPI, self._THL) = self._structures
        self._ALL_W, self._ALL_WM = [],[]
        self._W = weights(4)
        self._WM = weights(4)
        for w in range(self._W.size): self._ALL_W.append([])
        for w in range(self._WM.size): self._ALL_WM.append([])
        self._ALL_P = []
        self._ALL_R = []
        self._connections = []
        self.get_connections()
        self._trials_per_session = trials_per_session 
        self._num_sessions = num_sessions
        self._cues_value = np.ones(4) * 0.5
        self._mot_value = np.ones(4) * 0.5


    def reset(self):
        for structure in self._structures:
            structure.reset()

    def get_connections(self):
        self._connections.append(OneToOne( self._CTX.cog.V, self._STR.cog.Isyn, self._W,            gain=+1.0 ))
        if 1:
                self._connections.append(OneToOne( self._CTX.mot.V, self._STR.mot.Isyn, self._WM,           gain=+1.0 ))  #change_motor_learning
        else:
                self._connections.append(OneToOne( self._CTX.mot.V, self._STR.mot.Isyn, weights(4),   gain=+1.0 )) #uncomment_for_revert
        self._connections.append(OneToOne( self._CTX.ass.V, self._STR.ass.Isyn, weights(4*4), gain=+1.0 ))
        self._connections.append(CogToAss( self._CTX.cog.V, self._STR.ass.Isyn, weights(4),   gain=+0.2 ))
        self._connections.append(MotToAss( self._CTX.mot.V, self._STR.ass.Isyn, weights(4),   gain=+0.2 ))
        self._connections.append(OneToOne( self._CTX.cog.V, self._STN.cog.Isyn, np.ones(4),   gain=+1.0 ))
        self._connections.append(OneToOne( self._CTX.mot.V, self._STN.mot.Isyn, np.ones(4),   gain=+1.0 ))
        self._connections.append(OneToOne( self._STR.cog.V, self._GPI.cog.Isyn, np.ones(4),   gain=-2.0 ))
        self._connections.append(OneToOne( self._STR.mot.V, self._GPI.mot.Isyn, np.ones(4),   gain=-2.0 ))
        self._connections.append(AssToCog( self._STR.ass.V, self._GPI.cog.Isyn, np.ones(4),   gain=-2.0 ))
        self._connections.append(AssToMot( self._STR.ass.V, self._GPI.mot.Isyn, np.ones(4),   gain=-2.0 ))
        self._connections.append(OneToAll( self._STN.cog.V, self._GPI.cog.Isyn, np.ones(4),   gain=+1.0 ))
        self._connections.append(OneToAll( self._STN.mot.V, self._GPI.mot.Isyn, np.ones(4),   gain=+1.0 ))
        self._connections.append(OneToOne( self._GPI.cog.V, self._THL.cog.Isyn, np.ones(4),   gain=-0.5 ))
        self._connections.append(OneToOne( self._GPI.mot.V, self._THL.mot.Isyn, np.ones(4),   gain=-0.5 ))
        self._connections.append(OneToOne( self._THL.cog.V, self._CTX.cog.Isyn, np.ones(4),   gain=+1.0 ))
        self._connections.append(OneToOne( self._THL.mot.V, self._CTX.mot.Isyn, np.ones(4),   gain=+1.0 ))
        self._connections.append(OneToOne( self._CTX.cog.V, self._THL.cog.Isyn, np.ones(4),   gain=+0.4 ))
        self._connections.append(OneToOne( self._CTX.mot.V, self._THL.mot.Isyn, np.ones(4),   gain=+0.4 ))


    def iterate(self,dt):

        # Flush connections
        for connection in self._connections:
            connection.flush()

        # Propagate activities
        for connection in self._connections:
            connection.propagate()

        # Compute new activities
        for structure in self._structures:
            structure.evaluate(dt)

    def set_trial(self, cues_cog, cues_mot):
        c1,c2 = cues_cog[:2]
        m1,m2 = cues_mot[:2]
        v = 7
        noise = 0.01
        self._CTX.mot.Iext = 0
        self._CTX.cog.Iext = 0
        self._CTX.ass.Iext = 0
        self._CTX.mot.Iext[m1]  = v + np.random.normal(0,v*noise)
        self._CTX.mot.Iext[m2]  = v + np.random.normal(0,v*noise)
        self._CTX.cog.Iext[c1]  = v + np.random.normal(0,v*noise)
        self._CTX.cog.Iext[c2]  = v + np.random.normal(0,v*noise)
        self._CTX.ass.Iext[c1*4+m1] = v + np.random.normal(0,v*noise)
        self._CTX.ass.Iext[c2*4+m2] = v + np.random.normal(0,v*noise)

    def learn(self, cues_cog, cues_mot, P, R, time, debug=True):
        # A motor decision has been made
        c1, c2 = cues_cog[:2]
        m1, m2 = cues_mot[:2]
        mot_choice = np.argmax(self._CTX.mot.V)
        cog_choice = np.argmax(self._CTX.cog.V)

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
        error = reward - self._cues_value[choice]

        # Update cues values
        self._cues_value[choice] += error* alpha_c

        # Learn
        lrate = alpha_LTP if error > 0 else alpha_LTD
        dw = error * lrate * self._STR.cog.V[choice]
        self._W[choice] = self._W[choice] + dw * (self._W[choice]-Wmin)*(Wmax-self._W[choice])

        if mot_learning:
            # Motor Learn                                                                                #change_motor_learning
            error_mot = reward - self._mot_value[mot_choice]                                                   #change_motor_learning
            # Update motor values                                                                        #change_motor_learning
            self._mot_value[mot_choice] += error_mot* alpha_m                                                  #change_motor_learning
            mot_lrate = alpha_LTP_m if error_mot > 0 else alpha_LTD_m                                        #change_motor_learning
            dw_mot = error_mot * mot_lrate * self._STR.mot.V[mot_choice]                                       #change_motor_learning
            self._WM[mot_choice] = self._WM[mot_choice] + dw_mot * (self._WM[mot_choice]-WM_min)*(WM_max-self._WM[mot_choice])   #change_motor_learning

        if not debug: return

        # Just for displaying ordered cue
        oc1,oc2 = min(c1,c2), max(c1,c2)
        if choice == oc1:
            print "Choice:          [%d] / %d  (good)" % (oc1,oc2)
        else:
            print "Choice:           %d / [%d] (bad)" % (oc1,oc2)
        print "Reward (%3d%%) :   %d" % (int(100*cues_reward[choice]),reward)
        print "Response time:    %d ms" % (time)

    def run(self):
        print "Starting session thread - %d" % self._thread_count
        # All combinations of cues or positions
        Z = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
        # 20 x all cues combinations
        C = np.repeat(np.arange(6),self._trials_per_session/6)
        # 20 x all cues positions
        M = np.repeat(np.arange(6),self._trials_per_session/6)


        for s in range(self._num_sessions):
            P, R = [], []
            np.random.shuffle(C)
            np.random.shuffle(M)
            self._W[...] = weights(4)
            self._WM[...] = weights(4)
            WS, WMS = [],[]
            self._cues_value = np.ones(4) * 0.5
            self._mot_value = np.ones(4) * 0.5

            for j in range(self._trials_per_session):
                cues_cog = Z[C[j]]
                cues_mot = Z[M[j]]

                self.reset()
                # Settling phase (500ms)
                i0 = 0
                i1 = i0+int(settling/dt)
                for i in xrange(i0,i1):
                    self.iterate(dt)

                # Trial setup
                self.set_trial(cues_cog, cues_mot)

                decision = False
                # Learning phase (2500ms)
                i0 = int(settling/dt)
                i1 = i0+int(trial/dt)
                for i in xrange(i0,i1):
                    self.iterate(dt)
                    # Test if a decision has been made
                    if self._CTX.mot.delta > threshold:
                        self.learn(cues_cog, cues_mot, P, R, time=i-500, debug=debug)
                        decision = True
                        break
                if not decision:
                    P.append(0)
                    R.append(0)
                WS.append(np.array(self._W))
                WMS.append(np.array(self._WM))

                # Debug information
                if debug:
                    if i >= (i1-1):
                        print "! Failed trial"
                        failed_trials += 1
                    print
            session_weights = np.array(WS)
            session_motor_weights = np.array(WMS)
            for w in range(W.size): 
                self._ALL_W[w].append(session_weights[:,w])
            for wm in range(WM.size): 
                self._ALL_WM[wm].append(session_motor_weights[:,wm])

            self._ALL_P.append(np.array(P))
            self._ALL_R.append(np.array(R))

        threadLock.acquire()
        ALL_P.append(self._ALL_P)
        ALL_R.append(self._ALL_R)
        for w in range(W.size): 
            ALL_W[w].append(np.array(self._ALL_W[w]))
        for wm in range(WM.size): 
            ALL_WM[wm].append(np.array(self._ALL_WM[wm]))
        threadLock.release()
        print "Completed session thread - %d" % self._thread_count

####
def display_weights(fig_id, trials_per_session, ALL_W, ALL_WM, mot_learning, show_plt):
    trials_set = 1+np.arange(trials_per_session)
    plt.figure(fig_id)
    m = ''
    if mot_learning :
        plt.subplot(2,2,3)
        m = 'MOT & '
    else : plt.subplot(2,2,1)
    plt.plot(trials_set, np.array(ALL_W[0]).mean(axis=0), color='r', label='W0')
    plt.plot(trials_set, np.array(ALL_W[1]).mean(axis=0), color='b', label='W1')
    plt.plot(trials_set, np.array(ALL_W[2]).mean(axis=0), color='g', label='W2')
    plt.plot(trials_set, np.array(ALL_W[3]).mean(axis=0), color='c', label='W3')
    plt.title(m + 'COG Learning - COG Weights')
    plt.ylim(0.48,0.60)
    plt.legend(loc=2)
    if mot_learning :
        plt.subplot(2,2,4)
    else :
        plt.subplot(2,2,2)
    plt.plot(trials_set, np.array(ALL_WM[0]).mean(axis=0), color='r', label='D0')
    plt.plot(trials_set, np.array(ALL_WM[1]).mean(axis=0), color='b', label='D1')
    plt.plot(trials_set, np.array(ALL_WM[2]).mean(axis=0), color='g', label='D2')
    plt.plot(trials_set, np.array(ALL_WM[3]).mean(axis=0), color='c', label='D3')
    plt.title(m + 'COG Learning - MOT Weights')
    plt.legend(loc=2)
    plt.ylim(0.48,0.60)
    if show_plt:
        plt.show()

def display_performance(fig_id, trials_per_session, TP, title, mot_learning, show_plt):
    plt.figure(fig_id,figsize=(12,5.5))
    if mot_learning :
        ax = plt.subplot(212)
        color = 'r'
    else :
        ax = plt.subplot(211)
        color = 'b'
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="in")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="in")

    X = 1+np.arange(trials_per_session)
    plt.plot(X, TP.mean(axis=0), c=color, lw=2)
    plt.plot(X, TP.mean(axis=0)+TP.var(axis=0), c=color,lw=.5)
    plt.plot(X, TP.mean(axis=0)-TP.var(axis=0), c=color,lw=.5)
    plt.fill_between(X, TP.mean(axis=0)+TP.var(axis=0),
                        TP.mean(axis=0)-TP.var(axis=0), color=color, alpha=.1)
    #plt.xlabel("Trial number", fontsize=16)
    plt.ylabel("Performance", fontsize=16)
    plt.ylim(0,1.0)
    plt.xlim(1,trials_per_session)
    plt.title(title)
    if show_plt:
        plt.show()

# init simulation

threadLock = threading.Lock()
num_sessions = 50
sessions_per_thread = 25
num_threads = (num_sessions/sessions_per_thread)+min(num_sessions%sessions_per_thread,1) 
trials_per_session = 240 
ALL_P, ALL_R = [],[]
ALL_W, ALL_WM = [],[]

def run_simulation(end=False):
    global ALL_P, ALL_R, ALL_W, ALL_WM
    ALL_P, ALL_R = [],[]
    ALL_W, ALL_WM = [],[]
    for w in range(W.size): ALL_W.append([])
    for w in range(WM.size): ALL_WM.append([])
    sessions = []

    for i in range(num_threads):
        s = Session(i, num_sessions/num_threads, trials_per_session)
        s.start()
        sessions.append(s)

    for t in sessions:
        t.join()

    ALL_P = np.vstack(ALL_P)
    ALL_R = np.vstack(ALL_R)
    for i in range(W.size): ALL_W[i] = np.vstack(ALL_W[i])
    for i in range(WM.size): ALL_WM[i] = np.vstack(ALL_WM[i])
    ####
    TP = np.zeros((num_sessions,trials_per_session))
    for i in range(num_sessions):
        for j in range(trials_per_session):
            TP[i,j] = np.mean(ALL_P[i,j])
    # Plot the variation of weights over each trial as learning happens
    display_weights(1, trials_per_session, ALL_W, ALL_WM, mot_learning, False)
    # Plot the mean performance for each trial over all sessions
    title = 'alpha[_c = '+str(alpha_c)+', _LTP = '+str(alpha_LTP)+', _LTD = '+str(alpha_LTD)+']'
    display_performance(2, trials_per_session, TP, title, mot_learning, end)

both=True
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
run_simulation(not both)
if both:
    mot_learning = True
if mot_learning:
    trials_per_session = 240
    run_simulation(end=True)
print strftime("%Y-%m-%d %H:%M:%S", gmtime())
