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
from model import *
from display import *
import matplotlib.pyplot as plt
# --- Parameter
ms         = 0.001
settling   = 500*ms
trial      = 2500*ms
dt         = 1*ms

#debug      = True
testing    = True
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



def weights(shape):
    Wmin, Wmax = 0.25, 0.75
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)

def set_session():
    global W, WM, connections
    global cues_mot, cues_cog, cues_value, cues_reward 
    global mot_value
    global P, R, MP
    global DT_COG, DT_MOT
    global W_arr, WM_arr
    P, R, MP = [], [], []
    DT_COG, DT_MOT = [], []
    W_arr, WM_arr = [], [] 
    # These weights will change (learning)
    W = weights(4)
    WM = weights(4) #change_motor_learning

    global CTX, STR, STN, GPI, THL
    CTX = AssociativeStructure(
                     tau=tau, rest=- 3.0, noise=0.010, activation=clamp )
    STR = AssociativeStructure(
                     tau=tau, rest=  0.0, noise=0.001, activation=sigmoid )
    STN = Structure( tau=tau, rest=-10.0, noise=0.001, activation=clamp )
    GPI = Structure( tau=tau, rest=+10.0, noise=0.030, activation=clamp )
    THL = Structure( tau=tau, rest=-40.0, noise=0.001, activation=clamp )
    global structures
    structures = (CTX, STR, STN, GPI, THL)
    
    connections = []
    connections.append(OneToOne( CTX.cog.V, STR.cog.Isyn, W,            gain=+1.0 ))
    if mot_learning:
            connections.append(OneToOne( CTX.mot.V, STR.mot.Isyn, WM,           gain=+1.0 ))  #change_motor_learning
    else:
            connections.append(OneToOne( CTX.mot.V, STR.mot.Isyn, weights(4),   gain=+1.0 )) #uncomment_for_revert
    connections.append(OneToOne( CTX.ass.V, STR.ass.Isyn, weights(4*4), gain=+1.0 ))
    connections.append(CogToAss( CTX.cog.V, STR.ass.Isyn, weights(4),   gain=+0.2 ))
    connections.append(MotToAss( CTX.mot.V, STR.ass.Isyn, weights(4),   gain=+0.2 ))
    connections.append(OneToOne( CTX.cog.V, STN.cog.Isyn, np.ones(4),   gain=+1.0 ))
    connections.append(OneToOne( CTX.mot.V, STN.mot.Isyn, np.ones(4),   gain=+1.0 ))
    connections.append(OneToOne( STR.cog.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ))
    connections.append(OneToOne( STR.mot.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ))
    connections.append(AssToCog( STR.ass.V, GPI.cog.Isyn, np.ones(4),   gain=-2.0 ))
    connections.append(AssToMot( STR.ass.V, GPI.mot.Isyn, np.ones(4),   gain=-2.0 ))
    connections.append(OneToAll( STN.cog.V, GPI.cog.Isyn, np.ones(4),   gain=+1.0 ))
    connections.append(OneToAll( STN.mot.V, GPI.mot.Isyn, np.ones(4),   gain=+1.0 ))
    connections.append(OneToOne( GPI.cog.V, THL.cog.Isyn, np.ones(4),   gain=-0.5 ))
    connections.append(OneToOne( GPI.mot.V, THL.mot.Isyn, np.ones(4),   gain=-0.5 ))
    connections.append(OneToOne( THL.cog.V, CTX.cog.Isyn, np.ones(4),   gain=+1.0 ))
    connections.append(OneToOne( THL.mot.V, CTX.mot.Isyn, np.ones(4),   gain=+1.0 ))
    connections.append(OneToOne( CTX.cog.V, THL.cog.Isyn, np.ones(4),   gain=+0.4 ))
    connections.append(OneToOne( CTX.mot.V, THL.mot.Isyn, np.ones(4),   gain=+0.4 ))


    cues_mot = np.array([0,1,2,3])
    cues_cog = np.array([0,1,2,3])
    cues_value = np.ones(4) * 0.5
    mot_value = np.ones(4) * 0.5   #change_motor_learning
    cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0

def set_trial():
    global cues_mot, cues_cog, cues_values, cues_reward
    global CTX, STR, STN, GPI, THL

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
    global mot_value  #change_motor_learning 	
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

    if mot_learning:
        # Motor Learn                                                                                #change_motor_learning  
        error_mot = reward - mot_value[mot_choice]                                                   #change_motor_learning 
        # Update motor values                                                                        #change_motor_learning
        mot_value[mot_choice] += error_mot* alpha_m                                                  #change_motor_learning
        mot_lrate = alpha_LTP_m if error_mot > 0 else alpha_LTD_m                                        #change_motor_learning
        dw_mot = error_mot * mot_lrate * STR.mot.V[mot_choice]                                       #change_motor_learning
        WM[mot_choice] = WM[mot_choice] + dw_mot * (WM[mot_choice]-WM_min)*(WM_max-WM[mot_choice])   #change_motor_learning

    if not debug: return

    # Just for displaying ordered cue
    oc1,oc2 = min(c1,c2), max(c1,c2)
    if choice == oc1:
        print "Choice:          [%d] / %d  (good)" % (oc1,oc2)
    else:
        print "Choice:           %d / [%d] (bad)" % (oc1,oc2)
    print "Reward (%3d%%) :   %d" % (int(100*cues_reward[choice]),reward)
    print "Response time:    %d ms" % (time)



# 120 trials
if testing:
    num_trials = 40
else:
    num_trials = 120
     
### begin of run_session
def run_session():
    set_session()
    first = 0
    failed_trials = 0
    for j in range(num_trials):
        reset()

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
        cog_not_made = True
        mot_not_made = True
        for i in xrange(i0,i1):
            iterate(dt)
            if cog_not_made and CTX.cog.delta > threshold :
                cog_not_made = False
                cog_time = i - 500 
            # Test if a decision has been made
            if mot_not_made and CTX.mot.delta > threshold:
                learn(time=i-500, debug=debug)
                mot_not_made = False
                mot_time = i - 500
            if not (cog_not_made or mot_not_made):
                break

        if cog_not_made:
            cog_time = i1 
        if mot_not_made:
            mot_time = i1 
            P.append(0)
            failed_trials += 1

        DT_COG.append(cog_time)
        DT_MOT.append(mot_time)

        if first:
            for k in range(np.array(W).size) : W_arr[k].append(W[k])
            for k in range(np.array(WM).size): WM_arr[k].append(WM[k])
        else:
            for k in range(np.array(W).size) : W_arr.append([W[k]])
            for k in range(np.array(WM).size): WM_arr.append([WM[k]])
            first = 1


        # Debug information
        if debug:
            if i >= (i1-1):
                print "! Failed trial"
            print


    return failed_trials
### end of run_session


if testing:
    num_sessions = 4
else:
    num_sessions = 250 

trials_set = range(num_trials) 
cog_ft = np.zeros(num_sessions)
cog_mp = np.zeros((num_sessions,num_trials))
# diff b/w cog dec and mot dec = mot_dec_time - cog_dec_time
cog_dec_time = np.zeros((num_sessions,num_trials))
mot_dec_time = np.zeros((num_sessions,num_trials))
cog_mr = np.zeros(num_sessions)

for k in range(num_sessions):
    cog_ft[k] = run_session()
    cog_mp[k] = P 
    cog_dec_time[k] = DT_COG
    mot_dec_time[k] = DT_MOT
    cog_mr[k] = np.array(R)[-20:].mean()
    print "[Session %d] Mean performance: %.3f" % (k,np.mean(cog_mp[k]))
    print "[Session %d] Mean reward: %.3f" % (k,cog_mr[k])
    print

print "Failed trials in Cog learning - %d" % np.mean(cog_ft)
####
TP = np.zeros((num_sessions,num_trials))
DTCOG = np.zeros((num_sessions,num_trials))
DTMOT = np.zeros((num_sessions,num_trials))
for i in range(num_sessions):
    for j in range(num_trials):
        TP[i,j] = np.mean(cog_mp[i,j])
        DTCOG[i,j] = np.mean(cog_dec_time[i,j])
        DTMOT[i,j] = np.mean(mot_dec_time[i,j])

title = 'COG Learning'
plot_weights(1, 1, W_arr, WM_arr, num_trials, title)

title = """
          alpha[_c = %.3f, _LTP = %.3f, _LTD = %.3f]
        """ % (alpha_c, alpha_LTP, alpha_LTD) 
plot_performance(2, 1, num_trials, TP, title)

title = 'COG Learning'
plot_diff_decision_times(3, 1, num_trials, DTCOG, DTMOT, trial, title)

plot_decision_times(4,1, num_trials, DTCOG, DTMOT, trial, title)

#Turn mot_learning on and re-run
mot_learning = True
same_rates = False
if mot_learning:
    if not same_rates:
        alpha_c    = alpha_c/2 
        alpha_m    = alpha_m/2 
        alpha_LTP  = alpha_LTP/2
        alpha_LTD  = alpha_LTD/2
        alpha_LTP_m  = alpha_LTP_m/2
        alpha_LTD_m  = alpha_LTD_m/2

    if testing:
        num_trials = 40
    else:
        num_trials = 240 

    mot_cog_ft = np.zeros(num_sessions)
    mot_cog_mp = np.zeros((num_sessions,num_trials))
    # diff b/w cog dec and mot dec = mot_dec_time - cog_dec_time
    cog_dec_time = np.zeros((num_sessions,num_trials))
    mot_dec_time = np.zeros((num_sessions,num_trials))
    mot_cog_mr = np.zeros(num_sessions)
    for k in range(num_sessions):
        mot_cog_ft[k] = run_session()
        mot_cog_mp[k] = P
        cog_dec_time[k] = DT_COG
        mot_dec_time[k] = DT_MOT
        mot_cog_mr[k] = np.array(R)[-20:].mean()
        print "[Session %d] Mean performance: %.3f" % (k,np.mean(mot_cog_mp[k]))
        print "[Session %d] Mean reward: %.3f" % (k,mot_cog_mr[k])

    print "Mean failed trials in Cog & Mot learning - %d" % np.mean(mot_cog_ft)
    ####
    TP = np.zeros((num_sessions,num_trials))
    DTCOG = np.zeros((num_sessions,num_trials))
    DTMOT = np.zeros((num_sessions,num_trials))
    for i in range(num_sessions):
        for j in range(num_trials):
            TP[i,j] = np.mean(mot_cog_mp[i,j])
            DTCOG[i,j] = np.mean(cog_dec_time[i,j])
            DTMOT[i,j] = np.mean(mot_dec_time[i,j])


    title = 'COG & MOT Learning'
    plot_weights(1, 2, W_arr, WM_arr, num_trials, title)

    title = """
              alpha[_c = %.3f, _LTP = %.3f, _LTD = %.3f, _m = %.3f, _LTP_m = %.3f, _LTD_m = %.3f]
            """ % (alpha_c, alpha_LTP, alpha_LTD, alpha_m, alpha_LTP_m, alpha_LTD_m) 
    plot_performance(2, 2, num_trials, TP, title)

    title = 'COG & MOT Learning'
    plot_diff_decision_times(3, 2, num_trials, DTCOG, DTMOT, trial, title)

    plot_decision_times(4,2,num_trials, DTCOG, DTMOT, trial, title)

# END OF ALL SIMULATIONS
figname = 'lat-final-'+str(num_trials)+'-2500-both-'+ ('same' if same_rates else 'diff') +'-figure-'

if testing: figname = 'test-'+figname

for i in 1+np.arange(4):
    plt.figure(i)
    plt.tight_layout()
    plt.savefig(figname + "-"+str(i)+".pdf")

print "Only Cognitive learning"
print "Failed trials   : %d" % cog_ft.mean()
#print "Mean performance: %.3f" % cog_mp.mean()
print "Mean reward:      %.3f" % cog_mr.mean()
print "------------------------"

if mot_learning:
    print "Both Motor & Cognitive learning"
    print "Failed trials   : %d" % mot_cog_ft.mean()
#    print "Mean performance: %.3f" % mot_cog_mp.mean()
    print "Mean reward:      %.3f" % mot_cog_mr.mean()

print "Threshold %d" % threshold
plt.show()
