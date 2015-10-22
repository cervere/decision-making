from exutils import *
from display import *
import time

plot = True
learning = False
threshold  = 40
alpha_c    = 0.025
alpha_LTP  = 0.004
alpha_LTD  = 0.002
Wmin, Wmax = 0.25, 0.75
duration = 2 
clamp   = Clamp(min=0, max=1000)
sigmoid = Sigmoid(Vmin=0, Vmax=20, Vh=16, Vc=3)


popCTX = numOfCues*popPerCueCTX
popSTR = numOfCues*popPerCueSTR

CTX_GUASSIAN_INPUT = getNormal(popPerCueCTX)
CTX_GUASSIAN_INPUT_2D = get2DNormal(popPerCueCTX, popPerCueCTX)

CTX = AssociativeStructure(popCTX,
                 tau=CTX_tau, rest=CTX_rest, noise=0.010, activation=clamp )
STR = AssociativeStructure(
                 tau=STR_tau, rest=STR_rest, noise=0.001, activation=sigmoid )
STN = Structure( tau=STN_tau, rest=STN_rest, noise=0.001, activation=clamp )
GPI = Structure( tau=GPI_tau, rest=GPI_rest, noise=0.030, activation=clamp )
THL = Structure( tau=THL_tau, rest=THL_rest, noise=0.001, activation=clamp )
structures = (CTX, STR, STN, GPI, THL)

connections = [
    OneToOne(CTX.cog.U, STR.cog.Isyn, 1.0*strToCtx, clipWeights=False),
    OneToOne(CTX.mot.U, STR.mot.Isyn, 1.0*strToCtx, clipWeights=False),
    AscToAsc(CTX.ass.U, STR.ass.Isyn, 1.0*strToCtx*strToCtx, clipWeights=False),
    CogToAss(CTX.cog.U, STR.ass.Isyn, gain=+0.2*strToCtx, clipWeights=False),
    MotToAss(CTX.mot.U, STR.ass.Isyn, gain=+0.2*strToCtx, clipWeights=False),
    OneToOne(CTX.cog.U, STN.cog.Isyn, 1.0*stnToCtx),
    OneToOne(CTX.mot.U, STN.mot.Isyn, 1.0*stnToCtx),
    OneToOne(STR.cog.U, GPI.cog.Isyn, -2.0*gpiToStr),
#    OneToOne(STR.mot.U, GPI.mot.Isyn, -2.0*gpiToStr),
    AssToCog(STR.ass.U, GPI.cog.Isyn, gain=-2.0*gpiToStr*gpiToStr),
#    AssToMot(STR.ass.U, GPI.mot.Isyn, gain=-2.0*gpiToStr*gpiToStr),
    OneToAll(STN.cog.U, GPI.cog.Isyn, gain=+1.0 ),
    OneToAll(STN.mot.U, GPI.mot.Isyn, gain=+1.0 ),
    OneToOne(GPI.cog.U, THL.cog.Isyn, -0.5),
    OneToOne(GPI.mot.U, THL.mot.Isyn, -0.5),
    OneToOne(THL.cog.U, CTX.cog.Isyn, +1.0),
    OneToOne(THL.mot.U, CTX.mot.Isyn, +1.0),
    OneToOne(CTX.cog.U, THL.cog.Isyn, 0.4*thlToCtx),
    OneToOne(CTX.mot.U, THL.mot.Isyn, 0.4*thlToCtx)
]

ctxStrCog = connections[0]

W = ctxStrCog.weights
(tar, sou) = W.shape
eachsou = sou/numOfCues
eachtar = tar/numOfCues
each = sou/tar
def updateWeights(choice, dw):
    global W
    for j in range(eachtar):
        cue = choice*eachsou
        wc = W[choice*eachtar+j, cue+j*each:cue+j*each+each]
        wc = wc + dw * (wc-Wmin)*(Wmax-wc)
        W[choice*eachtar+j, cue+j*each:cue+j*each+each] = wc
    ctxStrCog._weights = W

dtype = [ ("CTX", [("mot", float, numOfCues), ("cog", float, numOfCues), ("ass", float, numOfCues*numOfCues)]),
          ("STR", [("mot", float, numOfCues), ("cog", float, numOfCues), ("ass", float, numOfCues*numOfCues)]),
          ("GPI", [("mot", float, 4), ("cog", float, 4)]),
          ("THL", [("mot", float, 4), ("cog", float, 4)]),
          ("STN", [("mot", float, 4), ("cog", float, 4)])]

history = np.zeros(duration, dtype)

def getExtInput():
    v = 18 
    noise = 0.01
    return (CTX_GUASSIAN_INPUT )*(np.random.normal(v,noise)) 
#+  np.random.normal(0,noise)

def get2DExtInput():
    v = 18 
    noise = 0.01
    return (CTX_GUASSIAN_INPUT_2D)*(np.random.normal(v,noise)) 
#+ np.random.normal(0,v*noise)

#cues_mot = np.array([0,1,2,3])
#cues_cog = np.array([0,1,2,3])
cues_value = np.ones(4) * 0.5
cues_reward = np.array([3.,2.,1.,0.0])/4.
Z = [[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]]
C, M = [], []

for i in range(20):
    pos = np.arange(6)
    np.random.shuffle(pos)
    # 20 x all cues combinations
    for j in pos : C.append(j)
    pos = np.arange(6)
    np.random.shuffle(pos)
    # 20 x all cues combinations
    for j in pos : M.append(j)

def start_trial():
    global cues_mot, cues_cog
    global c1,c2,m1,m2
    c1,c2 = cues_cog
    m1,m2 = cues_mot
    #c1,c2 = 0,1 
    #m1,m2 = 2,3
    cp1, cp2 = c1 * popPerCueCTX, c2 * popPerCueCTX
    mp1, mp2 = m1 * popPerCueCTX, m2 * popPerCueCTX
    CTX.cog.Iext = 0
    CTX.mot.Iext = 0
    CTX.ass.Iext = 0
    CTX.cog.Iext[cp1:cp1+popPerCueCTX] = getExtInput()
    CTX.cog.Iext[cp2:cp2+popPerCueCTX] = getExtInput()
    CTX.mot.Iext[mp1:mp1+popPerCueCTX] = getExtInput()
    CTX.mot.Iext[mp2:mp2+popPerCueCTX] = getExtInput()
    ext = np.zeros((popCTX,popCTX))
    ext[cp1:cp1+popPerCueCTX,mp1:mp1+popPerCueCTX] = get2DExtInput()
    ext[cp2:cp2+popPerCueCTX,mp2:mp2+popPerCueCTX] = get2DExtInput()
    ext = np.reshape(ext,(popCTX*popCTX,1))
    for i in range(ext.shape[0]):
        CTX.ass.Iext[i] = ext[i] 
    if plot:
        plt.figure(1)
        plt.subplot(211)
        plot_per_neuron(CTX.cog.Iext[:], CTX.mot.Iext[:], "Iext", "External input - Cortex - across trial")

global flu, prop, evaluate
flu = 0.0
prop = 0.0
evaluate = 0.0
con_times = np.zeros(len(connections))
def iterate(dt):
    global connections, structures
    global flu, prop, evaluate
    global con_times

    f = time.time()
    # Flush connections
    for connection in connections:
        connection.flush()

    flu += (time.time() - f)
    # Propagate activities
    f = time.time()
    #for connection in connections:
    for i in range(len(connections)):
        s = time.time()
        connections[i].propagate()
        con_times[i] += (time.time() - s)
    prop += (time.time() - f)

    # Compute new activities
    f = time.time()
    for structure in structures:
        structure.evaluate(dt)
    evaluate += (time.time() - f)

    

def reset():
    global cues_values, structures
    cues_value = np.ones(4) * 0.5
    for structure in structures:
        structure.reset()



def print_act(t):
    print "%d CTX Isyn %s" % (t*1000, sumActivity(CTX.mot['Isyn']))
    print "%d STR Isyn %s" %(t*1000, sumActivity(STR.mot['Isyn']))
    print "%d STR mot Z %s" %(t*1000, sumActivity(STR.mot['U']))
    print "%d STR ass Z %s" %(t*1000, sumActivity(STR.ass['U']))
    print "%d STN U %s" %(t*1000, sumActivity(STN.mot.U))
    print "%d GPI Isyn %s" %(t*1000, sumActivity(GPI.mot['Isyn']))
    print "%d GPI U %s" %(t*1000, sumActivity(GPI.mot.V))
    print "%d GPI Z %s" %(t*1000, sumActivity(GPI.mot['U']))
    print "%d THL Isyn %s" %(t*1000, sumActivity(THL.mot['Isyn']))
    #print "%d THL U %s" %(t*1000, sumActivity(THL.mot.U))
    #print "%d STR U %s" % (t*1000, sumActivity(STR.mot.U))

def plot_per_neuron(cog, mot, ylabel, title):
    plt.plot(1+np.arange(numOfCues*popPerCueCTX), cog, c='r', label='cognitive cortex')
    plt.plot(1+np.arange(numOfCues*popPerCueCTX), mot, c='b', label='motor cortex')
    plt.xlabel("Neuron label")
    plt.ylabel(ylabel)
    plt.xlim(1,numOfCues*popPerCueCTX)
    dire = ['Up', 'Right', 'Down', 'Left']
    shape = ['\/', '<>', '+', 'O']
    n,l = [], []
    n.append(0)
    for i in range(numOfCues):
        n.append(n[2*i] + popPerCueCTX/2)
        n.append(n[2*i+1] + popPerCueCTX/2 )
    n[0] = 1
    l.append(str(1) + '\n(\n(')
    for i in range(np.array(n).size):
        if i%2 == 1:
            l.append(str(n[i])+ '\n' + shape[i/2] +'\n'+ dire[i/2])
        elif i > 0 and i < (np.array(n).size - 1) : l.append(str(n[i])+'\n)(\n)(')
    l.append(str(n[i]) + '\n)')
    plt.xticks(n,l)
    plt.title(title)
    plt.legend(frameon=False, loc='upper center')

def reset_trial(t):
    print sumActivity(CTX.cog.U)
    print sumActivity(CTX.mot.U)
    if plot:
        plt.figure(1)
        plt.subplot(212)
        plot_per_neuron(np.maximum(CTX.cog.U,0.0), np.maximum(CTX.mot.U,0.0), "Activity (Hz)", 'Activity of each neuron - Cortex - end of trial')
    CTX.cog.Iext = 0
    CTX.mot.Iext = 0
    CTX.ass.Iext = 0

def meanActivity(population):
    percue = np.reshape(population, (numOfCues, population.size/numOfCues))
    return percue.mean(axis=1)

def sumActivity(population):
    if 1 : return population
    percue = np.reshape(population, (numOfCues, population.size/numOfCues))
    return percue.mean(axis=1)

P, R = [], []

def learn(choice, reward):
    # Compute prediction error
    error = reward - cues_value[choice]
    # Update cues values
    cues_value[choice] += error* alpha_c
    # Learn
    lrate = alpha_LTP if error > 0 else alpha_LTD
    dw = error * lrate * STR.cog.V[choice]
    updateWeights(choice, dw)

def register(t):
    history["CTX"]["cog"][t*1000] = meanActivity(CTX.cog.U)
    history["CTX"]["mot"][t*1000] = meanActivity(CTX.mot.U)
    global c1,c2,m1,m2
    global decision_not_made
    meanAct = meanActivity(CTX.mot.U)
    if decision_not_made and (meanAct.max() - meanAct.min() > 40.0):
        decision_not_made = False
        mot_choice = np.argmax(meanAct)
        cog_choice = np.argmax(meanAct)
        if mot_choice == m1:
            cog_choice = c1
        elif mot_choice == m2:
            cog_choice = c2
        if cog_choice == min(c1,c2):
            P.append(1)
            st = "good"
        else : 
            P.append(0)
            st = "bad"
        reward = np.random.uniform(0,1) < cues_reward[cog_choice]
        R.append(reward)
        print "choice made -%d- of (%d,%d) - %s" % (cog_choice, c1, c2, st)
        if learning : 
            learn(cog_choice, reward)
            end()

#print ctxStrCog.weights
global decision_not_made
start = time.time()
dt = 0.001
for j in range(1):
    cues_cog = Z[C[i]]
    cues_mot = Z[M[i]]
    reset()
    print "Running trial - %d" % j 
    decision_not_made = True
    for i in xrange(0,duration):
        #print "iterating %d" % i
        iterate(dt)
    #    if i > 10 and i < 20:
    #    print "%d - %s" % (i, str(STR.cog.U))
        print_act(i*0.001)
    start_trial()
    for i in xrange(500,500):
        print "iterating %d" % i
        iterate(dt)
    #stop_trial()
    #for i in xrange(2500,3000):
    #    print "iterating %d" % i
    #    iterate(dt)
history["CTX"]["mot"]   = CTX.mot.history[:duration]
history["CTX"]["cog"]   = CTX.cog.history[:duration]

end = time.time()
np.save("history.npy", history)
print "%d secs for the session" % (end - start)
print "(flush - %d, propagate - %d, evaluate - %d) secs for the session" % (flu, prop, evaluate)
#print con_times
print np.sum(con_times)
#print "Mean performance %f" % np.array(P).mean()
last = np.array(P).size/6
np.save("performance.npy",  P)
#print "Mean performance last %d trials %.3f" % (last, np.array(P)[-last:].mean())
#print "Mean reward %f" % np.array(P).mean()
#print ctxStrCog.weights

if plot:
    plt.tight_layout()
    display_ctx(history, 0.5)
