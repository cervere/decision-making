from guthrie_modelc import *
from display import *
import time

# Trial duration
duration =  3000.0*millisecond
pri = False
if duration*1000 < 50:
    pri = True
# Default Time resolution
dt = 1.0*millisecond

plot = True
learning = False
gaussian = True
threshold  = 40
alpha_c    = 0.025
alpha_LTP  = 0.004
alpha_LTD  = 0.002
Wmin, Wmax = 0.25, 0.75


popCTX = numOfCues*popPerCueCTX
popSTR = numOfCues*popPerCueSTR

if gaussian :
    #CTX_GUASSIAN_INPUT = getNormal(3*3*popPerCueCTX)[popPerCueCTX*1*4:popPerCueCTX*1*4+popPerCueCTX]
    CTX_GUASSIAN_INPUT = getNormal(popPerCueCTX)
    #CTX_GUASSIAN_INPUT_2D = get2DNormal(3*3*popPerCueCTX, 3*3*popPerCueCTX)[popPerCueCTX*1*4:popPerCueCTX*1*4+popPerCueCTX, popPerCueCTX*1*4:popPerCueCTX*1*4+popPerCueCTX]
    CTX_GUASSIAN_INPUT_2D = get2DNormal(popPerCueCTX, popPerCueCTX)
else :
    CTX_GUASSIAN_INPUT = np.ones(popPerCueCTX)
    CTX_GUASSIAN_INPUT_2D = np.ones((popPerCueCTX, popPerCueCTX))


CTX = AssociativeStructure("CTX", pop=numOfCues*popPerCueCTX)
STR = AssociativeStructure("STR", pop=numOfCues*popPerCueSTR)
STN = Structure("STN", pop=numOfCues*popPerCueSTN)
GPI = Structure("GPI", pop=numOfCues*popPerCueGPI)
THL = Structure("THL", pop=numOfCues*popPerCueTHL)

ctxStrCog = OneToOne(CTX.cog('U'), STR.cog('Isyn'), 1.0*strToCtx, clipWeights=True)
OneToOne(CTX.mot('U'), STR.mot('Isyn'), 1.0*strToCtx, clipWeights=True)
AscToAsc(CTX.ass('U'), STR.ass('Isyn'), 1.0*strToCtx*strToCtx, clipWeights=True)
CogToAss(CTX.cog('U'), STR.ass('Isyn'), gain=+0.2*strToCtx, clipWeights=True)
MotToAss(CTX.mot('U'), STR.ass('Isyn'), gain=+0.2*strToCtx, clipWeights=True)
OneToOne(CTX.cog('U'), STN.cog('Isyn'), 1.0*stnToCtx)
OneToOne(CTX.mot('U'), STN.mot('Isyn'), 1.0*stnToCtx)
OneToOne(STR.cog('U'), GPI.cog('Isyn'), -2.0*gpiToStr*popPerCueGPI)
OneToOne(STR.mot('U'), GPI.mot('Isyn'), -2.0*gpiToStr*popPerCueGPI)
AssToCog(STR.ass('U'), GPI.cog('Isyn'), gain=-2.0*gpiToStr*gpiToStr*2.0)
AssToMot(STR.ass('U'), GPI.mot('Isyn'), gain=-2.0*gpiToStr*gpiToStr*2.0)
#AssToCog(STR.ass('Z'), GPI.cog('Isyn'), gain=-2.0)
#AssToMot(STR.ass('Z'), GPI.mot('Isyn'), gain=-2.0)
OneToAll(STN.cog('U'), GPI.cog('Isyn'), gain=+1.0/popPerCueSTN)
OneToAll(STN.mot('U'), GPI.mot('Isyn'), gain=+1.0/popPerCueSTN)
OneToOne(GPI.cog('U'), THL.cog('Isyn'), -0.5)
OneToOne(GPI.mot('U'), THL.mot('Isyn'), -0.5)
OneToOne(THL.cog('U'), CTX.cog('Isyn'), +1.0)
OneToOne(THL.mot('U'), CTX.mot('Isyn'), +1.0)
OneToOne(CTX.cog('U'), THL.cog('Isyn'), 0.4*thlToCtx)
OneToOne(CTX.mot('U'), THL.mot('Isyn'), 0.4*thlToCtx)


W = ctxStrCog._weights
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

history = np.zeros(duration*1000+1, dtype)

def reset():
    clock.reset()
    for group in network.__default_network__._groups:
        group['U'] = 0
        group['V'] = 0
        group['Isyn'] = 0
    CTX.cog['Iext'] = 0
    CTX.mot['Iext'] = 0
    CTX.ass['Iext'] = 0

def getExtInput():
    v =  12
    noise = 0.01
    return (CTX_GUASSIAN_INPUT )*(np.random.normal(v,noise)) 
#+  np.random.normal(0,noise)

def get2DExtInput():
    v =  12
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



@clock.at(500*millisecond)
def set_trial(t):
    global cues_mot, cues_cog
    global c1,c2,m1,m2
    c1,c2 = cues_cog
    m1,m2 = cues_mot
    #c1,c2 = 0,1 
    #m1,m2 = 2,3
    cp1, cp2 = c1 * popPerCueCTX, c2 * popPerCueCTX
    mp1, mp2 = m1 * popPerCueCTX, m2 * popPerCueCTX
    CTX.cog['Iext'] = 0
    CTX.mot['Iext'] = 0
    CTX.ass['Iext'] = 0
    CTX.cog['Iext'][cp1:cp1+popPerCueCTX] = getExtInput()
    CTX.cog['Iext'][cp2:cp2+popPerCueCTX] = getExtInput()
    CTX.mot['Iext'][mp1:mp1+popPerCueCTX] = getExtInput()
    CTX.mot['Iext'][mp2:mp2+popPerCueCTX] = getExtInput()
    Iext = np.zeros((popCTX, popCTX))
    Iext[cp1:cp1+popPerCueCTX,mp1:mp1+popPerCueCTX] = get2DExtInput()
    Iext[cp2:cp2+popPerCueCTX,mp2:mp2+popPerCueCTX] = get2DExtInput()
    CTX.ass['Iext'] = CTX.ass['Iext'] + np.reshape(Iext, popCTX*popCTX)
    #CTX.ass['Iext'][cp1:cp1+popPerCueCTX,mp1:mp1+popPerCueCTX] = get2DExtInput()
    #CTX.ass['Iext'][cp2:cp2+popPerCueCTX,mp2:mp2+popPerCueCTX] = get2DExtInput()
    if plot:
        plt.figure(1)
        plt.subplot(211)
        plot_per_neuron(CTX.cog['Iext'], CTX.mot['Iext'], "Iext", "External input - Cortex - across trial")


def print_act(t):
    print "%d CTX Isyn %s" % (t*1000, sumActivity(CTX.mot['Isyn']))
    print "%d STR Isyn %s" %(t*1000, sumActivity(STR.mot['Isyn']))
    #print "%d STR mot Z %s" %(t*1000, sumActivity(STR.mot['Z']))
    #print "%d STR ass Z %s" %(t*1000, sumActivity(STR.ass['Z']))
    print "%d STR mot Z %s" %(t*1000, sumActivity(STR.mot['U']))
    print "%d STR ass Z %s" %(t*1000, sumActivity(STR.ass['U']))
    print "%d STN Isyn %s" %(t*1000, sumActivity(STN.mot['Isyn']))
    print "%d STN U %s" %(t*1000, sumActivity(STN.mot['U']))
    print "%d GPI Isyn %s" %(t*1000, sumActivity(GPI.mot['Isyn']))
    print "%d GPI V %s" %(t*1000, sumActivity(GPI.mot['V']))
    print "%d GPI U %s" %(t*1000, sumActivity(GPI.mot['U']))
    #print "%d GPI Z %s" %(t*1000, sumActivity(GPI.mot['Z']))
    print "%d THL Isyn %s" %(t*1000, sumActivity(THL.mot['Isyn']))
    #print "%d THL U %s" %(t*1000, sumActivity(THL.mot['U']))
    #print "%d STR U %s" % (t*1000, sumActivity(STR.mot['U']))

#@clock.at(1*millisecond)
def check_trial(t):
    print_act(t)

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

@clock.at(2500*millisecond)
def reset_trial(t):
    if plot:
        plt.figure(1)
        plt.subplot(212)
        plot_per_neuron(np.maximum(CTX.cog['U'],0.0), np.maximum(CTX.mot['U'],0.0), "Activity (Hz)", 'Activity of each neuron - Cortex - end of trial')
    CTX.cog['Iext'] = 0
    CTX.mot['Iext'] = 0
    CTX.ass['Iext'] = 0

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
    dw = error * lrate * STR.cog.U[choice]
    updateWeights(choice, dw)

def getDelta(meanAct):
    max1, max2 = 0, 0
    for i in range(meanAct.shape[0]):
        if meanAct[i] > max1:   max1 = meanAct[i]
        elif meanAct[i] > max2: max2 = meanAct[i]
    return max1 - max2

@after(clock.tick)
def register(t):
    #if (t*1000) > 498 and (t*1000) < 502:
    if pri:
        print_act(t)
    #print "%d - %s" % (int(t*1000), str(STR.cog['U']))    
    history["CTX"]["cog"][t*1000] = meanActivity(CTX.cog['U'])
    history["CTX"]["mot"][t*1000] = meanActivity(CTX.mot['U'])
    global c1,c2,m1,m2
    global decision_not_made
    meanAct = meanActivity(CTX.mot['U'])
    if decision_not_made and (getDelta(meanAct) > 40.0):
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
            #end()

#print ctxStrCog._weights
global decision_not_made
start = time.time()
for i in range(1):
    cues_cog = Z[C[i]]
    cues_mot = Z[M[i]]
    reset()
    decision_not_made = True
    run(time=duration,dt=dt)
end = time.time()
print "%d secs for the session" % (end - start)
print "Mean performance %f" % np.array(P).mean()
last = np.array(P).size/6
np.save("performance.npy",  P)
print "Mean performance last %d trials %.3f" % (last, np.array(P)[-last:].mean())
print "Mean reward %f" % np.array(P).mean()
np.save("weights-dana.npy", ctxStrCog._weights)
print ctxStrCog._weights
if plot:
    #plt.tight_layout()
    display_ctx(history, duration)
