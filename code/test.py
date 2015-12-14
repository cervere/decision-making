from guthrie_model import *

pop = 1024 

def getCTXGain():
    if pop == 1:
        return 7 
    elif pop == 4:
        return 4.45
    elif pop == 16:
        return 2.05
    else:
        return 1.775

source = Group(pop, 'dV/dt = (-V + Isyn + Iext - CTX_rest)/CTX_tau ; U = unoise(clamp(V) , CTX_noise); Isyn ; Iext ')
target = Group(popPerCueSTR, 'dV/dt = (-V + Isyn + Iext - CTX_rest)/CTX_tau ; U = unoise(clamp(V) , CTX_noise); Isyn ; Iext ')

gain = 0.002

C = DenseConnection(source('U'), target('Isyn'), gain * np.ones(pop))
v = 7
noise = 0.01
source['Iext'] = getCTXGain() * getNormal(pop) + np.random.normal(0,v*noise)
print source.Iext
source.evaluate()
C.propagate()
print C.output()
