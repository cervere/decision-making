import random
import time
import numpy as np
a=[]
for i in range(10000):
    a.append(random.randint(0,1))

b = np.array(a)
c=0
k=0.5
start = time.time()
for i in range(10000):
    c += b[i] * k 
end = time.time()
print c
print "%s milli secs" % str((end-start)*1000)
c = 0
nonzeroind = np.nonzero(b)[0]
start = time.time()
for j in nonzeroind:
    c += b[j] * k
end = time.time()
print c
print "%s milli secs" % str((end-start)*1000)
