import numpy as np
import matplotlib.pyplot as plt 

def get_gaussian(x, mu, sigma):
   return 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * sigma**2) ) 


mu, sigma = 63, 20.667

d = np.random.normal(mu, sigma, 125)
s = np.sort(d)

x, y = [], []
x.append(1)
y.append(get_gaussian(1, mu, sigma))

for i in s:
    if i >= 1 and i <= 125: 
        x.append(i)
        y.append(1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (i - mu)**2 / (2 * sigma**2) ))

x.append(125)
y.append(get_gaussian(125, mu, sigma))

print np.array(x).size
print np.array(y).size

plt.plot(x, y, linewidth=2, color='r')

x, y = [], []

for j in range(125):
    i = j + 1
    x.append(i)
    y.append(get_gaussian(i, mu, sigma))
for j in range(125,250):
    x.append(j+1)
    y.append(0)
for j in range(251,375):
    x.append(j+1)
    y.append(get_gaussian(j+1, mu + 250, sigma))

plt.plot(x, y, linewidth=2, color='b')

plt.show()
