import numpy as np
import matplotlib.pyplot as plt

#possible_cues = ([3, 0],[3, 2],[3, 1],[2, 0],[2, 1],[1, 0])
possible_cues = ([2])
saliences = np.array([0,10,15,20,25,30,35])
delays = (0,10,20,25,30,35,40)
all_perf = np.load("mean_perf_salience_delay_allcues.npy")

index = np.arange(np.size(saliences+1))+0.25
bar_width = 0.45

opacity = 0.4
error_config = {'ecolor': '0.3'}
i = 0
for st in possible_cues: #all_perf.dtype.names:
    stim = str(st)
    plt.figure(1)
    #perf_salience = all_perf[stim]["salience"][0] 
    perf_salience = np.load("mean_perf_salience.npy")
    #perf_delay = all_perf[stim]["delay"][0] 
    perf_delay = np.load("mean_perf_delay.npy")
    ps_mean = perf_salience.mean(axis=1)
    ps_var = perf_salience.var(axis=1)
    pd_mean = perf_delay.mean(axis=1)
    pd_var = perf_delay.var(axis=1)
    rects1 = plt.bar(index+i*bar_width, ps_mean, bar_width,
                 alpha=opacity,
                 color='b',
                 yerr=ps_var,
                 error_kw=error_config,
                 label='Salience')
    plt.figure(2)
    rects2 = plt.bar(index+i*bar_width, pd_mean, bar_width,
                 alpha=opacity,
                 color='r',
                 yerr=pd_var,
                 error_kw=error_config,
                 label='Delay', clip_on=False)
    i = i+1



plt.figure(1)
plt.ylim(0,1.2)
plt.xlabel('% Increase in salience of worst cue B', fontsize=18)
plt.ylabel('Performance', fontsize=18)
#plt.title('Performance with change in salience for cues [A, B]')
plt.xticks(index + bar_width/2, np.around(saliences))

plt.tight_layout()
plt.savefig("plot-cosyne-performance-salience.svg")


plt.figure(2)
plt.ylim(0,1.2)
plt.xlabel('Delay (in ms) in presentation of best cue A', fontsize=18)
plt.ylabel('Performance', fontsize=18)
#plt.title('Performance with change in delay for cues [A, B]')
plt.xticks(index + bar_width/2, delays)

plt.tight_layout()
plt.savefig("plot-cosyne-performance-delay.svg")
plt.show()

