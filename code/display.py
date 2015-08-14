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

def display_ctx(history, duration=3.0, filename=None):
    fig = plt.figure(figsize=(12,5))
    plt.subplots_adjust(bottom=0.15)

    timesteps = np.linspace(0,duration, len(history))

    fig.patch.set_facecolor('.9')
    ax = plt.subplot(1,1,1)

    plt.plot(timesteps, history["CTX"]["cog"][:,0],c='r', label="Cognitive Cortex")
    plt.plot(timesteps, history["CTX"]["cog"][:,1],c='r')
    plt.plot(timesteps, history["CTX"]["cog"][:,2],c='r')
    plt.plot(timesteps, history["CTX"]["cog"][:,3],c='r')
    plt.plot(timesteps, history["CTX"]["mot"][:,0],c='b', label="Motor Cortex")
    plt.plot(timesteps, history["CTX"]["mot"][:,1],c='b')
    plt.plot(timesteps, history["CTX"]["mot"][:,2],c='b')
    plt.plot(timesteps, history["CTX"]["mot"][:,3],c='b')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Activity (Hz)")
    plt.legend(frameon=False, loc='upper left')
    plt.xlim(0.0,duration)
    plt.ylim(0.0,60.0)

    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
               ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])

    if filename is not None:
        plt.savefig(filename)
    plt.show()



def display_all(history, duration=3.0, filename=None):
    fig = plt.figure(figsize=(18,12))
    fig.patch.set_facecolor('1.0')

    timesteps = np.linspace(0,duration, len(history))

    def subplot(rows,cols,n, alpha=0.0):
        ax = plt.subplot(rows,cols,n)
        ax.patch.set_facecolor("k")
        ax.patch.set_alpha(alpha)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(direction="outward")
        return ax

    ax = subplot(5,3,1)
    ax.set_title("Motor", fontsize=24)
    ax.set_ylabel("STN", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["STN"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,2)
    ax.set_title("Cognitive", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["STN"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,3,alpha=0)
    ax.set_title("Associative", fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_color('none')


    ax = subplot(5,3,4)
    ax.set_ylabel("Cortex", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["CTX"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,5)
    for i in range(4):
        plt.plot(timesteps, history["CTX"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,6)
    for i in range(16):
        plt.plot(timesteps, history["CTX"]["ass"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,7)
    ax.set_ylabel("Striatum", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["STR"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,8)
    for i in range(4):
        plt.plot(timesteps, history["STR"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,9)
    for i in range(16):
        plt.plot(timesteps, history["STR"]["ass"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,10)
    ax.set_ylabel("GPi", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["GPI"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,11)
    for i in range(4):
        plt.plot(timesteps, history["GPI"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,13)
    ax.set_ylabel("Thalamus", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, history["THL"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,14)
    for i in range(4):
        plt.plot(timesteps, history["THL"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    if filename is not None:
        plt.savefig(filename)
    plt.show()

#####################  DISPLAY METHODS  ########################################"
def plot_weights(fignum, figpos, W_arr, WM_arr, num_trials, title):
    # Plot the variation of weights over each trial as learning happens
    plt.figure(fignum)
    pos = 220 + (2*(figpos/2)+1)
    plt.subplot(pos)
    trials_set = 1+np.arange(num_trials)
    colors = ['r','b','g','c']
    for i in range(4):
        plt.plot(trials_set, W_arr[i], color=colors[i], label='W'+str(i))
    plt.title(title + '-COG Wts')
    plt.ylim(0.48,0.60)
    plt.legend(loc=2)
    plt.subplot(pos + 1)
    for i in range(4):
        plt.plot(trials_set, WM_arr[i], color=colors[i], label='D'+str(i))
    plt.title(title + '-MOT Wts')
    plt.ylim(0.48,0.60)
    plt.legend(loc=2)

#####################  DISPLAY METHODS  ########################################"
def plot_lines(fignum, figpos, data, trials_set, labels, title=''):
    # Plot the variation of weights over each trial as learning happens
    plt.figure(fignum)
    pos = 220 + figpos
    plt.subplot(pos)
    colors = ['r','b','g','c','m','y']
    for i in range(np.size(data)/np.size(trials_set)):
        if np.size(data) == np.size(trials_set):
            plt.plot(trials_set, data, color=colors[i], label=str(labels))
        else:
            plt.plot(trials_set, data[i], color=colors[i], label=str(labels[i]))
    plt.title(title)
    plt.ylim(0,2)
    plt.yticks([0.0,0.5,1.0,1.5,2.0],['0.0','0.5','1.0','',''])
    plt.legend(loc=2)

def plot_performance(fignum, figpos, num_trials, TP, title):
    # Plot the mean performance for each trial over all sessions
    plt.figure(fignum)
    pos = 210 + figpos
    ax = plt.subplot(pos)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="in")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="in")

    X = 1+np.arange(num_trials)
    plt.plot(X, TP.mean(axis=0), c='r', lw=2)
    plt.plot(X, TP.mean(axis=0)+TP.var(axis=0), c='r',lw=.5)
    plt.plot(X, TP.mean(axis=0)-TP.var(axis=0), c='r',lw=.5)
    plt.fill_between(X, TP.mean(axis=0)+TP.var(axis=0),
                        TP.mean(axis=0)-TP.var(axis=0), color='r', alpha=.1)
    if figpos > 1 : plt.xlabel("Trial number", fontsize=16)
    plt.ylabel("Performance", fontsize=16)
    plt.ylim(0,1.0)
    plt.xlim(1,num_trials)
    plt.title(title)


def autolabel(ax, rects):
    # attach some text labels
    for rect1, rect2 in zip(rects[0],rects[1]):
        height1 = rect1.get_height()
        height2 = rect2.get_height()
        ax.text(rect2.get_x()+rect2.get_width()/2., 1.05*height2, '%d'%(height2 - height1),
                ha='center', va='bottom')

def plot_diff_decision_times(fignum, figpos, num_trials, DTCOG, DTMOT, trial, title):
    # Plot mean decision times for COG and MOT over each trial
    plt.figure(fignum)
    pos = 210 + figpos
    ax = plt.subplot(pos)
    ind = np.arange(10)
    width = 0.35
    cog_var = DTCOG.var(axis=0)
    mot_var = DTMOT.var(axis=0)
    rects1 = ax.bar(ind, DTCOG.mean(axis=0)[-10:], width, color='r')
    rects2 = ax.bar(ind+width, DTMOT.mean(axis=0)[-10:], width, color='b')
    ax.set_ylabel('Decision time')
    if figpos > 1: ax.set_xlabel('Trial Number', fontsize=16)
    ax.set_title('Decision times - ' + title)
    ax.set_xticks(ind+width)
    ax.set_xticklabels(np.arange(num_trials-9,num_trials+1))
    ax.legend( (rects1[0], rects2[0]), ('COG', 'MOTOR') )
    plt.ylim(0,trial*1000)
    autolabel(ax, [rects1, rects2])

def plot_decision_times(fignum, figpos, num_trials, DTCOG, DTMOT, trial, title):
    # Plot the decision times of each trial over all sessions
    plt.figure(fignum)
    pos = 210 + figpos
    ax = plt.subplot(pos)
    ax.patch.set_facecolor("w")
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_tick_params(direction="in")
    ax.xaxis.set_ticks_position('bottom')
    ax.xaxis.set_tick_params(direction="in")

    X = 1+np.arange(num_trials)
    cog_times_mean = DTCOG.mean(axis=0)
    tot_cog_mean = np.mean(cog_times_mean[-20])
    mot_times_mean = DTMOT.mean(axis=0)
    tot_mot_mean = np.mean(mot_times_mean[-20])
    plt.plot(X, cog_times_mean, c='b', lw=2, label='COG')
    plt.plot([X[0],X[num_trials-1]],[tot_cog_mean,tot_cog_mean], 'b--', lw=2)
    plt.plot(X, mot_times_mean, c='r', lw=2, label='MOT')
    plt.plot([X[0],X[num_trials-1]],[tot_mot_mean,tot_mot_mean], 'r--', lw=2)
    plt.ylabel("Decision Time", fontsize=16)
    if figpos > 1: plt.xlabel("Trial Number", fontsize=16)
    plt.ylim(0,0.75*trial*1000)
    plt.xlim(1,num_trials)
    plt.legend(loc=2)
    plt.title(title)

#####################  END OF DISPLAY METHODS  ########################################"

