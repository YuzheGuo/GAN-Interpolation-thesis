import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter

def scatter_plot(A,savedir,namelist,start_date,end_date,figshape=[16,7],fig_name='demo',label=["Time","Temperature ($^\circ$C)"],plot_type = ['scatter','line','line']):
    plt.rcParams['font.family'] = 'Arial'
    fontsize = 20
    plot = A
    color = ['black','blue', 'red','green','orange']
    style = [None,':','--',]
    fig, ax = plt.subplots(figsize=(figshape[0],figshape[1]))
    date = pd.date_range(start_date, end_date, freq='H')
    nmb = []
    nan_index = np.where(np.isnan(A[0]))[0]
    for k in range(1,plot.shape[0]):
        num_ob,num_si = 0,0
        for j in range(plot.shape[1]):
            if j not in nan_index:
                num_ob += plot[0][j]
                num_si += plot[k][j]
        nmb_result = (num_ob-num_si)/num_ob
        print(nmb_result,'and',num_ob,'and',num_si)
        nmb.append(nmb_result)
    for i in range(plot.shape[0]):
        if plot_type[i] == 'line':
            ax.plot(date, plot[i], label=namelist[i], color=color[i], linewidth=2.3,linestyle=style[i])
        if plot_type[i] == 'scatter':
            ax.scatter(date, plot[i], label=namelist[i], color=color[i])
    # plt.xticks(rotation=45)
    ax.legend(fontsize=fontsize, borderpad=0, frameon=0, ncol=plot.shape[0])
    ax.tick_params(which='minor', length=3, width=1.5)
    ax.tick_params(labelsize=20, length=6, width=1.5)
    ax.set_xlim(date[0] - 24 * date.freq, date[len(date) - 1] + 24 * date.freq)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.xaxis.set_major_locator(DayLocator(0, 4))
    ax.xaxis.set_minor_locator(DayLocator())
    ax.xaxis.set_major_formatter(DateFormatter('%m/%d'))
    ax.xaxis.set_label_coords(0.5,-0.12)
    ax.yaxis.set_label_coords(-0.07,0.5)
    plt.xlabel(label[0], fontsize=fontsize)
    plt.ylabel(label[1], fontsize=fontsize)
    for g in range(len(nmb)):
        plt.text(date[-60],289-g*2,'NMB{}:{}'.format(g,round(nmb[g],3)),fontsize=fontsize)
    plt.show()
    figname = savedir + '%s.png' % (fig_name)
    fig.savefig(figname, dpi=300)
    return fig



obsfile = np.loadtxt('/Applications/pythonfile/cmaq/phyunet/metobs_58461_T2.txt')
simfile = np.loadtxt('/Applications/pythonfile/cmaq/phyunet/metsim_58461_T2.txt')
obsfile = obsfile+273.15
simfile_1 = simfile+2
simfile[simfile > 361] = np.nan
obsfile[obsfile > 361] = np.nan
simfile = simfile[:-7]
obsfile = obsfile[7:]
simfile_1 = simfile+2
#以上为数据准备
#一下为参数设置
A = np.array((obsfile,simfile,simfile_1))
namelist = ['Observation','Simulation','Simulation1']
start_date = '2016-01-01 08:00:00'
end_date = '2016-02-01 00:00:00'
savedir = '/Applications/pythonfile/cmaq/phyunet/'
plot_type = ['scatter','line','line']
scatter_plot(A,savedir,namelist,start_date,end_date,figshape=[16,7],fig_name='demo',label=["Time","Temperature (K)"])



