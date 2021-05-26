import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from sklearn import metrics
from scipy.stats import linregress
import statsmodels.api as sm
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
def get_statics(x,y):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    r2_score = r_value**2
    mse =  metrics.mean_squared_error(x,y)
    rmse = np.sqrt(mse)
    return r2_score,rmse,slope,intercept


def get_zvalue(xy,xymax,xymin,search_ratio=0.01):
    z=np.zeros(xy.shape[1])
    gap=(xymax-xymin)*search_ratio
    for id in range(xy.shape[1]):
        xt=xy[:,id][0]
        yt=xy[:,id][1]
        z[id]=np.sum((xy[0,:]<xt+gap)*(xy[1,:]<yt+gap)*(xy[0,:]>xt-gap)*(xy[1,:]>yt-gap))
    return z

def sci_notation(floatnumber,acc=2):
    expo_part = np.floor(np.log(abs(floatnumber))/np.log(10))
    main_part = floatnumber/10**expo_part
    command="'%0."+str(acc)+"f'% (main_part)"
    #print(command)
    string1=eval(command)
    string2=str(int(expo_part))
    return string1,string2
    
def plot_singlescatter(x,y,species_name,fontdict,norm):            
    
    ax = plt.axes([0.2,0.2,0.7,0.7])
    xy = np.vstack([x,y])
    a,b,c,d=get_statics(x,y)
    xymin=np.minimum(x.min(),y.min())
    xymax=np.maximum(x.max(),y.max())
    z=get_zvalue(xy,xymax,xymin,search_ratio=0.01)
    dense=ax.scatter(x,y,15,c=z,alpha=1,linewidths=0,cmap='jet',norm=norm)          
    diag=ax.plot((xymin,xymax),(xymin,xymax),color='#000000',linewidth=3)
    regression=ax.plot((xymin,xymax),(c*xymin+d,c*xymax+d),color='#FF5533',linewidth=3,linestyle='--')                  
    delta=(xymax-xymin)/6
    b1,b2=sci_notation(b)
    ax.text(xymin+delta/3,xymax-delta*2,s='$\mathrm{R}^2$=%0.2f\nRMSE=%s$\\times 10^{%s}$\nslope=%0.2f\nintercept=%0.2f\nN=%d'%(a,b1,b2,c,d,len(x)),ha='left',fontdict=fontdict)
    #ax.text(xymin+delta/3,xymax-delta*2,s='$\mathrm{R}^2$=%0.2f\nRMSE=%0.2e\nslope=%0.2f\nintercept=%0.2f\nN=%d'%(a,b,c,d,len(x)),ha='left',fontdict=fontdict)
    tx2=ax.text(xymax-delta/2,xymin+delta/2,s=species_name,ha='right',fontdict=fontdict)
    ax.set_xlim([xymin,xymax])
    ax.set_yticks(ax.get_xticks())
    ax.set_ylim([xymin,xymax])
    labels=ax.get_xticklabels()+ax.get_yticklabels()
    fontsize_ticks=fontdict['size']-5
    [label.set_fontsize(fontsize_ticks) for label in labels]
    ax.xaxis.set_label_coords(0.5,-0.12)
    ax.yaxis.set_label_coords(-0.2,0.5)
    ax.set_xlabel('Prediction (ppm)',fontdict)
    ax.set_ylabel('True value (ppm)',fontdict)


    
def single_plot(A,B,species_name,savedir,cb=True):
    sp=A.shape
    if sp != B.shape:
        print('check input please')
        return
    #

    fig = plt.figure(figsize = (5,5))
    fontsize= 14
    #fontsize_ticks=fontsize-2
    fontdict = {'family' : 'Arial',
                #       'color'  : 'darkred',
            #'mathtext.fontset':'stix',
            'color'  : 'black',
            'weight' : 'normal',
            'size'   : fontsize,            
                    }
    matplotlib.rcParams['mathtext.rm'] = 'Arial'
    matplotlib.rcParams['mathtext.it'] = 'Arial'
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    plot_singlescatter(A,B,species_name,fontdict=fontdict,norm=norm)
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    if cb:
        bar_ax=fig.add_axes()#位置[左,下,右,上]
        clb=plt.colorbar(sm,cax=bar_ax, orientation="horizontal",alpha=1) 
        clb.ax.xaxis.set_major_locator(MultipleLocator(5))
        clb.ax.tick_params(labelsize=1*fontsize)
    fig.savefig(savedir,dpi=500)
    return fig