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

def obtain_plot_region(sp,maxcol,interval=0.2,bb0_coeff=3):
    maxnum = sp[0]
    maxrow = sp[0]//maxcol
    remains = sp[0]%maxcol
    aa=1/((1+interval)*maxcol+interval)
    hhaa=interval*aa
    bb0 = bb0_coeff*hhaa
    bb = (1-bb0)/(interval+(1+interval)*(maxrow+(remains>0)))
    hhbb=interval*bb
    left_list=[ hhaa+(aa+hhaa)*i for i in range(maxcol) ]
    bottom_list=[bb0+(hhbb+bb)*(maxrow+(remains>0)-1-i) for i in range(maxrow+(remains>0))]
    outer_rect = maxcol*(maxrow+(remains>0))
    return interval,aa,hhaa,bb0,bb,hhbb,left_list,bottom_list,outer_rect,maxnum,maxrow,remains


def plot_subscatter(lft,btm,aa,bb,x,y,species_name,fontdict,norm):            
    rect_line=[lft,btm,aa,bb]
    ax = plt.axes(rect_line)
    xy = np.vstack([x,y])
    a,b,c,d=get_statics(x,y)
    xymin=np.minimum(x.min(),y.min())
    xymax=np.maximum(x.max(),y.max())
    z=get_zvalue(xy,xymax,xymin,search_ratio=0.01)
    dense=ax.scatter(x,y,5,c=z,alpha=0.5,linewidths=0,cmap='jet',norm=norm)          
    diag=ax.plot((xymin,xymax),(xymin,xymax),color='#000000',linewidth=0.8)
    regression=ax.plot((xymin,xymax),(c*xymin+d,c*xymax+d),color='#FF5533',linewidth=2,linestyle='--')                  
    delta=(xymax-xymin)/6
    ax.text(xymin+delta/2,xymax-delta*2,s='$\mathrm{R}^2$=%0.2f\nRMSE=%0.2e\nslope=%0.2f'%(a,b,c),ha='left',fontdict=fontdict)
    tx2=ax.text(xymax-delta/2,xymin+delta/2,s=species_name,ha='right',fontdict=fontdict)
    #tx2.set_bbox(dict(facecolor='steelblue', alpha=0.2, edgecolor='white'))
    ax.set_xlim([xymin,xymax])
    ax.set_yticks(ax.get_xticks())
    ax.set_ylim([xymin,xymax])
    labels=ax.get_xticklabels()+ax.get_yticklabels()
    fontsize_ticks=fontdict['size']-2
    [label.set_fontsize(fontsize_ticks) for label in labels]
    #[label.set_fontname('Times New Roman') for label in labels]

def scatter_plot(A,B,namelist,savedir,maxcol=3,cb=True,width=15):
    sp=A.shape
    if sp != B.shape:
        print('check input please')
        return
    #
    interval,aa,hhaa,bb0,bb,hhbb,left_list,bottom_list,outer_rect,maxnum,maxrow,remains=obtain_plot_region(sp=sp,maxcol=maxcol,interval=0.2,bb0_coeff=3)
    fig = plt.figure(figsize = (width,width*hhaa/hhbb))
    fontsize= min(int(fig.get_figwidth()*72//maxcol/15*72/96),int(fig.get_figheight()*72//(maxrow+remains>0)/15*72/96))
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
    name_index=0
    for row in range(maxrow):
        for col in range(maxcol):
            lft=left_list[col]
            btm=bottom_list[row]
            x=A[row*maxcol+col]
            y=B[row*maxcol+col]
            species_name=namelist[name_index]
            plot_subscatter(lft,btm,aa,bb,x,y,species_name,fontdict=fontdict,norm=norm)
            name_index+=1

    if remains>0:
        for subs in range(remains):
            lft=left_list[subs]
            btm=bottom_list[maxrow]
            x=A[maxrow*maxcol+subs]
            y=B[maxrow*maxcol+subs]
            species_name=namelist[name_index]
            plot_subscatter(lft,btm,aa,bb,x,y,species_name,fontdict=fontdict,norm=norm)
            name_index+=1
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    if cb:
        bar_ax=fig.add_axes([hhaa, 0.5*bb0, 1-2*hhaa, 0.2*hhaa])#位置[左,下,右,上]
        clb=plt.colorbar(sm,cax=bar_ax, orientation="horizontal",alpha=0.5) 
        clb.ax.xaxis.set_major_locator(MultipleLocator(5))
        clb.ax.tick_params(labelsize=1*fontsize)
    fig.savefig(savedir,dpi=500)
    return fig