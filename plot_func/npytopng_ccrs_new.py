# -*- coding: utf-8 -*-
"""
Created on Wed May 19 21:52:09 2021

@author: liujiayen
"""
#%%
import netCDF4 as nc
from netCDF4 import Dataset
#%%
import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
# import cartopy.io.shapereader as shapereader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as col
import matplotlib.cm as cm
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import os
from matplotlib import font_manager
import matplotlib.ticker as mticker
import shapely.geometry as sgeom
from copy import copy
#%%
dirpath = '/lustre/home/acct-esehazenet/hazenet-pg5/data/testoutput/numpy_matrix/conti_ts288_noweight_cctm_model_ydiff_batch750_0513_12:41:51.pth/conti_conc_SO2/'

outputpath = '/lustre/home/acct-esehazenet/hazenet-pg5/data/testoutput/paint_conti_ts288_noweight_cctm_model_ydiff_batch750_0513_12:41:51/conti_new/'


gridfile = Dataset('/lustre/home/acct-esehazenet/hazenet-pg1/5.3.1/data/mcip/CHINA_12KM/201601_modis_pbl08/GRIDCRO2D_CHINA_12KM_2016001.nc')
lon = np.array(gridfile['LON']).reshape(512,512)
lat = np.array(gridfile['LAT']).reshape(512,512)
def find_side(ls, side):
    """
 Given a shapely LineString which is assumed to be rectangular, return the
 line corresponding to a given side of the rectangle.

 """
    minx, miny, maxx, maxy = ls.bounds
    points = {'left': [(minx, miny), (minx, maxy)],
              'right': [(maxx, miny), (maxx, maxy)],
              'bottom': [(minx, miny), (maxx, miny)],
              'top': [(minx, maxy), (maxx, maxy)],}
    return sgeom.LineString(points[side])
def _lambert_ticks(ax, ticks, tick_location, line_constructor, tick_extractor):
    """得到一个兰伯特正形投影的轴的刻度位置和标签."""
    outline_patch = sgeom.LineString(ax.outline_patch.get_path().vertices.tolist())
    axis = find_side(outline_patch, tick_location)
    n_steps = 30
    extent = ax.get_extent(ccrs.PlateCarree())
    _ticks = []
    for t in ticks:
        xy = line_constructor(t, n_steps, extent)
        proj_xyz = ax.projection.transform_points(ccrs.Geodetic(), xy[:, 0], xy[:, 1])
        xyt = proj_xyz[..., :2]
        ls = sgeom.LineString(xyt.tolist())
        locs = axis.intersection(ls)
        if not locs:
            tick = [None]
        else:
            tick = tick_extractor(locs.xy)
        _ticks.append(tick[0])
    # Remove ticks that aren't visible:
    ticklabels = copy(ticks)
    while True:
        try:
            index = _ticks.index(None)
        except ValueError:
            break
        _ticks.pop(index)
        ticklabels.pop(index)
    return _ticks, ticklabels
#设置x轴标签（用来画纬度）
def lambert_xticks(ax, ticks):
    """Draw ticks on the bottom x-axis of a Lambert Conformal projection."""
    te = lambda xy: xy[0]
    lc = lambda t, n, b: np.vstack((np.zeros(n) + t, np.linspace(b[2], b[3], n))).T
    xticks, xticklabels = _lambert_ticks(ax, ticks, 'bottom', lc, te)
    ax.xaxis.tick_bottom()
    ax.set_xticks(xticks)
    ax.set_xticklabels([ax.xaxis.get_major_formatter()(xtick) for xtick in xticklabels])
#设置y轴标签（用来画经度）
def lambert_yticks(ax, ticks):
    """Draw ricks on the left y-axis of a Lamber Conformal projection."""
    te = lambda xy: xy[1]
    lc = lambda t, n, b: np.vstack((np.linspace(b[0], b[1], n), np.zeros(n) + t)).T
    yticks, yticklabels = _lambert_ticks(ax, ticks, 'left', lc, te)
    ax.yaxis.tick_left()
    ax.set_yticks(yticks)
    ax.set_yticklabels([ax.yaxis.get_major_formatter()(ytick) for ytick in yticklabels])

def timetrans(t):
    hour = t//60
    minu = t%60
    datetime = "%02d:%02d"%(hour,minu)
    return datetime

def nametotitle(name):
    #name = 20160112_SO2_0_truth_lay1
    #title = 2016/01/12 00:05 SO2第1层真实值
    pos1 = name.find('_')
    pos2 = pos1+name[pos1+1:].find('_')+1
    pos3 = pos2+name[pos2+1:].find('_')+1
    pos4 = name.rfind('_')
    pos5 = name.rfind('y')
    date = name[:4]+'/'+name[4:6]+'/'+name[6:8]
    sp = '$\mathregular{{%s}_{%s}}$'%(name[pos1+1:pos2-1],name[pos2-1:pos2])
    time = timetrans(int(name[pos2+1:pos3])*5)
    tag = name[pos3+1:pos4]
    if tag == "truth":
        tag = "真实值"
    elif tag == "predict":
        tag = "预测值"
    lay = name[pos5+1:]
    
    title = date+' '+time+' '+sp+"第%s层"%(lay)+tag
    return title

def showsingle(data,name,lay,minvalue,maxvalue,savepath):
    #name = 20160112_SO2_0_truth
    interval = (maxvalue-minvalue) / 10
    print(data.shape)
    data = data.reshape(512,512)
    fig = plt.figure()
    proj = ccrs.LambertConformal(central_longitude=120,
                             central_latitude=31,
                             false_easting=0,
                             false_northing=0, standard_parallels=(30, 60))

    ax = fig.subplots(1,1,subplot_kw={'projection': proj})
    norm = col.Normalize(vmin=minvalue, vmax=maxvalue)
    bounds = [x for x in np.arange(minvalue, maxvalue+0.001, interval)]
    
    plt.rc('font', family='Timesbd', size=14)
    font = font_manager.FontProperties(fname="/usr/share/fonts/cjkuni-uming/uming.ttc")
    
    title = nametotitle(name+'_lay'+str(lay+1))
    ax.set_title(title, fontdict={'weight': 'normal', 'size': 14},fontproperties=font)
    
    extent = [-4200000, 1944000, -2484000, 3660000]
    #extent=[50, 160, 3, 65]
    ax.set_extent(extent, crs=proj)
    
    ax.coastlines(resolution = '50m',linewidth=0.3) 
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),linewidth=0.3)
    


    owncolor = ['white', 'blue', 'lightgreen', 'yellow', 'orange', 'red', 'darkred']
    cmap2 = col.LinearSegmentedColormap.from_list('own', owncolor, gamma=0.6)
    cm.register_cmap(cmap=cmap2)
    levels = np.arange(0, maxvalue+0.1, interval)
    #p = ax.contourf(lon,lat,data,alpha = 0.9,cmap = 'own',levels=levels,zorder = 0,extent='both',transform=ccrs.PlateCarree())
    p = ax.imshow(data,  cmap='own', norm=norm, extent=extent)
    clb = plt.colorbar(p, ticks=bounds)
    clb.set_label('μg/$\mathregular{m^3}$')
    fig.canvas.draw()
    xticks = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
    yticks = [0, 10, 20, 30, 40, 50, 60, 70]
    ax.gridlines(xlocs=xticks, ylocs=yticks, linestyle='--', linewidth=0.3, color='k')
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
    lambert_xticks(ax, xticks)
    lambert_yticks(ax, yticks)
    savename = savepath + name+'_lay'+str(lay)+'.png'
    plt.savefig(savename, bbox_inches='tight', dpi=300)
    plt.close()

def showpicture(trupath,prepath,savepath):
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    tru = np.load(trupath)
    pre = np.load(prepath)
    print('tru:',tru.shape)
    print('pre:',pre.shape)
    tru = tru.reshape(16,512,512)
    pre = pre.reshape(16,512,512)
    for lay in range(tru.shape[0]):
        tru_l = tru[lay]*1000*2.2118*0.183
        pre_l = pre[lay]*1000*2.2118*0.183
        #minvalue = round(0.01*min(np.min(pre_l),np.min(tru_l)),3)
#        print('minvalue:',minvalue,'pmin:',np.min(pre_l),'tmin:',np.min(tru_l))
        minvalue = 0
        #maxvalue = max(minvalue+0.001,0.3*round(max(np.max(pre_l),np.max(tru_l)),3))
        maxvalue = 30
#        if maxvalue/100>=1:
#            maxvalue = round(maxvalue,2)*1000
#        elif maxvalue/0.01<1:
#            maxvalue = round(maxvalue,3)*1000
        #name = '20160112_SO2_0_truth'
        #'/lustre/home/acct-esehazenet/hazenet-pg5/data/testoutput/numpy_matrix/batch1000_0511_23:34:06.pth/20160115_SO2_0_predict.npy'

        truname = trupath[trupath.rfind('/')+1:-4]
        prename = prepath[prepath.rfind('/')+1:-4]
        print('truname:',truname,'prename:',prename)
        print('paintminvalue:',minvalue,'pmin:',np.min(pre_l),'tmin:',np.min(tru_l))
        print('paintmaxvalue:',maxvalue,'pmax:',np.max(pre_l),'tmax:',np.max(tru_l))
        showsingle(tru_l,truname,lay,minvalue,maxvalue,savepath)
        showsingle(pre_l,prename,lay,minvalue,maxvalue,savepath)
    return

if __name__ == '__main__':
    indexlist = [0,144,287]
    for i in indexlist:
        trupath = dirpath+'20160115_SO2_'+str(i)+'_truth.npy'
        prepath = dirpath+'20160115_SO2_'+str(i)+'_predict.npy'
        showpicture(trupath,prepath,outputpath)
        print('trupath:',trupath)
        print('prepath:',prepath)
