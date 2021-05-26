import pandas as pd
import numpy as np
from scatter_plot import *
from single_plot import *
ozonodes=pd.read_excel(r'../data/ozo_nodes.xlsx')
namelist=list(ozonodes.Name)
y11=np.load('../data/y1_L2APPEND1.npy').T[:45]
y12=np.load('../data/y2_L2APPEND1.npy').T[:45]
savedir='../output/'
#fig=scatter_plot(y11[0],y12[0],namelist,savedir+'a.jpg',1,True)
fig=single_plot(y11[10],y12[10],namelist[10],savedir+'a.jpg',False)
#fig=scatter_plot(y11,y12,namelist,savedir+'b.jpg',9)
#fig=scatter_plot(y11,y12,namelist,savedir+'c.jpg',15)
#fig=scatter_plot(y11[:12],y12[:12],namelist,savedir+'d.jpg',4)
#fig=scatter_plot(y11[:12],y12[:12],namelist,savedir+'e1.jpg',6,False,15)
#fig=scatter_plot(y11[:12],y12[:12],namelist,savedir+'e2.jpg',6,False,25)
#fig=scatter_plot(y11[:12],y12[:12],namelist,savedir+'e3.jpg',6,False,40)
#fig=scatter_plot(y11[:12],y12[:12],namelist,savedir+'f.jpg',8,False)
print('Successfully plotted!')