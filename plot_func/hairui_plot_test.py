
#%%
from scatter_plot import *
import numpy as np
# %%
x = np.array([list(range(1000))])
y = np.array([list(range(1000))])

arr = np.load("O3-original-point-compare.npy", allow_pickle=True)
x, y = np.array([arr[0]]), np.array([arr[1]])
scatter_plot(x, y, ["O3", "PM2.5"], "", maxcol=2)
# %%
