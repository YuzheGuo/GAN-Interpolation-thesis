#%%
from pykrige.ok import OrdinaryKriging
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# %%
def plot_distribution(data_array: np.array):
    """
    输入二维的array，输出相应的面积图片
    这里人为定义了vmax的数值，之后可能需要改改
    """
    shape = data_array.shape
    x = np.arange(0, shape[1])  # len = 11
    y = np.arange(0, shape[0])  # len = 7

    fig, ax = plt.subplots(dpi=150)
    pcm = ax.pcolormesh(x, y, data_array, vmax=data_array.max(), vmin=0, cmap="Blues")
    fig.colorbar(pcm, ax=ax)


def array_to_dataframe(arr: np.array)-> pd.DataFrame:
    '''
    array-> df: x, y, val
    '''
    raw_data = []
    x_lim = arr.shape[0]
    y_lim = arr.shape[1]
    for x in range(x_lim):
        for y in range(y_lim):
            val = arr[x][y]
            if val>0:
                raw_data.append([x, y, val])
    return pd.DataFrame(raw_data, columns=['x', 'y', 'val'])
#%%

path = '../data/O3_hourly_32_sh/'
arr = np.load(path+os.listdir(path)[10], allow_pickle=True)
plot_distribution(arr)
#%%
df = array_to_dataframe(arr)
# %%
OK = OrdinaryKriging(df.x, df.y, df.val)
xGrid = np.linspace(0, 32, 32)
yGrid = np.linspace(0, 32, 32)
z, ss = OK.execute('grid', xGrid, yGrid)
# %%
