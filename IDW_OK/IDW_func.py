#%%
import math
import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import matplotlib.pyplot as plt


# %%
def plot_distribution(data_array: np.array,
                      label=None,
                      image_res_save_folder=None):
    '''
    输入二维的array，输出相应的面积图片
    label: the label if this photo
    image_res_save_folder: the folder to save
    cal the max and min for the color bar
    '''
    shape = data_array.shape
    x = np.arange(0, shape[1])  # len = 11
    y = np.arange(0, shape[0])  # len = 7

    fig, ax = plt.subplots(dpi=100)
    pcm = ax.pcolormesh(x,
                        y,
                        data_array,
                        shading='auto',
                        vmax=data_array.max(),
                        vmin=data_array.min(),
                        cmap='Blues')
    fig.colorbar(pcm, ax=ax)
    if not image_res_save_folder:
        plt.show()
    else:
        path = "".join(
            [image_res_save_folder, "/", "plot-{}.jpg".format(label)])
        plt.savefig(path)


def array_to_dataframe(arr: np.array) -> pd.DataFrame:
    '''
    array-> df: x, y, val
    '''
    raw_data = []
    x_lim = arr.shape[0]
    y_lim = arr.shape[1]
    for x in range(x_lim):
        for y in range(y_lim):
            val = arr[x][y]
            if val > 0:
                raw_data.append([x, y, val])
    return pd.DataFrame(raw_data, columns=['x', 'y', 'val'])


# %%
def cal_distance(x, y, x1, y1):
    d = ((x1 - x)**2 + (y1 - y)**2)**0.5
    return d


# %%
def IDW(x, y, z, xi, yi):
    # have value dict
    index_dict = dict()
    for _x, _y, _z in zip(x, y, z):
        index_dict[(_x, _y)] = _z
    # print("length of x, y, z is {}, create index_dict length {}".format(
        # len(x), len(index_dict.keys())))
    lstxyzi = []
    for p in range(len(xi)):
        if (xi[p], yi[p]) in index_dict.keys():
            # print("do not need to cal, at {}".format((xi[p], yi[p])))
            lstxyzi.append([xi[p], yi[p], index_dict.get((xi[p], yi[p]))])
        else:
            lstdist = []
            for s in range(len(x)):
                d = cal_distance(x[s], y[s], xi[p], yi[p])
                lstdist.append(d)
            sumsup = list((1 / np.power(lstdist, 2)))
            suminf = np.sum(sumsup)
            sumsup = np.sum(np.array(sumsup) * np.array(z))
            u = sumsup / suminf
            xyzi = [xi[p], yi[p], u]
            lstxyzi.append(xyzi)
    # print("the length of the res is: ", len(lstxyzi))
    return (lstxyzi)

def gen_grid(size=32):
    """
    input size
    output: two list contains x_list and y_list, size 32*32
    """
    x_list = []
    y_list = []
    for x in range(32):
        for y in range(32):
            x_list.append(x)
            y_list.append(y)
    # print("generate x, y index list, length: {}".format(len(x_list)))
    return x_list, y_list


def IDW_interpolation(arr: np.array) -> np.array:
    """
    input the array, output the interpolated array
    using exact interpolation
    """
    df = array_to_dataframe(arr)
    df = df[df.val>0]
    xlist, ylist = gen_grid(size=32)
    res = IDW(df.x, df.y, df.val, xlist, ylist)
    # res -> 2-D array
    arr_gen = np.zeros((32, 32))
    for i in res:
        arr_gen[i[0]][i[1]] = i[2]
    return arr_gen


# %%
if __name__ == "__main__":
    path = '../data/O3_hourly_32_sh_sample_by_station/'
    arr = np.load(path + os.listdir(path)[110], allow_pickle=True)

    res = IDW_interpolation(arr)
    plot_distribution(res)
    plot_distribution(arr)
# %%
