#%%
import numpy as np
import matplotlib.pyplot as plt
def plot_distribution(data_array: np.array, label=None, save_folder=None):
    '''
    输入二维的array，输出相应的面积图片
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
    print(label)
    if not save_folder:
        plt.show()
    else:
        path = "".join([save_folder, "/", "plot-{}.jpg".format(label)])
        plt.savefig(path)
# %%
path = "data/O3_hourly/20190708-0.npy"
arr = np.load(path, allow_pickle=True)
plot_distribution(arr, save_folder="saved/", label="hourly")
# %%
path = "data/O3_hourly_sampleByStation/20190708-0.npy"
arr = np.load(path, allow_pickle=True)
plot_distribution(arr, save_folder="saved/", label="hourly-sample")
# %%
