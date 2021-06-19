
# %%
from sklearn.metrics import mean_squared_error, explained_variance_score, r2_score, mean_absolute_error
from scipy.stats import gaussian_kde
import matplotlib.colors as col
import shapely.geometry as sgeom
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)


def plot_distribution(data_array: np.array, label=None, image_res_save_folder=None):
    '''
    输入二维的array，输出相应的面积图片
    label: the label if this photo
    image_res_save_folder: the folder to save
    cal the max and min for the color bar
    '''
    plt.rcParams['font.family'] = 'Arial'
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
        path = "".join([image_res_save_folder, "/",
                       "plot-{}.jpg".format(label)])
        plt.savefig(path)

# %%
with open('plot_func/CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ')
               for block in blocks]
               
def plot_distribution_with_borders(arr: np.array, vmin=None, 
                                vmax=None, save_path=None,
                                fontsize=40,
                                colorbar_label="(μg/m3) or (ppb) or (ppm)"):
    """
    add province borders, return the photos!!!
    """
    
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=[18, 12])

    limit = [116.42491455078125, 122.91134948730469,
             28.028094482421875, 33.68136672973633]
    center_lat, center_lon = 0.5 * \
        (limit[2] + limit[3]), 0.5*(limit[0] + limit[1])

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(limit, crs=ccrs.PlateCarree())

    ax.gridlines(linestyle='--')

    # Plot border lines
    for line in borders:
        ax.plot(line[0::2], line[1::2], '--', lw=0.6, color='black',
                transform=ccrs.Geodetic())
    if not vmin and not vmax:
        norm = col.Normalize(vmin=0.0, vmax=np.max(arr))
    else:
        norm = col.Normalize(vmin=vmin, vmax=vmax)
    p = ax.imshow(arr, norm=norm, cmap="jet", extent=limit, origin="lower")
    cbar = fig.colorbar(p, ax=ax, shrink=1)
    cbar.set_label(colorbar_label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    fig.canvas.draw()

    xticks = np.arange(int(limit[0])+1, int(limit[1])+0.01, 1)
    yticks = np.arange(int(limit[2])+1, int(limit[3])+0.01, 1)
    ax.set_xlabel("longitude", fontsize=fontsize)
    ax.set_ylabel("latitude", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    if save_path:
        fig.savefig(save_path, dpi=300)
    return ax


def plot_estimation(Y_pred, Y_test, z, save_path=None):

    r2 = round(r2_score(Y_test, Y_pred), 2)
    # RMSE=round(mean_squared_error(Y_test,Y_pred)**0.5,2)
    MAE = round(mean_absolute_error(Y_test, Y_pred), 2)
    RMAE = round(MAE/np.mean(Y_test), 2)

    fontsize = 14
    plt.rcParams['font.family'] = 'Arial'
    plt.figure(figsize=(6, 5), dpi=300)
    lim = max(max(Y_pred), max(Y_test))

    plt.scatter(Y_test, Y_pred, linewidth=1, c=z, s=5,
                ls='-', cmap='jet', vmin=1, vmax=60)

    plt.plot(np.arange(0, lim), np.arange(
        0, lim), color='black', linestyle=':')

    a, b = np.polyfit(Y_test, Y_pred, 1)[
        0], np.polyfit(Y_test, Y_pred, 1)[1]
    x = np.arange(0, lim, 1)
    y = a*x+b
    cor = Y_test.corr(Y_pred)
    print('a:', a, 'b:', b)

    plt.plot(x, y, color='red', linestyle='-', alpha=0.7)

    step = (int(lim/60)+1) * 10

    plt.xlabel('CAMx simulated PM2.5(mg/L)', fontsize=fontsize)
    plt.ylabel('interpolated PM2.5(mg/L)', fontsize=fontsize)
    plt.xlim(-1, lim)
    plt.ylim(-1, lim)
    plt.tick_params(labelsize=fontsize)
    plt.xticks(ticks=np.arange(0, lim, step))
    plt.yticks(ticks=np.arange(0, lim, step))
    plt.colorbar()

    step = lim/12
    i, j = 15, lim - 20
    plt.text(i, j, 'R2: {}'.format(r2), fontsize=fontsize)
    plt.text(i, j-step, 'MAE: {}'.format(MAE), fontsize=fontsize)
    plt.text(i, j-2*step, 'RMAE: {}'.format(RMAE), fontsize=fontsize)
    plt.text(i, j-3*step, 'N: {}'.format(len(x_pred)), fontsize=fontsize)

    if save_path:
        plt.savefig(save_path, dpi=300)


def cal_stat_dict(X_test, X_pred):
    """cal the r2, MAE, RMAE"""
    res = dict()
    res["r2"] = r2_score(X_test, X_pred)
    res['MAE'] = mean_absolute_error(X_test, X_pred)
    res['RMAE'] = res['MAE']/np.mean(X_test)
    print(res)
    return res


def cal_z_val_scatter_plot(x, y):
    xy = np.vstack([x_pred, x_test])
    z = gaussian_kde(xy)(xy)
    z = 300 * (z - np.min(z))/np.max(z)-np.min(z)
    return z


# %%
if __name__ == "__main__":

    path1 = "../saved/final-plot/O3-original-point-compare.npy"
    path2 = "../saved/final-plot/PM25-original-point-compare.npy"
    path3 = "../saved/final-plot/PM25-province-point-compare.npy"
    path4 = "../saved/final-plot/O3-province-point-compare.npy"

    arr = np.load(path2, allow_pickle=True)
    arr = arr

    for i in [3]:
        x_test = arr[0]
        x_pred = arr[1]
        save_path = "../saved/final-plot/PM25-{}-original-point-compare.png".format(
            i)
        res = cal_stat_dict(x_test, x_pred)
        z = cal_z_val_scatter_plot(x_test, x_pred)
        print("cal z finished")
        plot_estimation(pd.Series(x_pred), pd.Series(x_test), z, save_path)
    # %%

    with open('CN-border-La.dat') as src:
        context = src.read()
        blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
        borders = [np.fromstring(block, dtype=float, sep=' ')
                   for block in blocks]

    path = "D:\大学\大四上\毕业设计-GAN插值-空气质量模型\WorkSpace-thesis\Cmax_test\sample_func\station_location.csv"
    df_station = pd.read_csv(path)
    path_pr = "province_station_loc.csv"
    df_pr_station = pd.read_csv(path_pr)
    # %%

    def build_test_array_dataset(data_type="O3"):
        """
        test dataset, in form of list, each element [sample, real]
        """
        test_real_base = "D:/大学/大四上/毕业设计-GAN插值-空气质量模型/WorkSpace-thesis/Cmax_test/data/data-final-copy/{}_hourly_32_sh_test/".format(
            data_type)
        test_sample_base = "D:/大学/大四上/毕业设计-GAN插值-空气质量模型/WorkSpace-thesis/Cmax_test/data/data-final-copy/{}_hourly_32_sh_sample_by_station_test/".format(
            data_type)

        test_dataset = []
        for real_img, sample_img in zip(os.listdir(test_real_base), os.listdir(test_sample_base)):
            real_arr = np.load(test_real_base+real_img, allow_pickle=True)
            sample_arr = np.load(
                test_sample_base+sample_img, allow_pickle=True)
            test_dataset.append([sample_arr, real_arr])

        return test_dataset

    dataset = build_test_array_dataset("PM25")
    sample = np.array([np.array([i[0]]) for i in dataset])
    real = np.array([np.array([i[1]]) for i in dataset])
    data = real[0][0]
    # %%
    limit = [116.42491455078125, 122.91134948730469,
             28.028094482421875, 33.68136672973633]
    df = df_station
    df_selected = df[df.lng < 122.91134948730469][df.lng >
                                                  116.42491455078125][df.lat > 28.028][df.lat < 33.681]
    city_list = list(df_selected.groupby(
        by="city").count().sample(frac=0.9).index)
    df_sample = df_selected[df_selected.city.isin(city_list)]
    df_test = df.iloc[list(set(df_selected.index) - set(df_sample.index)), :]
    # %%
    limit = [116.42491455078125, 122.91134948730469,
             28.028094482421875, 33.68136672973633]
    center_lat, center_lon = 0.5 * \
        (limit[2] + limit[3]), 0.5*(limit[0] + limit[1])
    plt.rcParams['font.family'] = 'Arial'
    fontsize = 20
    fig = plt.figure(figsize=[12, 12])
    # Set projection and plot the main figure
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(limit, crs=ccrs.PlateCarree())
    # Add ocean, land, rivers and lakes
    # ax.add_feature(cfeature.OCEAN.with_scale('110m'))
    # ax.add_feature(cfeature.LAND.with_scale('110m'))
    ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    ax.add_feature(cfeature.LAKES.with_scale('50m'))
    # Plot gridlines
    ax.gridlines(linestyle='--')
    # Set figure extent
    # Plot border lines
    for line in borders:
        ax.plot(line[0::2], line[1::2], '--', lw=0.6, color='black',
                transform=ccrs.PlateCarree())

    # ax.plot(df_sample.lng, df_sample.lat, 'o', color='blue',
    #         label='sampled stations', markersize=2, transform=ccrs.Geodetic())

    ax.plot(df_station.lng, df_station.lat, 'o', color='blue',
            label='national stations', markersize=2.5, transform=ccrs.Geodetic())
    ax.plot(df_pr_station.lng, df_pr_station.lat, 'o', color='red',
            label='province stations', markersize=2.5, transform=ccrs.Geodetic())

    ax.legend(fontsize=fontsize, loc="upper right")
    norm = col.Normalize(vmin=0.0, vmax=50)
    # ax.imshow(data, norm=norm, cmap="Blues", extent=limit, origin="lower")
    # fig.canvas.draw()
    xticks = np.arange(int(limit[0])+1, int(limit[1])+1.01, 1)
    yticks = np.arange(int(limit[2])+1, int(limit[3])+1.01, 1)
    ax.set_xlabel("longitude", fontsize=fontsize)
    ax.set_ylabel("latitude", fontsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    fig.savefig("../saved/final-plot/national_province_stations.png", dpi=300)
    # %%

    # %%

    # Plot South China Sea as a subfigure
    sub_ax = fig.add_axes([0.741, 0.11, 0.14, 0.155],
                          projection=ccrs.LambertConformal(central_latitude=30,
                                                           central_longitude=119))
    # Add ocean, land, rivers and lakes
    sub_ax.add_feature(cfeature.OCEAN.with_scale('110m'))
    sub_ax.add_feature(cfeature.LAND.with_scale('110m'))
    # sub_ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    # sub_ax.add_feature(cfeature.LAKES.with_scale('50m'))
    # Plot border lines
    for line in borders:
        sub_ax.plot(line[0::2], line[1::2], '-', lw=1, color='k',
                    transform=ccrs.Geodetic())
    # Set figure extent
    sub_ax.set_extent([105, 125, 0, 25])
    # Show figure
    plt.show()
    # %%
