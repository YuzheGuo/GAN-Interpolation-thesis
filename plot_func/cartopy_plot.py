#%%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np
import shapely.geometry as sgeom
import matplotlib.colors as col
from image_plot import plot_distribution

with open('CN-border-La.dat') as src:
    context = src.read()
    blocks = [cnt for cnt in context.split('>') if len(cnt) > 0]
    borders = [np.fromstring(block, dtype=float, sep=' ') for block in blocks]
#%%
def plot_distribution_with_border():
    pass

def plot_distribution_with_borders(arr: np.array, vmin=None, vmax=None):
    """
    add borders, return the photos!!!
    """
    limit = [116.42491455078125, 122.91134948730469, 28.028094482421875, 33.68136672973633]
    center_lat, center_lon = 0.5*(limit[2] + limit[3]), 0.5*(limit[0] + limit[1])
    fig = plt.figure(figsize=[12, 10])
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(limit, crs=ccrs.PlateCarree())

    # ax.add_feature(cfeature.OCEAN.with_scale('110m'))
    # ax.add_feature(cfeature.LAND.with_scale('110m'))
    # ax.add_feature(cfeature.RIVERS.with_scale('50m'))
    # ax.add_feature(cfeature.LAKES.with_scale('50m'))
    
    ax.gridlines(linestyle='--')
    
    # Plot border lines
    for line in borders:
        ax.plot(line[0::2], line[1::2], '--', lw=0.6, color='black',
                transform=ccrs.Geodetic())
    if not vmin and not vmax:
        norm = col.Normalize(vmin=0.0, vmax=np.max(arr)*0.8)
    else:
        norm = col.Normalize(vmin=vmin, vmax=vmax)
    p = ax.imshow(arr, norm=norm, cmap="jet", extent=limit, origin="lower")
    cbar = fig.colorbar(p, ax=ax, shrink=0.75)
    # cbar.set_label("PM2.5 values", fontsize=15)
    fig.canvas.draw()

    xticks = np.arange(int(limit[0])+1, int(limit[1])+0.01, 1)
    yticks = np.arange(int(limit[2])+1, int(limit[3])+0.01, 1)
    ax.set_xlabel("longitude", fontsize=15)
    ax.set_ylabel("latitude", fontsize=15)
    # set_xticklabels(xticks, fontsize=20)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    return ax


path = "D:\大学\大四上\毕业设计-GAN插值-空气质量模型\WorkSpace-thesis\Cmax_test\SampleFunction\station_location.csv"
df_station = pd.read_csv(path)
path_pr = "province_station_loc.csv"
df_pr_station = pd.read_csv(path_pr)
#%%
def build_test_array_dataset(data_type="O3"):
    """
    test dataset, in form of list, each element [sample, real]
    """
    test_real_base = "D:/大学/大四上/毕业设计-GAN插值-空气质量模型/WorkSpace-thesis/Cmax_test/data/data-final-copy/{}_hourly_32_sh_test/".format(data_type)
    test_sample_base = "D:/大学/大四上/毕业设计-GAN插值-空气质量模型/WorkSpace-thesis/Cmax_test/data/data-final-copy/{}_hourly_32_sh_sample_by_station_test/".format(data_type)

    test_dataset = []
    for real_img, sample_img in zip(os.listdir(test_real_base), os.listdir(test_sample_base)):
        real_arr = np.load(test_real_base+real_img, allow_pickle=True)
        sample_arr = np.load(test_sample_base+sample_img, allow_pickle=True)
        test_dataset.append([sample_arr, real_arr])

    return test_dataset


if __name__ == "__main__":
    dataset = build_test_array_dataset("PM25")
    sample = np.array([np.array([i[0]]) for i in dataset])
    real = np.array([np.array([i[1]]) for i in dataset])
    data = real[0][0]
#%%
limit = [116.42491455078125, 122.91134948730469, 28.028094482421875, 33.68136672973633]
df = df_station
df_selected = df[df.lng<122.91134948730469][df.lng>116.42491455078125][df.lat>28.028][df.lat<33.681]
city_list = list(df_selected.groupby(by="city").count().sample(frac=0.9).index)
df_sample = df_selected[df_selected.city.isin(city_list)]
df_test = df.iloc[list(set(df_selected.index) - set(df_sample.index)), :] 
#%%
limit = [116.42491455078125, 122.91134948730469, 28.028094482421875, 33.68136672973633]
center_lat, center_lon = 0.5*(limit[2] + limit[3]), 0.5*(limit[0] + limit[1])
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

ax.plot(df_station.lng, df_station.lat, 'o', color='red',
        label='test stations', markersize=2, transform=ccrs.Geodetic())
ax.plot(df_pr_station.lng, df_pr_station.lat, 'o', color='green',
        label='province stations', markersize=2, transform=ccrs.Geodetic())



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

#%%

#%%


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
