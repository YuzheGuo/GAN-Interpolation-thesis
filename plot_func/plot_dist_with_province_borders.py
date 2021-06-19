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

with open('CN-border-La.dat') as src:
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