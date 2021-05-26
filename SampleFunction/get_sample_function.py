# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import netCDF4 as nc
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import pickle

def get_location_array(basePath):
    # print(os.listdir())
    nameList = os.listdir(basePath)
    path = basePath + nameList[0]
    file_obj = nc.Dataset(path)
    lon = file_obj.variables['longitude'][:]
    lat = file_obj.variables['latitude'][:]

    res_arr = np.zeros((216, 270), dtype='object')
    for i in range(len(lon)):
        for j in range(len(lon[0])):
            res_arr[i][j] = (lon[i][j], lat[i][j])
    return res_arr

def cal_distance_helper(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    输出：距离km
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # 地球平均半径，单位为公里
    return c * r 

def cal_distance(location1:tuple, location2:tuple)-> float:
    return cal_distance_helper(location1[0], location1[1], location2[0], location2[1])

def get_nearist_index(location_array: np.array, location: tuple)-> tuple:
    '''
    input: the array stored all the location; the location
    output: the location index (i, j), not the (lon,lat)
    '''
    minDis, index = None, (0, 0)
    for i in range(len(location_array)):
        for j in range(len(location_array[0])):
            dis = cal_distance(location, location_array[i][j])
            if not minDis:
                minDis = dis
            if dis < minDis:
                minDis = dis
                index = (i, j)
    return index

# %%

def get_national_station_location_set(path):
    df = pd.read_csv(path)
    LocationSet = set()
    for loc in np.array(df[['lng', 'lat']]):
        loc_tuple = (loc[0], loc[1])
        LocationSet.add(loc_tuple)
    return LocationSet


def get_index_set(locationArray, locationSet):
    IndexSet = set()
    count = 0
    for i in locationSet:
        index = get_nearist_index(locationArray, i)
        IndexSet.add(index)
        count += 1
        print(count)
    return IndexSet

# the most useful function!
def SampleByStation(data_arr: np.array, IndexSet: set)-> np.array:
    arr_res = np.zeros(data_arr.shape)
    for i in range(len(data_arr)):
        for j in range(len(data_arr[0])):
            if (i, j) in IndexSet:
                arr_res[i][j] = data_arr[i][j]
    return arr_res

# %%
if __name__=="__main__":

    '''get the Set of index'''
    basePath = 'D:/thesis_helper/'
    res_arr = get_location_array(basePath)
    print(res_arr)
    path = 'province_station_loc.csv'
    NationalStationLocationSet = get_national_station_location_set(path)
    IndexSet = get_index_set(res_arr, NationalStationLocationSet)
    print(len(IndexSet))
    np.save('IndexSet-province.npy', np.array(list(IndexSet)))
    # print(IndexSet.pop())

    '''read Set of index'''

# %%

# Set = np.load('IndexSet.npy', allow_pickle=True)
# print(Set)
# for i in Set:
#     print(i)

# %%
