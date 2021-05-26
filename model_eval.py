#%%
import math
import torch
import numpy as np
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import os
import matplotlib.pyplot as plt
import datetime
from torch import nn
import random
from sklearn.metrics import mean_squared_error as mse

import normalization as n
from IDW_OK.IDW_func import IDW_interpolation
from IDW_OK.OK_func import OK_interpolation
from plot_func.image_plot import plot_distribution

def cal_print_error_list(error_list: list):
    """
    input list, each element contain dict
    cal mean of each key, and print
    """
    res = dict()
    for key in error_list[0].keys():
        val = np.mean([i[key] for i in error_list])
        res[key] = val
        # print(key, ":", round(val,4))
    return res

def build_test_array_dataset(data_type="O3"):
    """
    test dataset, in form of list, each element [sample, real]
    """
    test_real_base = "data/data-final-copy/{}_hourly_32_sh_test/".format(data_type)
    test_sample_base = "data/data-final-copy/{}_hourly_32_sh_sample_by_station_test/".format(data_type)

    test_dataset = []
    for real_img, sample_img in zip(os.listdir(test_real_base), os.listdir(test_sample_base)):
        real_arr = np.load(test_real_base+real_img, allow_pickle=True)
        sample_arr = np.load(test_sample_base+sample_img, allow_pickle=True)
        test_dataset.append([sample_arr, real_arr])

    return test_dataset

def cal_abse(arr1, arr2):
    return np.abs(arr1 - arr2).mean()
def cal_mse(arr1, arr2):
    return np.power(np.power(arr1 - arr2, 2).mean(), 0.5)

def gan_model_predict(gan_model, data, data_type):
    """
    use normalization func!!!
    input: gan model, data: 2-D or 4-D np arr, data_type: O3 or PM25
    output: np array, same as input dimention
    """
    if not (len(data.shape)==2 or len(data.shape)==4):
        print("data dimention error...")
        raise

    if data_type=="PM25":
        data_n = (data - n.PM25_min)/(n.PM25_max-n.PM25_min)
    elif data_type=="O3":
        data_n = (data - n.O3_min)/(n.O3_max-n.O3_min)
    else:
        print("data type error!!!")
        raise
    
    if len(data_n.shape)==2:
        data_n = np.array([[data_n]])

    res = gan_model(torch.from_numpy(data_n).float()).detach().numpy()

    if len(data.shape) == 2:
        res = res[0][0]

    if data_type=="PM25":
        res = res * (n.PM25_max-n.PM25_min) + n.PM25_min
    else:
        res = res * (n.O3_max-n.O3_min) + n.O3_min
    
    return res

def get_error_list(arr_res, arr_real, progress_step=None):
    """
    input: two np array, 4-D
    output: the list of the error dict
    """
    error = arr_res - arr_real
    error_list_G = []
    count = 0
    for e, sub_arr_real  in zip(error, arr_real):
        mean = sub_arr_real.mean()
        error_arr = e[0]
        e_res = dict()

        e_res['abse_g'] = np.abs(error_arr).mean()
        e_res['r_abse_g'] = e_res['abse_g']/mean
        e_res['mse_g'] = np.power(np.power(error_arr, 2).mean(), 0.5)
        e_res['r_mse_g'] = e_res['mse_g']/mean
        error_list_G.append(e_res)

        if progress_step and count%progress_step == 0:
            print(e_res)

        count += 1
    return error_list_G

def get_station_location_list_from_array(arr: np.array):
    """
    arr: 2-D, return the list of index tuple
    """
    length = len(arr)
    index_list = []
    for i in range(length):
        for j in range(length):
            if arr[i][j] > 0:
                index_list.append((i, j))
    # print("generate finished, there're {} points that have values...".format(len(index_list)))
    return index_list


def build_omitted_dataset(dataset: np.array, rate: float):
    """
    dataset: 4-D, rate: 0.9, 0.8 ...
    return the new sampled dataset
    """
    arr = dataset.copy()
    val_list = get_station_location_list_from_array(arr[0][0])
    num_points_deleted = int((1-rate)*len(val_list))
    print("after sampling, deleted {} points".format(num_points_deleted))

    for data in arr:
        data = data[0]
        sampled_val_index = random.sample(range(len(val_list)), num_points_deleted)
        sampled_val_list = [val_list[i] for i in sampled_val_index]
        for i, j in sampled_val_list:
            data[i][j] = 0
    val_list = []
    
    return arr

def load_gan_model(path):
    """
    return G model for prediction
    """
    class Generator(nn.Module):
        def __init__(self, nc, ngf):
            super(Generator, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(nc, ngf, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ngf), nn.LeakyReLU(0.2, inplace=True))
            # 16 x 16 x 64
            self.layer2 = nn.Sequential(
                nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ngf * 2), nn.LeakyReLU(0.2, inplace=True))
            # 8 x 8 x 128

            self.layer3 = nn.Sequential(
                nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(ngf * 4), nn.LeakyReLU(0.2, inplace=True))
            # 4 x 4 x 256
            # 4 x 4 x 256
            self.layer4 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 4,
                                ngf * 2,
                                kernel_size=4,
                                stride=2,
                                padding=1), nn.BatchNorm2d(ngf * 2), nn.ReLU())
            # 8 x 8 x 128
            self.layer5 = nn.Sequential(
                nn.ConvTranspose2d(ngf * 2,
                                ngf,
                                kernel_size=4,
                                stride=2,
                                padding=1), nn.BatchNorm2d(ngf), nn.ReLU())
            # 16 x 16 x 64
            self.layer6 = nn.Sequential(
                nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid())
            # 32 x 32 x 1
        def forward(self, _cpLayer):
            out = self.layer1(_cpLayer)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.layer5(out)
            out = self.layer6(out)
            return out

    gan_model = Generator(nc=1, ngf=64)
    gan_model.load_state_dict(torch.load(path))
    gan_model.eval()
    return gan_model

O3_test_dataset = build_test_array_dataset("O3")
PM25_test_dataset = build_test_array_dataset("PM25")

for data_type in ["PM25"]:
    print("data type is: ", data_type)
    if data_type == "O3":
        sample = np.array([np.array([i[0]]) for i in O3_test_dataset])
        real = np.array([np.array([i[1]]) for i in O3_test_dataset])
    elif data_type == "PM25":
        sample = np.array([np.array([i[0]]) for i in PM25_test_dataset])
        real = np.array([np.array([i[1]]) for i in PM25_test_dataset])
    else:
        print("data type error!!!")
        raise

    path_O3 = "saved/O3-G/netG-saved-O3-epoch959-20210523_174823.pt"
    path_PM25 = "saved/PM25-G/netG-saved-PM25-epoch999-20210523_174823.pt"
    G = load_gan_model(path_O3) if data_type=="O3" else load_gan_model(path_PM25)

    def count_value(arr):
        df = pd.DataFrame(arr)
        index = df>0
        return index.sum().sum()

    def get_data_by_index(arr: np.array, index_list: list):
        """arr: 2-D, index_list: [(1, 4), (4,6)]
        return list of data"""
        return [arr[i][j] for i, j in index_list]

    def sample_array(arr: np.array, rate=0.9):
        """arr: 2-D, return the sampled data, rate = 0.9"""
        data = arr.copy()
        length = len(data)
        for i in range(length):
            for j in range(length):
                if (data[i][j]> 0 and random.random()>rate):
                    data[i][j] = 0
        return data

    def get_province_index_list():
        """read the province index csv, return [(2, 5)]"""
        pr_index = list(np.load("sample_func/IndexSet-province.npy", allow_pickle=True))
        pr_index = [(i[0]-65, i[1]-196) for i in pr_index]
        return pr_index

    pr_index = get_province_index_list()
    real_array, gan_array, idw_array, ok_array = [], [], [], []

    for i, dataset in enumerate(zip(sample[:], real[:])):
        print(i, "/", len(sample))
        sample_data, real_data = dataset[0][0], dataset[1][0]
        
        data_pred = sample_data.copy()
        # data_pred = sample_array(data_pred, 0.97)
        index_all = get_station_location_list_from_array(sample_data)
        index_test = list(set(pr_index) - set(index_all))

        res = gan_model_predict(G, data_pred, data_type)
        res_idw = IDW_interpolation(data_pred)
        res_ok = OK_interpolation(data_pred)

        real_array += get_data_by_index(real_data, index_test)
        gan_array += get_data_by_index(res, index_test)
        idw_array += get_data_by_index(res_idw, index_test)
        ok_array += get_data_by_index(res_ok, index_test)

    # plt.scatter(real_array, idw_array, color="blue")
    # plt.scatter(real_array, ok_array, color="green")
    # plt.scatter(real_array, gan_array, color="red")
    saved_array = np.array([real_array, gan_array, idw_array, ok_array])
    np.save("saved/final-plot/{}-province-point-compare.npy".format(data_type), saved_array)
    print("res saved finished...")
#%%
res_dict_list = []
for name in os.listdir(path):
    net = load_gan_model(path+name)
    res = gan_model_predict(net, sample, "O3")
    error_list_G = get_error_list(res, real)
    epoch = int(name[name.find("epoch")+5:][:name[name.find("epoch")+5:].find("-")])
    print(epoch)
    res_dict_list.append([epoch, cal_print_error_list(error_list_G)])

#%%

def plot_trend(x, y1, y2, xlabel, ylabel, size=(16, 10)):
    plt.rcParams['font.family'] = 'Arial'
    fontsize = 20
    fig, ax = plt.subplots(figsize=(size[0], size[1]))
    ax.tick_params(which='minor', length=3, width=1.5)
    ax.tick_params(labelsize=fontsize, length=6, width=1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.plot(x, y1, linewidth=2.3, label="Generator")
    ax.plot(x, y2, linewidth=2.3, label="Discriminator")
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.legend(fontsize=fontsize, borderpad=0, frameon=0, ncol=size[0])

res_dict_list.sort(key=lambda x: x[0])
y1 = [i[1]['r_abse_g'] for i in res_dict_list]
x = [i[0] for i in res_dict_list]
y2 = [i[1]['r_mse_g'] for i in res_dict_list]


# np.save("saved/final-plot/O3-test-error-per-epoch.npy", 
#             np.array(res_dict_list))
#%%
path = "saved/O3-loss-32size-20210523_174823.npy"
arr = np.load(path, allow_pickle=True)
cut = [0, 620]
x = np.array(range(62000))[cut[0]: cut[1]]/62
y1 = arr[0][cut[0]: cut[1]]
y2 = arr[1][cut[0]: cut[1]]
# y1 = np.array(pd.DataFrame(y1).rolling(62).mean())
# y2 = np.array(pd.DataFrame(y2).rolling(62).mean())
plot_trend(x, y1, y2, "epoch", "loss")

#%%
path = 


imag = sample[10][0]
arr = OK_interpolation(imag)
plot_distribution(imag)
plot_distribution(OK_interpolation(imag))
plot_distribution(IDW_interpolation(imag))
gan_img = gan_model_predict(netG, imag, "PM25")
plot_distribution(gan_img)
plot_distribution(real[10][0])

#%%
sample = np.array([np.array([i[0]]) for i in PM25_test_dataset])
real = np.array([np.array([i[1]]) for i in PM25_test_dataset])
res = gan_model_predict(netG, sample, "PM25")
error_list_G = get_error_list(res, real, 1000)
cal_print_error_list(error_list_G)
#%%
def get_data_by_index(arr: np.array, index_list: list):
    """arr: 2-D, index_list: [(1, 4), (4,6)]
    return list of data"""
    return [arr[i][j] for i, j in index_list]
index_list = get_station_location_list_from_array(sample[0][0])
lis1 = get_data_by_index(res[100][0], index_list)
lis2 = get_data_by_index(real[100][0], index_list)
print(mse(lis1, lis2))
print(np.mean(lis1))
plt.scatter(lis1, lis2)
#%%
omitted = build_omitted_dataset(sample, 0.9)
res = gan_model_predict(netG, omitted, data_type="PM25")
error_list = get_error_list(res, real, 200)
cal_print_error_list(error_list)

#%%
"""cal the error of IDW, OK, finished"""
error_list = []
step = 20
for i, data in enumerate(O3_test_dataset):
    sample, real = data[0], data[1]
    mean = real.mean()
    e_res = dict()

    OK_arr = OK_interpolation(sample)
    IDW_arr = IDW_interpolation(sample)

    e_OK = cal_abse(real, OK_arr)
    e_IDW = cal_abse(real, IDW_arr)
    mse_OK = cal_mse(real, OK_arr)
    mse_IDW = cal_mse(real, IDW_arr)

    e_res['abse_OK'] = e_OK
    e_res['r_abse_OK'] = e_OK/mean
    e_res['abse_IDW'] = e_IDW
    e_res['r_abse_IDW'] = e_IDW/mean

    e_res['mse_OK'] = mse_OK
    e_res['r_mse_OK'] = mse_OK/mean
    e_res['mse_IDW'] = mse_IDW
    e_res['r_mse_IDW'] = mse_IDW/mean

    error_list.append(e_res)

    if (i%step==0):
        print("mean: ", mean)
        print(i, "/", len(O3_test_dataset))
        print(error_list[-1])
        # plot_distribution(OK_arr)
        # plot_distribution(IDW_arr)
        # plot_distribution(data[1])
# np.save("saved/IDW_OK_test/O3_error_IDW_OK.npy", np.array(error_list))

#%%
# import time
# start = time.time()
# for i, data in enumerate(O3_test_dataset):
#     sample, real = data[0], data[1]
#     OK_arr = OK_interpolation(sample)
#     # IDW_arr = IDW_interpolation(sample)
# end = time.time()
# print('Running time: {} Seconds'.format(end-start))

# #运行结果如下
# # %%
# import time
# start = time.time()
# for i, data in enumerate(O3_test_dataset[:10]):
#     # print(i)
#     sample, real = data[0], data[1]
#     # OK_arr = OK_interpolation(sample)
#     IDW_arr = IDW_interpolation(sample)
# end = time.time()
# print('Running time: {} Seconds'.format(end-start))
