# %%
'''
读取之前保存好的station的IndexSet
批量加工处理，得到采样完好的数据
'''
import numpy as np 
from get_sample_function import SampleByStation
import matplotlib.pyplot as plt
# %%
def read_IndexSet(filePath):
    IndexSetList = np.load(filePath, allow_pickle=True)
    IndexSet = set()
    for loc in IndexSetList:
        IndexSet.add((loc[0], loc[1]))
    return IndexSet

filePath = 'IndexSet.npy'
IndexSet = read_IndexSet(filePath)
print(len(IndexSet))
# %%
# print(IndexSet)
# %%

def array_yielder(base):
    fileName = os.listdir(base)
    for name in fileName:
        yield name, np.load(base+name)

base = 'O3_numpy_split/'
for name, array in array_yielder(base):
    sampled_array = SampleByStation(data_arr=array, IndexSet=IndexSet)
    np.save('O3_sample_by_station_numpy_split/{}'.format(name), sampled_array)
    print(name)
# %%


