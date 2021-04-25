#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
arr = np.load("O3-loss-64batchsize-0001rate-normalize_uniform.npy", allow_pickle=True)
# %%
arr.shape
# %%
x = range(0, 720*64, 64)
plt.plot(x, arr[0])
plt.plot(x, arr[1])
plt.xlabel("batch")
plt.ylabel("loss")
plt.figure(figsize=(10, 6), dpi=200)
# plt.plot(arr[0])
# plt.xlabel("")
# %%
