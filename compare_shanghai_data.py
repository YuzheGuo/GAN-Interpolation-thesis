#%%
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
from SampleFunction.get_sample_function import get_nearist_index, SampleByStation, get_location_array

# %%
loc_arr = get_location_array("Cmax_test\data\china_camx_original")
index = get_nearist_index(loc_arr, (2, 5))
# %%
