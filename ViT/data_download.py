# !pip install the_well[benchmark]

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
# from neuralop.models import FNO
from tqdm import tqdm

from the_well.benchmark.metrics import VRMSE
from the_well.data import WellDataset
from the_well.utils.download import well_download

device = "cuda"
base_path = "/home/namancho/datasets"  # path/to/storage
well_download(base_path=base_path, dataset="turbulent_radiative_layer_2D", split="train")
well_download(base_path=base_path, dataset="turbulent_radiative_layer_2D", split="valid")