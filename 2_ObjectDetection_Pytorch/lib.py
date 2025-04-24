import os
import os.path as osp

import random
import xml.etree.ElementTree as ET
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Function
import torch.utils.data as data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import itertools
from math import sqrt

torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)