import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import signal

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

import timm
from torchvision import datasets
import glob
import util.misc as misc
import models_mae_shared
from engine_test_time import train_on_test, get_prameters_from_args
from data import tt_image_folder
from util.misc import NativeScalerWithGradNormCount as NativeScaler