import os
from io import BytesIO
from pathlib import Path

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUNClass
import torch
import pandas as pd

import torchvision.transforms.functional as Ftrans

env = lmdb.open(
            path = 'datasets/cxr.lmdb',
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )


with env.begin(write=False) as txn:
            key = f'{128}-{str(1).zfill(6)}'.encode(
                'utf-8')
            print(key)
            length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))
            print(length)
            img_bytes = txn.get(key)
            print(img_bytes)

buffer = BytesIO(img_bytes)
print(buffer)
img = Image.open(buffer)