import os

import numpy as np
import torch
from scipy import io as sio
from torch.utils.data import Dataset



class MyTestDataSet(Dataset):
    def __init__(self, path, transform=None,max_num=None,key='img',rotate_90=False):
        super(MyTestDataSet, self).__init__()

        self.transform = transform
        imgs = []
        scene_list = os.listdir(path)
        if max_num != None:
            scene_list = scene_list[:max_num]
        scene_list.sort()
        self.rotate_90 = rotate_90
        self.key = key
        print("scene_list len:",len(scene_list))
        # test_data = np.zeros((len(scene_list), 256, 256, 28))
        for i in range(len(scene_list)):
            scene_path = path + scene_list[i]
            print("load:",scene_path)
            img = sio.loadmat(scene_path)[self.key]
            img = img.astype(np.float32)

            if self.rotate_90:
                img = np.rot90(img, k=-1)
                img = img.copy()
                # print("train rotate")
            img = img.transpose(2, 0, 1)
            imgs.append(img)

        self.imgs = imgs

    def __getitem__(self, item):
        x = self.imgs[item]
        x = torch.FloatTensor(x)
        if self.transform is not None:
            x = self.transform(x)
        return x,x

    def __len__(self):
        return len(self.imgs)
