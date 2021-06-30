### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
import random
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_path, train_file, test_file, train_size, isTrain = True, if_gt=False, num_class=8):
        self.gt_path = "/mnt/blob/datasets/FFHQ/image256_/"
        add_gt_image = if_gt
        self.isTrain = isTrain
        if isTrain == True:
            image_file =  open(train_file, "r") 
            self.image_list = image_file.readlines()
            # add_gt_image = True
            if add_gt_image == True:
                image_file = open("/mnt/blob/datasets/generate_results/gt_train.txt")
                gt_list = image_file.readlines()
                self.image_list = self.image_list + gt_list
                
            self.transform = transforms.Compose([
                transforms.RandomCrop((train_size, train_size)),
                transforms.RandomRotation(15.),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        else:
            image_file =  open(test_file, "r") 
            self.image_list = image_file.readlines()
            add_gt_image = False
            if add_gt_image == True:
                image_file = open("/mnt/blob/datasets/generate_results/gt_test.txt")
                gt_list = image_file.readlines()
                self.image_list = self.image_list + gt_list

            # i will update it later
            self.transform = transforms.Compose([
                transforms.CenterCrop((train_size,train_size)),
                # transforms.RandomCrop((train_size, train_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

        print("root path is --------------- ")
        print(root_path)
        self.root_path = root_path
        self.num_class = num_class
        
        self.max_number = 770000
        self.min_number = 107000
        if self.num_class == 2:
            self.judge_list = [100000]
        if self.num_class == 8:
            self.judge_list = [190000, 280000, 370000, 460000, 550000, 640000, 730000]
        if self.num_class == 12:
            self.judge_list = [150000, 210000, 270000, 330000, 390000, 450000, 510000, 570000, 630000, 680000, 730000]
        if self.num_class == 5:
            self.judge_list = [210000, 350000, 490000, 630000]
        if self.num_class == 7:
            self.judge_list = [240000, 350000, 450000, 540000, 620000, 690000]
        if self.num_class == 9:
            self.judge_list = [180000, 260000, 340000, 420000, 500000, 580000, 660000, 740000]
        if self.num_class == 6:
            self.judge_list = [200000, 300000, 410000, 530000, 660000]

    # def label_transform(self, label):
    #     label = float(label)/1000000.0-0.1
    #     # label = label*(1/0.4928-0.1)
    #     label = label*1.5
    #     return label

    def label_transform(self, label):
        if label == 0:
            out_label = torch.ones(self.num_class)
        else:
            out_label = torch.zeros(self.num_class)
            for judge_index in range(self.num_class-1):
                if label > self.judge_list[judge_index]:
                    out_label[judge_index] = 1
        return out_label


    def __getitem__(self, index):
        # print("do some  debug in temporal datasets ------------")
        image_name = self.image_list[index].replace("\n","")
        if "_" in image_name:
            image_path = os.path.join(self.root_path, image_name)
        else:
            # image_path = os.path.join(self.root_path, image_name)   # for cat test
            image_path = os.path.join(self.gt_path, image_name)   # for ffhq test
        image = Image.open(image_path)
        image_tensor = self.transform(image)
        label = image_name.split("_")[0]
        if self.isTrain == True:
            if ".png" in label:
                label = self.label_transform(0)
            else:
                label = int(label)
                label = self.label_transform(label)
        else:
            label = 77        

        return_list = {"image": image_tensor, "label": label}

        return return_list

    def __len__(self):
        return len(self.image_list)

    def name(self):
        return 'CustomDataset'
