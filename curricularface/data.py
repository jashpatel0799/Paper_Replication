import os
import torch
import torchvision


import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import os
from collections import defaultdict


from torchvision.datasets import LFWPeople, ImageFolder
import torchvision.transforms as transforms 

import csv
from collections import defaultdict
from pathlib import Path
import shutil

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    # transforms.CenterCrop(224),
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


train_data = LFWPeople(root = "./data", split = 'train', image_set = 'original', 
                      transform = transform, download = True)

test_data = LFWPeople(root = "./data", split = 'test', image_set = 'original', 
                      transform = transform, download = True)


lfw_train_dataloader = DataLoader(train_data, batch_size = 32, shuffle = True, drop_last = True)
lfw_test_dataloader = DataLoader(test_data, batch_size = 32, shuffle = False, drop_last = True)
# val_dataloader = DataLoader(val_dataset, batch_size = 32, shuffle = False, drop_last = True)

# Timy Face
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(ImageDataset, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        classes, class_to_idx = self._find_classes(self.root_dir)
        samples, label_to_indexes = self._make_dataset(self.root_dir, class_to_idx)
        print('samples num', len(samples))
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.label_to_indexes = label_to_indexes
        self.classes = sorted(self.label_to_indexes.keys())
        print('class num', len(self.classes))

    def _find_classes(self, dir):
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    
    def _make_dataset(self, root_dir, class_to_idx):
        root_dir = os.path.expanduser(root_dir)
        images = []
        label2index = defaultdict(list)
        image_index = 0
        for target in sorted(class_to_idx.keys()):
            d = os.path.join(root_dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
                    label2index[class_to_idx[target]].append(image_index)
                    image_index += 1

        return images, label2index

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)
    

# opening the CSV file
with open('data/tinyface/Testing_Set/gallery_match_img_ID_pairs.csv', mode ='r')as file:
   
  # reading the CSV file
  csvFile = csv.reader(file)
  i = 0
  # displaying the contents of the CSV file
  for lines in csvFile:
        if i == 0:
            i = 1
            continue
        
        Data_path = Path(f"data/tinyface/Testing_Set/Test/{lines[0]}")
        Data_path.mkdir(parents=True, # create parent directories if needed
                        exist_ok=True # if models directory already exists, don't error
                       )
        try:
            shutil.move(f'data/tinyface/Testing_Set/Gallery_Match/{lines[1][1:-1]}', Data_path)

        except shutil.Error as e:
            pass
        


tiny_train = ImageDataset(root_dir="data/tinyface/Training_Set", transform=transform)
# tint_test = ImageDataset(root_dir=)
# print(tiny_train)
tiny_test = ImageDataset(root_dir="data/tinyface/Testing_Set/Test", transform=transform)

tiny_train_dataloader = DataLoader(tiny_train, batch_size = 32, shuffle = True, drop_last = True)
tiny_test_dataloader = DataLoader(tiny_test, batch_size = 32, shuffle = False, drop_last = True)