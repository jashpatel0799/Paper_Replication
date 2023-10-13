import torch
import torchvision
from torchvision.datasets import Food101
from torchvision.datasets import FashionMNIST, MNIST
from torch.utils.data import Subset, DataLoader
import torchvision.transforms as transforms 

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

train_dataset = Food101(root = "./data", split= "train", transform = transform, download = True)
test_dataset = Food101(root = "./data", split= "test", transform = transform, download = True)

num_samples_train = int(0.20 * len(train_dataset))
indices_train = torch.randperm(len(train_dataset))[:num_samples_train]

num_samples_test = int(0.20 * len(test_dataset))
indices_test = torch.randperm(len(test_dataset))[:num_samples_test]

train_subset_dataset = Subset(train_dataset, indices_train)
test_subset_dataset = Subset(test_dataset, indices_test)

# train_dataset = FashionMNIST(root = "./data", train= True, transform = transform, download = True)
# test_dataset = FashionMNIST(root = "./data", train= False, transform = transform, download = True)

# train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True, drop_last = True)
# test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False, drop_last = True)

train_dataloader = DataLoader(train_dataset, batch_size = 32, shuffle = True, drop_last = True)
test_dataloader = DataLoader(test_dataset, batch_size = 32, shuffle = False, drop_last = True)