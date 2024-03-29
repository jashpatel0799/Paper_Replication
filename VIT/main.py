import torch
import torch.nn as nn
from torchvision.models import vit_b_16
from torchmetrics.classification import MulticlassAccuracy

import data, engine, model, utils

# HYPERPARAMETERS
SEED = 64
NUM_EPOCH = 50
LEARNIGN_RATE = 7e-6 # 3e-4, 4e-5, 7e-6, 5e-7, 3e-9
inchannels = 3
patch_size = 16
embedding_size = 768
img_size = 224
depth = 12
num_class = 101
# CUDA_LAUNCH_BLOCKING=1

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# print(device)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# BEST MODEL WEIGHTS LINK: https://drive.google.com/file/d/1Z5e9YejKjs3_EJTyDRpvDIl-vjJt3eZC/view?usp=sharing
vit_model = model.ViT(in_channels = inchannels, patch_size = patch_size,
                      embedding_size = embedding_size, img_size = img_size,
                      depth = depth, n_classes = num_class).to(device)

# torch_vit = vit_b_16().to(device)

# vit_model= nn.DataParallel(vit_model).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
accuracy_fn = MulticlassAccuracy(num_classes = num_class).to(device)
optimizer = torch.optim.SGD(vit_model.parameters(), lr = LEARNIGN_RATE, weight_decay=0.03)
# scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor = 0.1, end_factor = 0.07,
#                                               total_iters = NUM_EPOCH - 10)

# print(f"lr: {scheduler.optimizer.param_groups[0]['lr']}")

train_model, train_loss, test_loss, train_acc, test_acc = engine.train(model = vit_model, 
                                                                       train_dataloader = data.train_dataloader,
                                                                       test_dataloader = data.test_dataloader, 
                                                                       optimizer = optimizer,
                                                                    #    scheduler = scheduler,
                                                                       loss_fn = loss_fn, 
                                                                       accuracy_fn = accuracy_fn, 
                                                                       epochs = NUM_EPOCH, 
                                                                       device = device)

utils.save_model(model = train_model, target_dir = "./save_model", model_name = "vit_model_7e-6.pth")

utils.plot_graph(train_losses = train_loss, test_losses = test_loss, train_accs = train_acc, 
                 test_accs = test_acc, fig_name = "plots/vit_Loss_and_accuracy_plot_7e-6.jpg")
