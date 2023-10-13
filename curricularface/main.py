import torch
import torch.nn as nn
# from torchvision.model import MyMACNN, DisLoss, DivLoss
from torchmetrics.classification import MulticlassAccuracy

import data, model, engine, utils


if __name__ == "__main__":
    # HYPERPARAMETERS
    model_name = "resnet_101" # write resnet_50 for resnet 50 and resnet_101 fpr resnet 101
    datasets = "TINYFace"  # write LFW for lfw datasets and TINYFace for tinyface
    SEED = 64
    NUM_EPOCH = 100
    LEARNIGN_RATE = 2e-3 # 3e-4, 4e-5, 7e-6, 5e-7, 3e-9
    inchannels = 3
    img_size = 112
    depth = 12
    num_class = 5749
    # CUDA_LAUNCH_BLOCKING=1

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # print(device)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    if model_name == "resnet_50":
      resnet = model.ResNet_50(input_size = [112, 112]).to(device)
    else:
      resnet = model.ResNet_101(input_size = [112, 112]).to(device)

    head = model.CurricularFace(in_features = 512, out_features = num_class)

    if datasets == "LFW":
      train_dataloader = data.lfw_train_dataloader
      test_dataloader = data.lfw_test_dataloader

    else:
      train_dataloader = data.tiny_train_dataloader
      test_dataloader = data.tiny_test_dataloader

    loss_fn = torch.nn.CrossEntropyLoss()
    accuracy_fn = MulticlassAccuracy(num_classes = num_class).to(device)
    optimizer = torch.optim.Adam(resnet.parameters(), lr = LEARNIGN_RATE)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-1, end_factor=1e-7, total_iters=100, last_epoch=-1, verbose=True)


    # print(f"lr: {scheduler.optimizer.param_groups[0]['lr']}")

    train_model, train_loss, test_loss, train_acc, test_acc = engine.train(model = resnet, head = head,
                                                                           train_dataloader = train_dataloader,
                                                                           test_dataloader = test_dataloader, 
                                                                           optimizer = optimizer,
                                                                           scheduler = scheduler,
                                                                           loss_fn = loss_fn, 
                                                                           accuracy_fn = accuracy_fn, 
                                                                           epochs = NUM_EPOCH, 
                                                                           device = device)

    utils.save_model(model = train_model, target_dir = "./save_model", model_name = f"{model_name}_sh_{datasets}.pth")

    utils.plot_graph(train_losses = train_loss, test_losses = test_loss, train_accs = train_acc, 
                        test_accs = test_acc, fig_name = f"plots/train_Loss_and_accuracy_plot_{model_name}_sh_{datasets}.jpg")