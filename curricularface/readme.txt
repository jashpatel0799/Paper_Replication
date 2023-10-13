curricularface
    |-data          -----> for datasets
        |-store your datsets here   ## for tinyface store unzip folder here
        ## link for tinyface https://www.kaggle.com/datasets/panchalvipul/tinyface

    |-plots         -----> for to store loss and accuracy curve
    |-save_model    -----> to store trained model
    data.py         -----> python file to make datloader of datasets
    engine.py       -----> python file for train the model
    model.py        -----> python file to built model resnet50 and resnet101 
                           and also have **curricularface algorithm
    utils.py        -----> for utility functions like plot, svae model, load model
    main.py         -----> python file to execute the model (train the model)

    {model_name}_sh_{datsetname}.txt -----> txt file to save model results on different datasets



how to run code:
model_name = "resnet_101"  # write resnet_50 for resnet 50 and resnet_101 fpr resnet 101
datasets = "LFW"           # write LFW for lfw datasets and TINYFace for tinyface


but first run requirements.txt like pip install -r requirements.txt
now if you ar on GPU and have nvidia docker
write below line to run code:
----->   nohup python main.py > {modelname}_sh_{datasetname}.txt &
if you have python environment
----->   python main.py > {modelname}_sh_{datasetname}.txt &