import torch
from torch import nn
from torchvision.models import get_model
import torchvision.models as models

import numpy as np

import logging

from monai.metrics.confusion_matrix import ConfusionMatrixMetric

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import (
    Compose,
    ToTensor,
    Normalize,
    RandomAffine,
    ColorJitter,
    Resize
    )
from tqdm import tqdm
import itertools
from glob import glob
import os

from PIL import Image

from argparse import ArgumentParser
device = "cuda" if torch.cuda.is_available() else "cpu"

class DomainDS(Dataset):
    def __init__(self, list_of_file_paths, transform=None) -> None:
        super().__init__()
        self.list_of_file_paths = list_of_file_paths
        self.labels = [[i]*len(self.list_of_file_paths[i]) for i in range(len(self.list_of_file_paths))]
        self.list_of_file_paths = list(itertools.chain(*self.list_of_file_paths))
        self.labels = list(itertools.chain(*self.labels))
        assert len(self.labels) == len(self.list_of_file_paths), "Expect the same number of labels and files"
        self.transform = transform
    
    def __len__(self):
        return len(self.list_of_file_paths) 

    def __getitem__(self, index):
        img = Image.open(self.list_of_file_paths[index])
        label = self.labels[index]
        img = self.transform(img) if self.transform is not None else img
        return img, label
        

def train_epoch(train_dataloader, model, optimizer, loss_function):
    model.train()
    epoch_loss = []
    for batch in tqdm(train_dataloader):
        x,y = batch
        y_pred = model(x.to(device))
        loss = loss_function(y_pred.cpu(),y)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        epoch_loss.append(loss.detach().cpu().item())
    print(f"Mean Epoch Loss: {sum(epoch_loss)/len(epoch_loss)}")
    return model

def val_epoch(val_dataloader, model, metric):
    model.eval()
    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            x,y = batch
            y_pred = model(x.to(device))
            metric(y_pred.cpu().argmax(dim=1).unsqueeze(1), y.unsqueeze(1))
    return metric


def train_model(train_list, val_list, model_name, nof_datasets):
    train_transforms = Compose([
        Resize((520,720)), #mean size of images
        ToTensor(),
        Normalize(
            mean=(0.213, 0.213, 0.213), 
            std=(0.103, 0.103, 0.103)
            ),
            ColorJitter(),
    ]
    )

    train_ds = DomainDS(train_list, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_ds = DomainDS(val_list, transform=train_transforms)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = get_model(model_name, weights=None)

    loss = nn.CrossEntropyLoss()
    metric = ConfusionMatrixMetric(metric_name="accuracy")
    #Replace the last layer to nof classes 
    if model_name == "resnet18":
        model.fc = nn.Linear(512, nof_datasets)
    elif model_name == "mobilenet_v3_small":
        model.classifier[3] = nn.Linear(1024, nof_datasets)
    else:
        raise NotImplementedError(f"Adaptation of {model_name} has not yet been implemented")
    
    model = model.to(device=device)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)

    for epoch in range(1,30+1):
        
        if epoch % 10 == 0:
            print(f"Start eval on epoch {epoch}")
            validation_results = val_epoch(model=model,
                                           val_dataloader=val_loader,
                                           metric=metric
                                           )
            print(validation_results.aggregate(reduction="mean"))
            validation_results.reset()
        print(f"Start epoch {epoch}")
        train_epoch(
            train_dataloader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_function=loss
        )
    return model


def main(args):
    torch_model_names = sorted(name for name in models.__dict__
                     if name.islower()
                     and not name.startswith("__") and not name.startswith('get_') and not name.startswith('list_')
                     and callable(models.__dict__[name]))

    logging_dir = args.logging_dir
    if not os.path.exists(logging_dir):
        os.mkdir(logging_dir)
    logger = logging.Logger("Default Logger")
    logging.basicConfig(filename=os.path.join(logging_dir,"logs.txt"), encoding='utf-8', level=logging.DEBUG)

    model_name = args.classifier
    assert model_name in torch_model_names, f"Expected to be in torch models {torch_model_names}, but got {model_name}"
    dataset_list = args.dataset_list
    image_suffix = args.image_suffix
    nof_datasets = len(dataset_list)
    for dl in dataset_list:
        assert os.path.exists(dl), f"{dl} does not exist"
    
    #Generate the dataset list
    nof_splits = args.nof_splits
    dataset_list = [glob(os.path.join(x, "*" + y)) for x,y in zip(dataset_list,image_suffix)]

    #Split each of the datasets into 5 equal sized parts

    split_list = [np.array_split(x, nof_splits) for x in dataset_list]
    
    for split in range(nof_splits):
        cur_train_list = []
        cur_val_list = []
        for dataset_idx in range(nof_datasets):
            cur_dataset_trainset_contributions = [split_list[dataset_idx][i] for i in range(nof_splits) if i!= split]
            cur_dataset_trainset_contributions = list(itertools.chain(*cur_dataset_trainset_contributions))
            cur_dataset_val_contribution = list(split_list[dataset_idx][split])
            cur_train_list.append(cur_dataset_trainset_contributions)
            cur_val_list.append(cur_dataset_val_contribution)

        model = train_model(cur_train_list, cur_val_list, model_name, nof_datasets)
        torch.save(model.state_dict(), os.path.join(logging_dir, f"{model_name}_split_{split}.pth"))



    

if __name__ == '__main__':
    parser = ArgumentParser("Train a classifier to distinguish between two domains")
    parser.add_argument("--classifier", default="resnet18", help="Model architecture to use as a classifier")
    parser.add_argument("--dataset_list", default=[], nargs="+", help="List folder of images, where each folder represents one domain")
    parser.add_argument("--image_suffix", default=[], nargs="+", help="Identifier suffix of images over the gt")
    parser.add_argument("--nof_splits", default=5, help="Number of splits to be used to train and evaluate the models.")
    parser.add_argument("--logging_dir", default="log_dir", help="Directory to which the script logs")
    main(parser.parse_args())
    

