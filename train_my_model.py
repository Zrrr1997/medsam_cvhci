# %%
import os
import random
import monai
from os import makedirs
from os.path import join, isfile, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

from utils.mobileunet import MobileUNet

import cc3d
from torch.utils.tensorboard import SummaryWriter

import cv2

from matplotlib import pyplot as plt
import argparse
# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=str, default="./data/npy",
    help="Path to the npy data root."
)
parser.add_argument(
    "-pretrained_checkpoint", type=str, default=None,
    help="Path to the pretrained checkpoint."
)
parser.add_argument(
    "-resume", type=str, default=None,
    help="Path to the checkpoint to continue training."
)
parser.add_argument(
    "-work_dir", type=str, default="./workdir_mobileunet",
    help="Path to the working directory where checkpoints and logs will be saved."
)
parser.add_argument(
    "-num_epochs", type=int, default=10,
    help="Number of epochs to train."
)
parser.add_argument(
    "-batch_size", type=int, default=4,
    help="Batch size."
)
parser.add_argument(
    "-num_workers", type=int, default=8,
    help="Number of workers for dataloader."
)
parser.add_argument(
    "-device", type=str, default="cuda:0",
    help="Device to train on."
)
parser.add_argument(
    "-bbox_shift", type=int, default=5,
    help="Perturbation to bounding box coordinates during training."
)
parser.add_argument(
    "-lr", type=float, default=0.00005,
    help="Learning rate."
)
parser.add_argument(
    "-weight_decay", type=float, default=0.01,
    help="Weight decay."
)
parser.add_argument(
    "-iou_loss_weight", type=float, default=1.0,
    help="Weight of IoU loss."
)
parser.add_argument(
    "-seg_loss_weight", type=float, default=1.0,
    help="Weight of segmentation loss."
)
parser.add_argument(
    "-ce_loss_weight", type=float, default=1.0,
    help="Weight of cross entropy loss."
)
parser.add_argument(
    "--sanity_check", action="store_true",
    help="Whether to do sanity check for dataloading."
)
parser.add_argument(
    '--model',
    type=str,
    default='mobileunet',
    choices=['mobileunet'],
    help='Model architecture'
)

parser.add_argument(
    '--show_preds',
    default=False,
    action='store_true',
    help='Show predictions of the model'
)

parser.add_argument(
    '--crop_instances',
    default=False,
    action='store_true',
    help='Crop image with a random bbox to train'
)

parser.add_argument(
    '--compute_fp_fn',
    default=False,
    action='store_true',
    help='Compute False Positives and False Negatives'
)

args = parser.parse_args()
# %%
work_dir = args.work_dir
data_root = args.data_root
model_checkpoint = args.pretrained_checkpoint
num_epochs = args.num_epochs
batch_size = args.batch_size
num_workers = args.num_workers
device = args.device
bbox_shift = args.bbox_shift
lr = args.lr
weight_decay = args.weight_decay
iou_loss_weight = args.iou_loss_weight
seg_loss_weight = args.seg_loss_weight
ce_loss_weight = args.ce_loss_weight
do_sancheck = args.sanity_check
checkpoint = args.resume
show_preds = args.show_preds
compute_fp_fn = args.compute_fp_fn

makedirs(work_dir, exist_ok=True)

log_dir = "./logs"

# Create a summary writer
writer = SummaryWriter(work_dir)


# %%
torch.cuda.empty_cache()
os.environ["OMP_NUM_THREADS"] = "4" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "6" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6" # export NUMEXPR_NUM_THREADS=6

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.45])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, 0.45])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='blue', facecolor=(0,0,0,0), lw=2))

def cal_iou(result, reference):
    
    intersection = torch.count_nonzero(torch.logical_and(result, reference), dim=[i for i in range(1, result.ndim)])
    union = torch.count_nonzero(torch.logical_or(result, reference), dim=[i for i in range(1, result.ndim)])
    
    iou = intersection.float() / union.float()
    
    return iou.unsqueeze(1)

def plot_preds(image, gt2D, logits_pred):
    _, axes = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure with 1 row and 3 columns

    # Plot GT2D
    axes[0].imshow(gt2D[0][0].detach().numpy()*255, cmap='gray')
    axes[0].set_title('GT2D')
    axes[0].axis('off')

    # Plot original image
    axes[1].imshow(image[0][0].detach().numpy())
    axes[1].set_title('Original Image')
    axes[1].axis('off')

    # Plot logits_pred
    low_res_pred = torch.sigmoid(logits_pred)  
    low_res_pred = low_res_pred.squeeze().detach().numpy()  
    my_model_mask = (low_res_pred > 0.5).astype(np.uint8)
    axes[2].imshow(my_model_mask[0], cmap='gray')
    axes[2].set_title('Logits Pred')
    axes[2].axis('off')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

# %%
class NpyDataset(Dataset): 
    def __init__(self, data_root, image_size=256, bbox_shift=5, data_aug=True):
        self.data_root = data_root
        self.gt_path = join(data_root, 'gts')
        self.img_path = join(data_root, 'imgs')
        self.gt_path_files = sorted(glob(join(self.gt_path, '*.npy'), recursive=True))
        self.gt_path_files = [
            file for file in self.gt_path_files
            if isfile(join(self.img_path, basename(file)))
        ]
        self.image_size = image_size
        self.target_length = image_size
        self.bbox_shift = bbox_shift
        self.data_aug = data_aug
    
    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        img_name = basename(self.gt_path_files[index])
        assert img_name == basename(self.gt_path_files[index]), 'img gt name error' + self.gt_path_files[index] + self.npy_files[index]
        img_3c = np.load(join(self.img_path, img_name), 'r', allow_pickle=True) # (H, W, 3)

        gt = np.load(self.gt_path_files[index], 'r', allow_pickle=True) # multiple labels [0, 1,4,5...], (256,256)
        
        connected_components = cc3d.connected_components(gt.copy(), connectivity=26)

        boxes= []
        for label in np.unique(connected_components)[1:]:  # Skip background label 0

            labeled_mask = connected_components == label
            indices = np.nonzero(labeled_mask)

            bounding_box = [min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0])]  # Format: (x_min, y_min, x_max, y_max)
            boxes.append(bounding_box)

        bboxes = np.array(boxes)
        if bboxes.shape[0] == 0:
            random_ind = 0
            bboxes = [[0, 0, img_3c.shape[0], img_3c.shape[1]]]
        else:
            random_ind = np.random.randint(0, bboxes.shape[0])
        bboxes = np.expand_dims(bboxes[random_ind], axis=0) # use one random bbox from all bboxes during training
        box = bboxes[0]

        # expand bounding box to avoid degenerate bboxes
        if (box[3] - box[1]) <= 1:
            box[3] += 1
        if (box[2] - box[0]) <= 1:
            box[2] += 1

        if args.crop_instances:
            img_3c = img_3c[box[1]:box[3], box[0]:box[2]] # Crop image to bounding box
            gt = gt[box[1]:box[3], box[0]:box[2]]





        img_resize = self.resize_longest_side(img_3c)
        # Resizing
        img_resize = (img_resize - img_resize.min()) / np.clip(img_resize.max() - img_resize.min(), a_min=1e-8, a_max=None) # normalize to [0, 1], (H, W, 3
        img_padded = self.pad_image(img_resize) # (256, 256, 3)
        # convert the shape to (3, H, W)
        img_padded = np.transpose(img_padded, (2, 0, 1)) # (3, 256, 256)
        assert np.max(img_padded)<=1.0 and np.min(img_padded)>=0.0, 'image should be normalized to [0, 1]'
        gt = cv2.resize(
            gt,
            (img_resize.shape[1], img_resize.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(np.uint8)
        gt = self.pad_image(gt) # (256, 256)


        label_ids = np.unique(gt)[1:]
        try:
            if args.crop_instances:
                gt2D = np.uint8(gt == random.choice(label_ids.tolist())) # only one label, (256, 256)
            else:
                gt2D = (gt > 0).astype(np.uint8)
        except:
            print(img_name, 'label_ids.tolist()', label_ids.tolist())
            gt2D = np.uint8(gt == np.max(gt)) # only one label, (256, 256)
        # add data augmentation: random fliplr and random flipud
        if self.data_aug:
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-1))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-1))
                # print('DA with flip left right')
            if random.random() > 0.5:
                img_padded = np.ascontiguousarray(np.flip(img_padded, axis=-2))
                gt2D = np.ascontiguousarray(np.flip(gt2D, axis=-2))
                # print('DA with flip upside down')
        gt2D = np.uint8(gt2D > 0)


        
        return {
            "image": torch.tensor(img_padded).float(),
            "gt2D": torch.tensor(gt2D[None, :,:]).long(),
            "bboxes": torch.tensor(bboxes[None, None, ...]).float(), # (B, 1, 4)
            "image_name": img_name,
            "new_size": torch.tensor(np.array([img_resize.shape[0], img_resize.shape[1]])).long(),
            "original_size": torch.tensor(np.array([img_3c.shape[0], img_3c.shape[1]])).long()
        }

    def resize_longest_side(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        long_side_length = self.target_length
        oldh, oldw = image.shape[0], image.shape[1]
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww, newh = int(neww + 0.5), int(newh + 0.5)
        target_size = (neww, newh)


        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    def pad_image(self, image):
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        # Pad
        h, w = image.shape[0], image.shape[1]
        padh = self.image_size - h
        padw = self.image_size - w
        if len(image.shape) == 3: ## Pad image
            image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
        else: ## Pad gt mask
            image_padded = np.pad(image, ((0, padh), (0, padw)))

        return image_padded


#%% sanity test of dataset class
if do_sancheck:
    tr_dataset = NpyDataset(data_root, data_aug=True)
    tr_dataloader = DataLoader(tr_dataset, batch_size=8, shuffle=True)
    for step, batch in enumerate(tr_dataloader):
        # show the example
        _, axs = plt.subplots(1, 2, figsize=(10, 10))
        idx = random.randint(0, 4)

        image = batch["image"]
        gt = batch["gt2D"]
        bboxes = batch["bboxes"]
        names_temp = batch["image_name"]

        axs[0].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[0])
        show_box(bboxes[idx].numpy().squeeze(), axs[0])
        axs[0].axis('off')
        # set title
        axs[0].set_title(names_temp[idx])
        idx = random.randint(4, 7)
        axs[1].imshow(image[idx].cpu().permute(1,2,0).numpy())
        show_mask(gt[idx].cpu().squeeze().numpy(), axs[1])
        show_box(bboxes[idx].numpy().squeeze(), axs[1])
        axs[1].axis('off')
        # set title
        axs[1].set_title(names_temp[idx])
        plt.subplots_adjust(wspace=0.01, hspace=0)
        plt.savefig(
            join(work_dir, 'medsam_lite-train_bbox_prompt_sanitycheck_DA.png'),
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
        break



if args.model == 'mobileunet':
    model = MobileUNet()
else:
    print(f'Only mobileunet is implemented...')
    exit()



if model_checkpoint is not None:
    if isfile(model_checkpoint):
        print(f"Finetuning with pretrained weights {model_checkpoint}")
        medsam_lite_ckpt = torch.load(
            model_checkpoint,
            map_location="cpu"
        )
        model.load_state_dict(medsam_lite_ckpt, strict=True)
    else:
        print(f"Pretained weights {model_checkpoint} not found, training from scratch")

model = model.to(device)
model.train()

# %%
print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
# %%
optimizer = optim.AdamW(
    model.parameters(),
    lr=lr,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=weight_decay,
)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.9,
    patience=5,
    cooldown=0
)
seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction='mean')
ce_loss = nn.BCEWithLogitsLoss(reduction='mean')
# %%
train_dataset = NpyDataset(data_root=data_root, data_aug=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

if checkpoint and isfile(checkpoint):
    print(f"Resuming from checkpoint {checkpoint}")
    checkpoint = torch.load(checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"]
    best_loss = checkpoint["loss"]
    print(f"Loaded checkpoint from epoch {start_epoch}")
else:
    start_epoch = 0
    best_loss = 1e10
# %%
train_losses = []
fps, fns = [], []

for epoch in range(start_epoch + 1, num_epochs):
    epoch_loss = [1e10 for _ in range(len(train_loader))]
    epoch_start_time = time()
    pbar = tqdm(train_loader)
    for step, batch in enumerate(pbar):
        image = batch["image"]
        gt2D = batch["gt2D"]
        boxes = batch["bboxes"]

        optimizer.zero_grad()
        image, gt2D, boxes = image.to(device), gt2D.to(device), boxes.to(device)
        

        logits_pred = model(image) 
        
        l_seg = seg_loss(logits_pred, gt2D)
        l_ce = ce_loss(logits_pred, gt2D.float())
        loss = seg_loss_weight * l_seg + ce_loss_weight * l_ce

        epoch_loss[step] = loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        pbar.set_description(f"Epoch {epoch} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, loss: {loss.item():.4f}")

        if show_preds:
            plot_preds(image, gt2D, logits_pred)
        if compute_fp_fn:
            if epoch > start_epoch + 1:
                print('FP', np.nanmean(fps))
                print('FN', np.nanmean(fns))
                exit()
            low_res_pred = torch.sigmoid(logits_pred)  
            low_res_pred = (low_res_pred.squeeze().cpu().detach().numpy() > 0.5)
            gt = gt2D.cpu().detach().numpy()
            fp = (gt != low_res_pred) * (gt == 0)
            fn = (gt != low_res_pred) * (gt != 0)
            vol = (gt.shape[0] * gt.shape[1] * gt.shape[2] * gt.shape[3])
            print(np.sum(fp) / vol)
            print(np.sum(fn) / vol)
            fps.append(np.sum(fp) / vol)
            fns.append(np.sum(fn) / vol)



    epoch_end_time = time()
    epoch_loss_reduced = sum(epoch_loss) / len(epoch_loss)
    train_losses.append(epoch_loss_reduced)
    lr_scheduler.step(epoch_loss_reduced)
    model_weights = model.state_dict()
    checkpoint = {
        "model": model_weights,
        "epoch": epoch,
        "optimizer": optimizer.state_dict(),
        "loss": epoch_loss_reduced,
        "best_loss": best_loss,
    }
    torch.save(checkpoint, join(work_dir, "model_latest.pth"))
    if epoch_loss_reduced < best_loss:
        print(f"New best loss: {best_loss:.4f} -> {epoch_loss_reduced:.4f}")
        best_loss = epoch_loss_reduced
        checkpoint["best_loss"] = best_loss
        torch.save(checkpoint, join(work_dir, "model_best.pth"))



    # Write the loss value to TensorBoard
    writer.add_scalar('Epoch Loss', epoch_loss_reduced, global_step=epoch)  # 'global_step' is the step or epoch number

    # Close the writer
writer.close()

