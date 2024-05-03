from os import makedirs
from os.path import join, basename
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes, binary_closing

import os
import gc
import threading
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interpn


from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn\


from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
from tiny_vit_sam import TinyViT
from matplotlib import pyplot as plt
import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime
import cc3d

from utils.mobileunet import MobileUNet
from utils.medsam_model import my_model_Lite

#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
torch.cuda.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='test_demo/imgs/',
    # required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='test_demo/segs/',
    help='directory to save the prediction',
)
parser.add_argument(
    '-my_model',
    type=str,
    default=None,
    help='path to the checkpoint of my_model-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '--debug_vis',
    default=False,
    action='store_true',
    help='visualize predictions to debug'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='./overlay',
    help='directory to save the overlay image'
)

parser.add_argument(
    '-model',
    type=str,
    default='medsam',
    choices=['medsam', 'grabcut', 'mobileunet', 'th'],
    help='Model architecture'
)

parser.add_argument(
    '-th',
    type=int,
    default=50,
    help='percentile thresholding for the PET data',
)

parser.add_argument(
    '-sampling_rate',
    type=int,
    default=2,
    help='1/sampling rate of all slices are inferred, the other are interpolated',
)

parser.add_argument(
    '--filter_background',
    default=True,
    action='store_false',
    help='whether to omit all the predictions outside the bbox'
)

parser.add_argument(
    '--force_volume',
    default=False,
    action='store_true',
    help='whether to force the prediction to [20,80] percent of the bbox volume'
)

parser.add_argument(
    '-upper_interval',
    type=float,
    default=0.81,
    help='upper value for the interval for volume forcing',
)

parser.add_argument(
    '-lower_interval',
    type=float,
    default=0.19,
    help='lower value for the interval for volume forcing',
)


args = parser.parse_args()

import resource

if not args.save_overlay:
    # Set the soft limit for memory usage to 8GB (8 * 1024 * 1024 * 1024 bytes)
    soft, hard = 8 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024
    #resource.setrlimit(resource.RLIMIT_AS, (soft, hard))

data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
num_workers = args.num_workers
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)


makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256

def resize_longest_side(image, target_length=256):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (max(neww, 2), max(newh, 2))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size=256):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def kmean(img, k=3):

    or_shape = img.shape
    img = img.reshape((-1, 3))
    img = np.float32(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)

    _, labels, (centers) = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    labels = labels.reshape(or_shape[0], or_shape[1], 1)


    return labels


def largest_connected_component(segmentation):
    connected_components = cc3d.connected_components(segmentation, connectivity=26)

    max_size = 0
    largest_label = None
    for label in np.unique(connected_components)[1:]:  # Skip background label 0
        size = np.sum(label == connected_components)
        if size >= max_size:
            max_size = size
            largest_label = label


    largest_ccp = (((connected_components == largest_label)) * 1).astype(np.uint16)

    largest_ccp = binary_fill_holes(largest_ccp)

    return largest_ccp

@torch.no_grad()
def postprocess_masks(masks, new_size, original_size):
    """
    Do cropping and resizing

    Parameters
    ----------
    masks : torch.Tensor
        masks predicted by the model
    new_size : tuple
        the shape of the image after resizing to the longest side of 256
    original_size : tuple
        the original shape of the image

    Returns
    -------
    torch.Tensor
        the upsampled mask to the original size
    """
    # Crop
    masks = masks[..., :new_size[0], :new_size[1]]
    # Resize

    masks = F.interpolate(
        masks,
        size=(original_size[0], original_size[1]),
        mode="bilinear",
        align_corners=False,
    )
    return masks


def show_mask(mask, ax, mask_color=None, alpha=0.5):
    """
    show mask on the image

    Parameters
    ----------
    mask : numpy.ndarray
        mask of the image
    ax : matplotlib.axes.Axes
        axes to plot the mask
    mask_color : numpy.ndarray
        color of the mask
    alpha : float
        transparency of the mask
    """
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1) #.astype(np.uint8) * 255
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor='blue'):
    """
    show bounding box on the image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    ax : matplotlib.axes.Axes
        axes to plot the bounding box
    edgecolor : str
        color of the bounding box
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0,0,0,0), lw=2))


def get_bbox256(mask_256, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256


def resize_box_to_256(box, original_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box


@torch.no_grad()
def my_model_inference(my_model_model, img_embed, box_256, new_size, original_size):
    """
    Perform inference using the my_model model.

    Args:
        my_model_model (my_modelModel): The my_model model.
        img_embed (torch.Tensor): The image embeddings.
        box_256 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=img_embed.device)
    sparse_embeddings, dense_embeddings = my_model_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = my_model_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=my_model_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )



    low_res_pred = my_model_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    my_model_seg = (low_res_pred > 0.5).astype(np.uint8)

    return my_model_seg, iou

def get_medsam(my_model_checkpoint):
    my_model_lite_image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[
            64, ## (64, 256, 256)
            128, ## (128, 128, 128)
            160, ## (160, 64, 64)
            320 ## (320, 64, 64)
        ],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8
    )

    my_model_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    my_model_lite_mask_decoder = MaskDecoder(
        num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
    )

    my_model_lite_model = my_model_Lite(
        image_encoder = my_model_lite_image_encoder,
        mask_decoder = my_model_lite_mask_decoder,
        prompt_encoder = my_model_lite_prompt_encoder
    )

    my_model_checkpoint = torch.load(my_model_checkpoint, map_location='cpu')
    my_model_lite_model.load_state_dict(my_model_checkpoint)
    my_model_lite_model.to(device)
    my_model_lite_model.eval()
    return my_model_lite_model


def infer_mobileunet(model, data, res):
    with torch.no_grad():
        res.append(model(data))

#my_second_model = my_model
#models = [my_model.eval(), my_second_model.eval()]


def grabcut_pred(rect, mask, image, new_size, original_size):
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")
    # apply GrabCut using the the bounding box segmentation method
    (mask, bgModel, fgModel) = cv2.grabCut(image, mask, rect, bgModel,
        fgModel, iterCount=1, mode=cv2.GC_INIT_WITH_RECT)

    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
    outputMask = (outputMask * 255).astype("uint8")



    x_min, y_min, x_max, y_max = rect

    # Create a mask to represent the region outside the bounding box
    outside_bbox_mask = np.zeros_like(outputMask)
    outside_bbox_mask[y_min:y_max, x_min:x_max] = 1

    # Apply the outside bounding box mask to the output mask
    outputMask[outside_bbox_mask == 0] = 0


    # Crop
    masks = torch.Tensor(outputMask[..., :new_size[0], :new_size[1]]).unsqueeze(0).unsqueeze(0)
    # Resize


    masks = F.interpolate(
        masks,
        size=(original_size[0], original_size[1]),
        mode="bilinear",
        align_corners=False,
    ).squeeze()

    return masks

def get_mobileunet_path(img_npz_file):
    if 'Microscope' in img_npz_file or 'Microscopy' in img_npz_file:
        return 'workdir_mobileunet/best_Microscopy.pth'
    elif 'Dermoscopy' in img_npz_file:
        return 'workdir_mobileunet/best_Dermoscopy.pth'
    elif 'Fundus' in img_npz_file:
        return 'workdir_mobileunet/best_Fundus.pth'
    elif 'Mammography' in img_npz_file:
        return 'workdir_mobileunet/best_Mammography.pth'
    elif 'Endoscopy' in img_npz_file:
        return 'workdir_mobileunet/best_Endoscopy.pth'
    elif 'OCT' in img_npz_file:
        return 'workdir_mobileunet/best_OCT.pth'
    elif 'US' in img_npz_file:
        return 'workdir_mobileunet/best_US.pth'
    elif 'XRay' in img_npz_file:
        return 'workdir_mobileunet/best_XRay.pth'

def get_mobileunet(path, device):
    my_model = MobileUNet()
    my_model_checkpoint = torch.load(path, map_location='cpu')
    my_model.load_state_dict(my_model_checkpoint['model'])
    my_model.to(torch.device('cpu'))
    my_model.eval()
    return my_model

# Create an instance of the model

def my_model_infer_npz_2D(img_npz_file, model_name):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)


    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    if 'boxes' in npz_data.keys():
        boxes = npz_data['boxes']
    else:
        gts = npz_data['gts']
        connected_components = cc3d.connected_components(gts, connectivity=26)

        boxes= []
        for label in np.unique(connected_components)[1:]:  # Skip background label 0
            labeled_mask = connected_components == label
            indices = np.nonzero(labeled_mask)
            bounding_box = (min(indices[1]), min(indices[0]), max(indices[1]), max(indices[0]))  # Format: (x_min, y_min, x_max, y_max)
            boxes.append(bounding_box)


    segs = np.zeros(img_3c.shape[:2], dtype=np.uint16)

    if model_name != 'mobileunet':

        ## preprocessing
        img_256 = resize_longest_side(img_3c, 256)
        newh, neww = img_256.shape[:2]

        img_256_norm = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )

        img_256_padded_non_norm = pad_image(img_256, 256).astype(np.uint8)
        img_256_padded = pad_image(img_256_norm, 256)


    if model_name == 'medsam':
        model_path = 'work_dir/LiteMedSAM/lite_medsam.pth'
        my_model_lite_model = get_medsam(model_path)
        with torch.no_grad():
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device) # (B, 3, 256, 256)
            image_embedding = my_model_lite_model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box_to_256(box, original_size=(H, W))


        if model_name == 'grabcut':
            my_model_mask = grabcut_pred(box256, segs, img_256_padded_non_norm, (newh, neww), (H, W)).to(torch.uint8) # GrabCut works on non-normalized images

        elif model_name == 'mobileunet':
            model_path = get_mobileunet_path(img_npz_file)
            my_model = get_mobileunet(model_path, device)
            rgb = [np.sum(img_3c[:,:,i]) for i in range(3)]
            bbox = list(box)

            # Avoid empty bounding boxes
            if bbox[2] - bbox[0] <= 1:
                bbox[2] += 1
                bbox[0] -= 1
                bbox[0] = max(0, bbox[0])
            if bbox[3] - bbox[1] <= 1:
                bbox[3] += 1
                bbox[1] -= 1
                bbox[1] = max(0, bbox[1])


            img_3c_input = img_3c[bbox[1]:bbox[3], bbox[0]:bbox[2]] # Crop image to bounding box
            H, W = img_3c_input.shape[:2]

            ## preprocessing
            img_256 = resize_longest_side(img_3c_input, 256)
            newh, neww = img_256.shape[:2]

            img_256_norm = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )

            img_256_padded = pad_image(img_256_norm, 256)

            if ('Microscope' in img_npz_file or 'Microscopy' in img_npz_file) and ((len([el for el in rgb if el == 0]) == 1) or (np.sum(img_3c[:,:,0] - img_3c[:,:,1]) == 0)): # Grayscale or empty color channel

                k = 2
                kmeans_mask = kmean(img_3c_input, k=k).astype(np.uint8)
                sh = kmeans_mask.shape

                center_context = np.mean(kmeans_mask[int(sh[0]/2)-5:int(sh[0]/2)+5,int(sh[1]/2)-5:int(sh[1]/2)+5,:])

                if center_context < 0.5: # invert prediction if the central structure is considered as background
                    kmeans_mask = np.logical_not(kmeans_mask).astype(np.uint8)

                #corner_context = np.mean(kmeans_mask[:5, :5]) + np.mean(kmeans_mask[-5:, :5]) + np.mean(kmeans_mask[-5:, -5:]) +  + np.mean(kmeans_mask[:5, -5:])
                #corner_context /= 4

                if center_context < 0.5: # invert prediction if the central structure is considered as background
                    kmeans_mask = np.logical_not(kmeans_mask).astype(np.uint8)
                kmeans_mask = largest_connected_component(kmeans_mask).astype(np.uint8)

                temp_pred = np.zeros_like(segs).astype(np.uint16)

                temp_pred[bbox[1]:bbox[3], bbox[0]:bbox[2]] = kmeans_mask[:,:,0]

                temp_pred = largest_connected_component(temp_pred).astype(np.uint8)


            else:

                my_model_mask = my_model(torch.Tensor(img_256_padded.transpose(2, 0, 1)).unsqueeze(0))
                '''
                input_tensor = torch.Tensor(img_256_padded.transpose(2, 0, 1)).unsqueeze(0)
                results1, results2 = [], []

                thread1 = threading.Thread(target=infer_mobileunet, args=(models[0], input_tensor, results1))
                thread2 = threading.Thread(target=infer_mobileunet, args=(models[1], input_tensor, results2))
                thread1.start()
                thread2.start()

                # Wait for the threads to finish
                thread1.join()
                thread2.join()
                '''
                low_res_pred = postprocess_masks(my_model_mask, (newh, neww), (H, W))
                low_res_pred = torch.sigmoid(low_res_pred)
                low_res_pred = low_res_pred.squeeze().cpu().numpy()
                my_model_mask = (low_res_pred > 0.5).astype(np.uint16)

                my_model_mask = largest_connected_component(my_model_mask)

                temp_pred = np.zeros_like(segs).astype(np.uint16)

                temp_pred[bbox[1]:bbox[3], bbox[0]:bbox[2]] = my_model_mask
                temp_pred = largest_connected_component(temp_pred).astype(np.uint8)






        elif model_name == 'medsam':
            box256 = box256[None, ...] # (1, 4)
            my_model_mask, _ = my_model_inference(my_model_lite_model, image_embedding, box256, (newh, neww), (H, W))
        if args.debug_vis:
            cv2.imshow('img', cv2.resize(img_3c[bbox[1]:bbox[3], bbox[0]:bbox[2]], (256, 256), interpolation=cv2.INTER_AREA))
            if model_name == 'mobileunet':
                cv2.imshow('test', cv2.resize(((temp_pred>0) * 255).astype(np.uint8)[bbox[1]:bbox[3], bbox[0]:bbox[2]], (256,256), interpolation=cv2.INTER_AREA))
                #cv2.imshow('gts', cv2.resize(((gts) ).astype(np.uint8)[box[1]:box[3], box[0]:box[2]], (256,256), interpolation=cv2.INTER_AREA))

            elif model_name == 'medsam':
                cv2.imshow('test', cv2.resize(((my_model_mask[box[1]:box[3], box[0]:box[2]]>0) * 255).astype(np.uint8), (256,256), interpolation=cv2.INTER_AREA))

            cv2.waitKey(0)
            #cv2.destroyAllWindows()


        if model_name == "mobileunet":
            segs[temp_pred>0] = idx
        else:
            segs[my_model_mask>0] = idx
        gc.collect()
        # print(f'{npz_name}, box: {box}, predicted iou: {np.round(iou_pred.item(), 4)}')
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )
    # visualize image, mask and bounding box
    if save_overlay:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("my_model Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()
    gc.collect()

def ndgrid(*args,**kwargs):
    """
    Same as calling ``meshgrid`` with *indexing* = ``'ij'`` (see
    ``meshgrid`` for documentation).
    """
    kwargs['indexing'] = 'ij'
    return np.meshgrid(*args,**kwargs)

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

def signed_bwdist(im):
    '''
    Find perim and return masked image (signed/reversed)
    '''
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

def bwdist(im):
    '''
    Find distance map of image
    '''
    dist_im = distance_transform_edt(1-im)
    return dist_im

def interp_shape(top, bottom, precision):
    '''
    Interpolate between two contours

    Input: top
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision>2:
        print("Error: Precision must be between 0 and 1 (float)")

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids
    points = (np.r_[0, 2], np.arange(r), np.arange(c))

    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
    xi = np.c_[np.full((r*c),precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out

def compute_3d_bounding_boxes(binary_mask):

    # Label connected components
    connected_components = cc3d.connected_components(binary_mask, connectivity=26)

    bounding_boxes = []
    for label in np.unique(connected_components)[1:]:  # Skip background label 0
        labeled_mask = connected_components == label
        indices = np.nonzero(labeled_mask)

        bounding_box = (
            min(indices[2]), min(indices[1]), min(indices[0]),
            max(indices[2]), max(indices[1]), max(indices[0])
        )  # Format: [[x_min, y_min, z_min, x_max, y_max, z_max]]

        bounding_boxes.append(bounding_box)

    return bounding_boxes
def box_vol(box3D):
    return max((box3D[3] - box3D[0]), 1) * max((box3D[4] - box3D[1]), 1) * max((box3D[5] - box3D[2]), 1)
def box_center(box3D):
    return (int(box3D[0] + (box3D[3] - box3D[0]) / 2), int(box3D[1] + (box3D[4] - box3D[1]) / 2), int(box3D[2] + (box3D[5] - box3D[2]) / 2))
def my_model_infer_npz_3D(img_npz_file, model_name):

    if model_name == 'medsam':
        model_path = 'work_dir/LiteMedSAM/lite_medsam.pth'
        my_model_lite_model = get_medsam(model_path)
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    segs = np.zeros_like(img_3D, dtype=np.uint16)

    if 'boxes' in npz_data.keys():
        boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]
    else:
        gts = npz_data['gts']
        gts = (cc3d.connected_components(gts, connectivity=26) > 0)
        boxes_3D = compute_3d_bounding_boxes(gts)


    for idx, box3D in enumerate(boxes_3D, start=1):

        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint16)
        x_min, y_min, z_min, x_max, y_max, z_max = box3D

        assert z_min <= z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        sampling_rate = args.sampling_rate
        sampled_z = np.arange(z_middle, max(z_max, z_middle+1), sampling_rate).astype(np.uint16)
        if max(z_max, z_middle) not in sampled_z:
            sampled_z = np.append(sampled_z, max(z_max, z_middle))


        for z in range(z_middle, max(z_max, z_middle+1)):

            if z not in sampled_z:
                continue

            if model_name == 'th': # No slice-wise inference for thresholds
                break
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            if not model_name == 'grabcut' and not model_name == 'th':
                img_256 = (img_256 - img_256.min()) / np.clip(
                    img_256.max() - img_256.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)


            if model_name == 'medsam':
                if z in sampled_z:
                    # convert the shape to (3, H, W)
                    img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
                    # get the image embedding
                    with torch.no_grad():
                        image_embedding = my_model_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs_3d_temp[z-1, :, :]
                pre_seg256 = resize_longest_side(pre_seg)
                if np.max(pre_seg256) > 0:
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))

            if model_name == 'grabcut':
                img_2d_seg = grabcut_pred(box_256.astype(np.uint8), np.zeros_like(img_256, dtype=np.uint8), img_256, (new_H, new_W), (H, W)).to(torch.uint8) # GrabCut works on non-normalized images
            elif model_name == 'medsam':

                if z in sampled_z:
                    img_2d_seg, _ = my_model_inference(my_model_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])

            segs_3d_temp[z, img_2d_seg>0] = idx
        for z_ind, z in enumerate(sampled_z):
            if z == sampled_z[-1]:
                break
            for zs in range(z+1, sampled_z[z_ind + 1]):
                step_size = 1 / (sampled_z[z_ind + 1] - z)
                img_2d_seg = interp_shape((segs_3d_temp[z] == idx).astype(np.uint8), (segs_3d_temp[sampled_z[z_ind + 1]] == idx).astype(np.uint8), step_size * (zs - z)).astype(np.uint8)
                segs_3d_temp[zs, img_2d_seg>0] = idx





        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        sampled_z = np.arange(z_min, z_middle-1, sampling_rate).astype(np.uint16)
        if z_middle-1 not in sampled_z:
            sampled_z = np.append(sampled_z, z_middle-1)
        for z in range(z_middle-1, z_min, -1):
            if z not in sampled_z:
                continue
            if model_name == 'th':
                break
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]
            if model_name == 'medsam':
                img_256 = (img_256 - img_256.min()) / np.clip(
                    img_256.max() - img_256.min(), a_min=1e-8, a_max=None
                )  # normalize to [0, 1], (H, W, 3)
                ## Pad image to 256x256
            img_256 = pad_image(img_256)

            if model_name == 'medsam':
                img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
                # get the image embedding
                with torch.no_grad():
                    image_embedding = my_model_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[z+1, :, :]
            pre_seg256 = resize_longest_side(pre_seg)

            if np.max(pre_seg256) > 0:
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            if model_name == 'grabcut':
                img_2d_seg = grabcut_pred(box_256.astype(np.uint8), np.zeros_like(img_256, dtype=np.uint8), img_256, (new_H, new_W), (H, W)).to(torch.uint8) # GrabCut works on non-normalized images
            elif model_name == 'medsam':
                img_2d_seg, _ = my_model_inference(my_model_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])

            segs_3d_temp[z, img_2d_seg>0] = idx
        for z_ind, z in enumerate(sampled_z):
            if z == sampled_z[-1]:
                break
            for zs in range(z+1, sampled_z[z_ind + 1]):
                step_size = 1 / (sampled_z[z_ind + 1] - z)
                img_2d_seg = interp_shape((segs_3d_temp[z] == idx).astype(np.uint8), (segs_3d_temp[sampled_z[z_ind + 1]] == idx).astype(np.uint8), step_size * (zs - z)).astype(np.uint8)
                segs_3d_temp[zs, img_2d_seg>0] = idx

        if model_name == 'th':
            segs_3d_temp = (img_3D > args.th) * idx
        curr_seg = (segs_3d_temp == idx) * 1

        x_min, y_min, z_max, x_max, y_max, z_max = box3D
        if args.filter_background: # Omit everything outside the bbox
            outside_bbox_mask = np.zeros_like(segs_3d_temp)
            outside_bbox_mask[z_min:z_max, y_min:y_max, x_min:x_max] = 1
            # Apply the outside bounding box mask to the output mask
            curr_seg[outside_bbox_mask == 0] = 0 # for th
            segs_3d_temp[outside_bbox_mask == 0] = 0 # for everything else
        if args.force_volume: # try to push the tumor to bbox ratio within a certain range
            if model_name == 'th':
                ratio = np.sum(curr_seg) / box_vol(box3D)
                structuring_element = np.ones((2, 2, 2), dtype=np.uint8)

                if ratio == 0.0: # initialize mask with at least one voxel
                    box_c = box_center(box3D)
                    curr_seg[box_c[2], box_c[1], box_c[0]] = 1
                if ratio < args.lower_interval:
                    c_ratio = ratio
                    while c_ratio < args.lower_interval:
                        curr_seg = binary_dilation(curr_seg, structure=structuring_element)
                        c_ratio = np.sum(curr_seg / box_vol(box3D))
                    if c_ratio >= 1.0:
                        curr_seg = binary_erosion(curr_seg, structure=structuring_element)
                        c_ratio = np.sum(curr_seg / box_vol(box3D))
                    if c_ratio == 0:
                        box_c = box_center(box3D)
                        curr_seg[box_c[2], box_c[1], box_c[0]] = 1

                if ratio > args.upper_interval:
                    c_ratio = ratio
                    while c_ratio > args.upper_interval:
                        curr_seg = binary_erosion(curr_seg, structure=structuring_element)
                        c_ratio = np.sum(curr_seg / box_vol(box3D))
                    if c_ratio == 0:
                        box_c = box_center(box3D)
                        curr_seg[box_c[2], box_c[1], box_c[0]] = 1
        if np.sum(curr_seg) / box_vol(box3D) > 1:
            print(f'[WARNING] Segmentation bleeds out of the bounding box with a seg/bbox ratio of {np.sum(curr_seg) / box_vol(box3D)}')


        if get_model(img_npz_file) == 'th':
            segs[curr_seg > 0] = idx
        else:
            segs[segs_3d_temp>0] = idx
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

    # visualize image, mask and bounding box
    if save_overlay:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("my_model Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')
        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0: # predictions in the middle slice
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()

def get_model(img_npz_file):
    if 'PET' in img_npz_file:
        return 'th'
    if ('CT' in img_npz_file) or ('MR' in img_npz_file) or ('XRay' in img_npz_file) or ('Fundus' in img_npz_file) or ('US' in img_npz_file):
        return 'medsam'
    else: # Dermoscopy, Endoscopy, US, Microscopy, Mammography, OCT, Fundus
        return 'mobileunet'

if __name__ == '__main__':
    img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    all_files = len(img_npz_files)
    already_computed = [el for el in os.listdir(pred_save_dir) if 'npz' in el][:-1] # omit last ones
    img_npz_files = [el for el in img_npz_files if el.split('/')[-1] not in already_computed]
    print(all_files - len(img_npz_files), f'files have already been computed. Predicting for the rest of the {len(img_npz_files)} files...')
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []
    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        print('Using', get_model(img_npz_file))
        print(img_npz_file)
        if basename(img_npz_file).startswith('3D') or basename(img_npz_file).startswith('CT') or basename(img_npz_file).startswith('MR') or basename(img_npz_file).startswith('PET'):
            my_model_infer_npz_3D(img_npz_file, get_model(img_npz_file))
        else:
            my_model_infer_npz_2D(img_npz_file, get_model(img_npz_file))
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))

    print('Average Time:', np.mean(efficiency['time']))
    efficiency_df = pd.DataFrame(efficiency)
    file_path = join(pred_save_dir, 'efficiency.csv')
    if os.path.exists(file_path):
        efficiency_df.to_csv(file_path, mode='a', index=False, header=False)
    else:
       efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
