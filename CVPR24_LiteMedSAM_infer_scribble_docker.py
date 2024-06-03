from glob import glob
from os import makedirs
from os.path import join, basename
from tqdm import tqdm
import numpy as np
np.random.seed(2024)

import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='/demo_scribble/imgs/',
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='/demo_scribble/segs/',
    help='directory to save the prediction',
)
parser.add_argument(
    '-lite_medsam_checkpoint_path',
    type=str,
    default="work_dir/LiteMedSAM/medsam_lite_scribble.pth",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)

parser.add_argument(
    '--save_overlay',
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '-png_save_dir',
    type=str,
    default='.',
    help='directory to save the overlay image'
)

parser.add_argument(
    '--filter_background',
    default=True,
    action='store_false',
    help='whether to omit all the predictions outside the bbox'
)
parser.add_argument(
    '--debug_vis',
    default=False,
    action='store_true',
    help='visualize predictions to debug'
)


args = parser.parse_args()

data_root = args.input_dir
pred_save_dir = args.output_dir
save_overlay = args.save_overlay
lite_medsam_checkpoint_path = args.lite_medsam_checkpoint_path
if save_overlay:
    assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
    png_save_dir = args.png_save_dir
    makedirs(png_save_dir, exist_ok=True)
makedirs(pred_save_dir, exist_ok=True)
gt_path_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
image_size = 256

def largest_connected_component(segmentation, fill_holes=True, seed=None):
    import cv2
    from scipy.ndimage import binary_fill_holes
    import cc3d



    connected_components = cc3d.connected_components(segmentation, connectivity=26)
    max_size = 0
    largest_label = None

    for label in np.unique(connected_components)[1:]:  # Skip background label 0
        
        if seed is None:
            size = np.sum(label == connected_components)
        else:
            c_label = (label == connected_components)
            size = np.sum(c_label * seed)
        
        if size >= max_size:
            max_size = size
            largest_label = label

    largest_ccp = (((connected_components == largest_label)) * 1).astype(np.uint8)

    if fill_holes:
        largest_ccp = cv2.resize(largest_ccp, None, fx=2, fy=2, interpolation=cv2.INTER_AREA)
        largest_ccp = binary_fill_holes(largest_ccp).astype(np.uint8)
        largest_ccp = cv2.resize(largest_ccp, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    if len(segmentation.shape) == 3:
        largest_ccp = np.expand_dims(largest_ccp, axis=-1)

    return largest_ccp

def resize_longest_side(image, target_length):
    import cv2
    """
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    long_side_length = target_length
    oldh, oldw = image.shape[0], image.shape[1]
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    """
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

# %%



def show_mask(mask, ax, mask_color=None, alpha=0.5):
    if mask_color is not None:
        color = np.concatenate([mask_color, np.array([alpha])], axis=0)
    else:
        color = np.array([251/255, 252/255, 30/255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)



def medsam_inference(medsam_model, img_embed, scribble, new_size, original_size):
    #from fvcore.nn import FlopCountAnalysis

    import torch
    with torch.no_grad():

        scribble_torch = torch.as_tensor(scribble, dtype=torch.float, device=img_embed.device)


        #flops = FlopCountAnalysis(medsam_model.prompt_encoder, (None, None, scribble_torch))
        #print('flops, prompt', flops.total())
        sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
            points = None,
            boxes = None,
            masks = scribble_torch,
        )
        #flops = FlopCountAnalysis(medsam_model.mask_decoder, (img_embed, medsam_model.prompt_encoder.get_dense_pe(), sparse_embeddings, dense_embeddings, False))
        #print('flops, mask decoder', flops.total())
        low_res_logits, iou = medsam_model.mask_decoder(
            image_embeddings=img_embed, # (B, 256, 64, 64)
            image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
            multimask_output=False
        )

        low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
        low_res_pred = torch.sigmoid(low_res_pred)
        low_res_pred = low_res_pred.squeeze().cpu().numpy()
        medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou



def get_medsam():
    import torch
    from tiny_vit_sam import TinyViT
    from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer
    from utils.medsam_lite import MedSAM_Lite
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    device = torch.device(args.device)

    medsam_lite_image_encoder = TinyViT(
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
    # %%
    medsam_lite_prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16
    )

    medsam_lite_mask_decoder = MaskDecoder(
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

    # %%
    medsam_lite_model = MedSAM_Lite(
        image_encoder = medsam_lite_image_encoder,
        mask_decoder = medsam_lite_mask_decoder,
        prompt_encoder = medsam_lite_prompt_encoder
    )
    #print(f"Number of model parameters: {sum(p.numel() for p in medsam_lite_model.parameters())}")

    lite_medsam_checkpoint = torch.load(lite_medsam_checkpoint_path, map_location='cpu')
    medsam_lite_model.load_state_dict(lite_medsam_checkpoint)
    medsam_lite_model.to(device)
    medsam_lite_model.eval()
    return medsam_lite_model

def detect_circular_object(gray):
    import cv2


    # Convert the image to grayscale

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    cir_zeros = np.zeros_like(edges)
    

    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=4000)
    

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :1]:
            center = (i[0], i[1])
            # circle center
            # circle outline
            radius = i[2]
            cv2.circle(cir_zeros, center, radius, (1, 1, 1), -1)


        return cir_zeros
    else:

        return None

csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)

def preliminaries(n, x):
  """Some math that is shared across multiple algorithms."""
  assert np.all(n >= 0)
  x = np.arange(len(n), dtype=n.dtype) if x is None else x
  assert np.all(x[1:] >= x[:-1])
  w0 = clip(csum(n))
  w1 = clip(dsum(n))
  p0 = w0 / (w0 + w1)
  p1 = w1 / (w0 + w1)
  mu0 = csum(n * x) / w0
  mu1 = dsum(n * x) / w1
  d0 = csum(n * x**2) - w0 * mu0**2
  d1 = dsum(n * x**2) - w1 * mu1**2
  return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
  assert nu >= 0
  assert tau >= 0
  assert kappa >= 0
  assert omega >= 0 and omega <= 1
  x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
  v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
  v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
  f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
  f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
  return argmax(x, f0 + f1), f0 + f1


def compute_dice(pred, gt):
    
    tp = np.sum((pred == 1) * (gt == 1))
    fp = np.sum((pred == 1) * (gt == 0))
    fn = np.sum((pred == 0) * (gt == 1))

    dice = (2 * tp) / (2 * tp + fp + fn)

    return dice

def kmean(img, k=2):
    import cv2


    or_shape = img.shape
    img = img.reshape((-1, 1))
    img = np.float32(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)

    _, labels, _ = cv2.kmeans(img, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


    labels = labels.reshape(or_shape[0], or_shape[1], or_shape[2])[:,:,0]


    return labels
def detect_ridges(gray, sigma=3.0):
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals

    h_elems = hessian_matrix(gray, sigma)
    eigens = hessian_matrix_eigvals(h_elems)
    return eigens[0], eigens[1]

def mask_edge(kmeans_mask):
    if (np.sum(kmeans_mask[:,0]) / kmeans_mask.shape[0]) > 0.8:
        return True
    if (np.sum(kmeans_mask[0,:]) / kmeans_mask.shape[1]) > 0.8:
        return True
    if (np.sum(kmeans_mask[:,-1]) / kmeans_mask.shape[0]) > 0.8:
        return True
    if (np.sum(kmeans_mask[-1,:]) / kmeans_mask.shape[1]) > 0.8:
        return True
    return False



def mask_sum(kmeans_mask):

    return np.mean(kmeans_mask[:,0]) + np.mean(kmeans_mask[0,:]) + np.mean(kmeans_mask[:,-1]) + np.mean(kmeans_mask[-1,:])
# %%
def MedSAM_infer_npz(gt_path_file):
    npz_name = basename(gt_path_file)
    npz_data = np.load(gt_path_file, 'r', allow_pickle=True) # (H, W, 3)


    img_3c = npz_data['imgs'] # (H, W, 3)




    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    if 'scribbles' in npz_data.keys():
        scribble = npz_data['scribbles']
    else:
        import cc3d
        scribble = np.zeros(img_3c.shape[:2])

        components = cc3d.connected_components(npz_data['gts'])
        for c in components[1:]:
            label = components == c
            scribble += (label * c)
        #scribble = npz_data['gts']
        

    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)




    if 'Fundus' not in gt_path_file:
        grayscale, two_channels = False, False
        if 'Microscope' in gt_path_file:
            grayscale = (np.sum(img_3c[:,:,0] - img_3c[:,:,1]) == 0)
            rgb = [np.sum(img_3c[:,:,i]) for i in range(3)]
            two_channels = (len([el for el in rgb if el == 0]) == 1) 


        ## MedSAM Lite preprocessing
        img_256 = resize_longest_side(img_3c, 256)
        newh, neww = img_256.shape[:2]
        img_256_norm = (img_256 - img_256.min()) / np.clip(
            img_256.max() - img_256.min(), a_min=1e-8, a_max=None
        )
        img_256_padded = pad_image(img_256_norm, 256)

        is_medsam = 'Microscope' not in gt_path_file and 'Fundus' not in gt_path_file and 'PET' not in gt_path_file  and 'OCT' not in gt_path_file
        if is_medsam:
            import torch
            device = torch.device(args.device)

            medsam_lite_model = get_medsam()
            img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = medsam_lite_model.image_encoder(img_256_tensor)


        label_ids = np.unique(scribble[(scribble != 0) & (scribble != 1000)])
        scribbles_list = []
        for label_id in label_ids:            



            if is_medsam:
                scribble_input = np.uint8(scribble == label_id)
                scribble_input = pad_image(resize_longest_side(scribble_input[...,np.newaxis],256)[:, :, None], 256)
                scribble_input = torch.from_numpy(scribble_input).permute(2,0,1)[None,]
                scribble_input = (scribble_input > 0) * 1
                medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, scribble_input, (newh, neww), (H, W))
                medsam_mask = largest_connected_component(medsam_mask).astype(np.uint8)   

            else:
                medsam_mask = np.zeros_like(segs).astype(np.uint16)

            if 'OCT' in gt_path_file or 'PET' in gt_path_file:
                raw_scribble = (scribble == label_id)
                scribble_indices = np.nonzero(raw_scribble)
                ys = [el for el in scribble_indices[0]]
                xs = [el for el in scribble_indices[1]]
                context = 50

                xmin, ymin, xmax, ymax = (np.min(xs)-context), (np.min(ys)-context), (np.max(xs) + context), (np.max(ys) + context)
                xmin = max(0, xmin)
                ymin = max(0, ymin)

                img_gs = img_3c[:,:,0][ymin:ymax,xmin:xmax]
                vals = np.array(img_gs.flatten())
                vals = np.array(vals[vals > 0])


                if 'OCT' in gt_path_file:
                    bins = 10
                else:
                    bins = 100
                n, x = np.histogram((vals).flatten(), bins=bins)

                step_size = (x[1] - x[0]) / 2
                x = np.array([el - step_size for el in x[1:]])

                new_th = GHT(n, x=x, nu=1e60, tau=1e-30)[0] + 30
                
                my_model_mask = np.zeros((img_3c.shape[0], img_3c.shape[1])).astype(np.uint8)
                if 'OCT' in gt_path_file:
                    my_model_mask = (img_gs < new_th).astype(np.uint8)
                else:
                    my_model_mask = (img_gs > new_th).astype(np.uint8)

                my_model_mask = largest_connected_component(my_model_mask, seed=raw_scribble[ymin:ymax,xmin:xmax]).astype(np.uint8)   
                medsam_mask[ymin:ymax,xmin:xmax] = my_model_mask
            elif 'Microscope' in gt_path_file:
                raw_scribble = (scribble == label_id)
                scribble_indices = np.nonzero(raw_scribble)
                ys = [el for el in scribble_indices[0]]
                xs = [el for el in scribble_indices[1]]
                context = 50

                xmin, ymin, xmax, ymax = (np.min(xs)-context), (np.min(ys)-context), (np.max(xs) + context), (np.max(ys) + context)
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                if (grayscale or two_channels):
                    high_contrast = False
                    img_crop = img_3c[ymin:ymax,xmin:xmax]
                    if two_channels:
                        img_gs = np.mean(img_3c, axis=-1)
                        simple_th = np.percentile(img_gs, 75)
                        temp_pred = (np.mean(img_crop, axis=-1) > simple_th).astype(np.uint8)
                        high_contrast = True
                    if not high_contrast:
                        circles = detect_circular_object(img_crop)

                        if circles is not None and grayscale:
                            temp_pred = circles
                        else:
                            temp_pred = kmean(img_crop)
                            scribble_crop = raw_scribble[ymin:ymax,xmin:xmax]
                            majority = np.mean(temp_pred[scribble_crop > 0])
                            if majority < 0.5:
                                temp_pred = np.logical_not(temp_pred).astype(np.uint8)
                            temp_pred = largest_connected_component(temp_pred, seed=scribble_crop)
                            if mask_edge(temp_pred):
                                temp_pred = np.logical_not(temp_pred).astype(np.uint8)    
                    medsam_mask = np.zeros_like(segs).astype(np.uint16)
                    medsam_mask[ymin:ymax,xmin:xmax] = temp_pred            
                else:
                    import GeodisTK
                    import cv2
                    img_crop = img_3c[ymin:ymax,xmin:xmax]
                    img_gs = np.mean(img_crop, axis=-1)
                    scribble_crop = raw_scribble[ymin:ymax,xmin:xmax]

                    D1 = GeodisTK.geodesic2d_raster_scan(img_crop.astype(np.float32), scribble_crop, 0.5, 2)

                    D1 = (D1 - np.min(D1)) / (np.max(D1) - np.min(D1))



                    temp_pred = (D1 < np.percentile(D1, 70)).astype(np.uint8)
   
                    medsam_mask = np.zeros_like(segs).astype(np.uint16)
                    medsam_mask[ymin:ymax,xmin:xmax] = temp_pred      

                


            segs[medsam_mask>0] = label_id
            scribbles_list.append(scribble)

    elif 'Fundus' in gt_path_file:
        import cv2
        from skimage.filters import meijering

        gray = np.mean(img_3c, axis=-1).astype(np.uint8)

        ridges = meijering(gray)



        filter_th = np.percentile((ridges[ridges > 0]), 95)
        ridges *= (ridges < filter_th)

        second_th = np.percentile((ridges[ridges > 0]), 90)
        ridges = (ridges > second_th).astype(np.uint8) * 255

        or_size = ridges.shape
        ridges = cv2.resize(ridges.astype(np.uint8), (512, 512))
        size_filter = int(ridges.shape[1] / 20)

        import cc3d
        components = cc3d.connected_components(ridges, connectivity=8)

        new_ridges = np.zeros_like(ridges)
        
        for label in np.unique(components)[1:]:
            labeled_mask = components == label
            if not np.sum(labeled_mask) < size_filter:
                new_ridges += labeled_mask

        ridges = new_ridges 

        cir_zeros = np.zeros_like(ridges).astype(np.uint8)
        erode = int(cir_zeros.shape[1] * 0.019) # remove large circle
        cv2.circle(cir_zeros, (int(cir_zeros.shape[0]/2), int(cir_zeros.shape[1]/2)), int(cir_zeros.shape[1]/2) - erode, (1, 1, 1), -1)
        ridges = ridges * cir_zeros

        ridges = cv2.resize(ridges, or_size)

        segs = ridges 

    

   

    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs
    )

    if args.debug_vis:
        import cv2

        if 'Fundus' in gt_path_file:
            img_3c = cv2.cvtColor(img_3c, cv2.COLOR_RGB2BGR)
        cv2.imshow('img', cv2.resize(img_3c, (256, 256), interpolation=cv2.INTER_AREA))
        cv2.imshow('test', cv2.resize(((segs>0) * 255).astype(np.uint8), (256,256), interpolation=cv2.INTER_AREA))

        #cv2.imshow('scribbles', cv2.resize(((scribble) * (255)).astype(np.uint8), (256,256), interpolation=cv2.INTER_AREA))

        cv2.waitKey(0)

    # visualize image, mask and bounding box
    if save_overlay:
        from matplotlib import pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(14, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(segs)
        ax[2].imshow(segs)
        ax[0].set_title("Image")
        ax[1].set_title("Scribbled Image")
        ax[2].set_title(f"Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')

        for i, label_id in enumerate(label_ids):
            color = np.random.rand(3)
            show_mask((scribbles_list[i]==label_id).astype(np.uint8), ax[1], mask_color=color)
            show_mask((segs == label_id).astype(np.uint8), ax[2], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'))#, dpi=300)
        plt.close()

if __name__ == '__main__':
    all_files = len(gt_path_files)
    for gt_path_file in tqdm(gt_path_files):
        print(gt_path_file)
        MedSAM_infer_npz(gt_path_file)

