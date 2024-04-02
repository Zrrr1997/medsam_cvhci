# %% move files based on csv file
import numpy as np
from os import listdir
from os.path import join
from collections import OrderedDict
import pandas as pd
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient
from tqdm import tqdm
import multiprocessing as mp
import argparse
import cc3d


def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)

def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in range(1, gt.max()+1):
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(
            gt_i, seg_i, spacing_mm=spacing
        )
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    print(np.mean(nsd))
    return np.mean(nsd)


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--seg_dir', default='test_demo/segs', type=str)
parser.add_argument('-g', '--gt_dir', default='test_demo/gts', type=str)
parser.add_argument('-csv_dir', default='test_demo/metrics.csv', type=str)
parser.add_argument('-num_workers', type=int, default=5)
parser.add_argument('-nsd', default=True, type=bool, help='set it to False to disable NSD computation and save time')
args = parser.parse_args()

seg_dir = args.seg_dir
gt_dir = args.gt_dir
csv_dir = args.csv_dir
num_workers = args.num_workers
compute_NSD = args.nsd

def compute_metrics(npz_name):
    metric_dict = {'dsc': -1.}
    if compute_NSD:
        metric_dict['nsd'] = -1.
    npz_seg = np.load(join(seg_dir, npz_name), allow_pickle=True, mmap_mode='r')
    npz_gt = np.load(join(gt_dir, npz_name), allow_pickle=True, mmap_mode='r')
    gts = npz_gt['gts']
    #if 255 in gts:
    gts = cc3d.connected_components((gts > 0) * 1, connectivity=26) # solves the weird GTs with [0, 2] labels


    segs = npz_seg['segs']

    if npz_name.startswith('3D') or npz_name.startswith('CT') or npz_name.startswith('MR') or npz_name.startswith('PET'):
        spacing = npz_gt['spacing']
    dsc = compute_multi_class_dsc(gts, segs)
    # comupute nsd
    if compute_NSD:
        if dsc > 0.2:
        # only compute nsd when dice > 0.2 because NSD is also low when dice is too low
            if npz_name.startswith('3D') or npz_name.startswith('CT') or npz_name.startswith('MR') or npz_name.startswith('PET'):
                nsd = compute_multi_class_nsd(gts, segs, spacing)
            else:
                spacing = [1.0, 1.0, 1.0]
                nsd = compute_multi_class_nsd(np.expand_dims(gts, -1), np.expand_dims(segs, -1), spacing)
        else:
            nsd = 10000.0 # Assume model performs poor on this sample
    return npz_name, dsc, nsd

if __name__ == '__main__':
    seg_metrics = OrderedDict()
    seg_metrics['case'] = []
    seg_metrics['dsc'] = []
    if compute_NSD:
        seg_metrics['nsd'] = []
    
    npz_names_seg = listdir(seg_dir)
    npz_names = listdir(gt_dir)


    npz_names = [npz_name for npz_name in npz_names if npz_name.endswith('.npz')]
    npz_names_seg = [npz_name for npz_name in npz_names_seg if npz_name.endswith('.npz')]

    if len(npz_names_seg) < len(npz_names):
        print(f'[WARNING] Prediction dir {seg_dir} is incomplete with {len(npz_names_seg)} predictions from {len(npz_names)} labels...')
        npz_names = npz_names_seg

    with mp.Pool(num_workers) as pool:
        with tqdm(total=len(npz_names)) as pbar:
            for i, (npz_name, dsc, nsd) in enumerate(pool.imap_unordered(compute_metrics, npz_names)):
                seg_metrics['case'].append(npz_name)
                seg_metrics['dsc'].append(np.round(dsc, 4))
                if compute_NSD:
                    seg_metrics['nsd'].append(np.round(nsd, 4))
                pbar.update()
    print("Mean Dice", np.mean(seg_metrics['dsc']))
    df = pd.DataFrame(seg_metrics)
    # rank based on case column
    df = df.sort_values(by=['case'])
    df.to_csv(csv_dir, index=False)