import argparse
import numpy as np
import cv2 
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-data_root", type=str, default=None,
    help="Path to the npy data root."
)

args = parser.parse_args()

if args.data_root is not None:
    npz_files = [os.path.join(args.data_root, el) for el in os.listdir(args.data_root) if '.npz' in el]
    for f in npz_files:
        x = np.load(f)['segs'].astype(np.uint8) * 255
        if len(x.shape) == 2:
            cv2.imwrite(f.replace('npz', 'png'), x)
print('Done')