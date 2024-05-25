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
    npz_files = [os.path.join(args.data_root, el) for el in os.listdir(args.data_root) if '.npz' in el and 'US' in el]
    for f in npz_files:
        x = np.load(f)['segs']
        #if 'Fundus' in f:
        #    y = (x['segs'] > 0).astype(np.uint8)
        #    np.savez_compressed(f, segs=y)
        ##print(np.sum(xx))
        #print(xx.shape, f)
        cv2.imwrite(f.replace('npz', 'png'), (x  > 0).astype(np.uint8)* 255 )
print('Done')