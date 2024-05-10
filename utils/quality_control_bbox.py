import argparse
import numpy as np
import cv2 
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "-pred_root", type=str, default=None,
    help="Path to the predictions."
)
parser.add_argument(
    "-box_root", type=str, default=None,
    help="Path to the boxes."
)

parser.add_argument(
    "-modality", type=str, default=None,
    help="Imaging modality."
)

args = parser.parse_args()

if args.pred_root is not None:
    npz_files = [os.path.join(args.pred_root, el) for el in os.listdir(args.pred_root) if '.npz' in el and args.modality in el]
    for f in npz_files:
        x = np.load(f)['segs'].astype(np.uint16)
        boxes = np.load(f.replace(args.pred_root, args.box_root))['boxes']
        imgs = np.load(f.replace(args.pred_root, args.box_root))['imgs']
        for i, box in enumerate(boxes):
                pred = ((x == (i+1)) * 255).astype(np.uint8)
                pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
                x_min, y_min, width, height = box
                cv2.rectangle(pred, (x_min, y_min), (width, height), (0, 255, 0), 5)
                pred = cv2.resize(pred, (256, 256))
                cv2.imshow('pred', pred)
                imgs = cv2.resize(imgs, (256, 256))
                cv2.imshow('img', imgs)
                
                cv2.waitKey(0)

print('Done')