import numpy as np
import nibabel as nib
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input_npz',
    type=str,
    help='path to npz file',
)
parser.add_argument(
    '-o',
    '--output_nii',
    type=str,
    help='path to nifti file',
)
args = parser.parse_args()
seg = np.load(args.input_npz)['imgs'].astype(np.uint8) * 255
affine = np.eye(4)
affine[0][0] = -1
ni_img = nib.Nifti1Image(seg, affine=affine)
ni_img.header.get_xyzt_units()

ni_img.to_filename(args.output_nii) 
print('Done')