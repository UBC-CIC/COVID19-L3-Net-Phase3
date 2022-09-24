import argparse
import numpy as np
import os
from src.utilities.infer_utils import save_nii


def convert_to_nii(inference_fp, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    inference = np.load(inference_fp)

    ct = inference['in_']
    outputs = inference['out_']
    slice_thickness = inference['slice_thickness']

    save_nii(output_dir, ct, outputs, slice_thickness)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inference_fp', type=str, default='None', help="Filepath to inference.npz")
    parser.add_argument('--output_dir', type=str, default='None', help="If None, will save to same directory as inference file")
    args = parser.parse_args()

    if args.output_dir == "None" or len(args.output_dir) == 0:
        args.output_dir = os.path.split(args.inference_fp)[0]
    
    convert_to_nii(args.inference_fp, args.output_dir)

