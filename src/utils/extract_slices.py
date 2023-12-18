"""
NIfTI Image Extraction Utility

This script is designed for extracting 2D slices from 3D segmentation masks stored in NIfTI (.nii.gz) format.

Author: [Your Name]
Date: [Current Date]
"""

import os
import glob
import argparse
import nibabel as nib
import numpy as np
from typing import Tuple

def get_conversion_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
            description="Use this script for slice extraction from 3D segmentation masks",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
            "--input_path",
            type=str,
            default="datasets/prostate_data.npy",
            help="Filepath to test data",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Filepath to output folder",
    )
    return parser.parse_args()

def extract_slices(input_path: str, output_path: str) -> None:
    """
    Extracts 2D slices from 3D segmentation masks.

    Args:
        input_path (str): Path to the input data.
        output_path (str): Path to the output folder.
    """
    print(input_path)
    data_filepaths = glob.glob(os.path.join(input_path, '*.nii.gz'))
    print(data_filepaths)
    for filepath in data_filepaths:
        mask = nib.load(filepath).get_fdata()
        for i in range(mask.shape[0]):
            filename =  filepath.split("/")[-1].split(".")[0]
            print(filename)
            np.save(os.path.join(output_path, filename + "_" + str(i)), mask[i, ...])

# Load input args
if __name__ == "__main__":
    args


# python extract_slices --input_path ../../data/seg/3d_masks/Prostate --output_mask ../../data/seg/2d_slices
