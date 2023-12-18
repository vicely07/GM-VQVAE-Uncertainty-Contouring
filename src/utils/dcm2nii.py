"""
Medical Image Data Conversion Script

This script facilitates the conversion of medical imaging data from DICOM to NIfTI format and from NIfTI to NumPy format. It provides a flexible command-line interface for specifying input and output paths, making it convenient for batch processing of subject data.

The script consists of two main functions:
1. `dcm2nii`: Converts DICOM files to NIfTI format, with the option to specify the output folder for NIfTI files. The resulting NIfTI files are saved with a naming convention that includes the original file name and an additional identifier.

2. `nii2npy`: Converts NIfTI files to NumPy format, providing the flexibility to choose the output folder for NumPy files. The resulting NumPy files are saved with the original file name.

To use the script, simply provide the input path containing DICOM or NIfTI files and an optional output path for saving the converted files.
"""

import os
from dicom_utils import dcm2npy
import argparse
import nibabel as nib
import numpy as np
from typing import Optional

def get_conversion_args() -> argparse.Namespace:
    """
    Parse command line arguments for data conversion.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Use this script for data conversion.",
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

def dcm2nii(input_path: str, output_path: str) -> None:
    """
    Convert DICOM files to NIfTI format.

    Args:
        input_path (str): Path to the input DICOM data.
        output_path (str): Path to the output folder for NIfTI files.
    """
    data, _, _, _, _ = dcm2npy(input_path)
    data_nii = nib.Nifti1Image(np.rot90(data, k=2).transpose((2, 1, 0)), affine=np.eye(4))
    file_name = os.path.split(input_path)[-1]
    nifti_filename = os.path.join(output_path, f"{file_name}_0000.nii.gz")
    print(nifti_filename)
    print(data_nii.shape)
    nib.save(data_nii, nifti_filename)

def nii2npy(input_path: str, output_path: str) -> None:
    """
    Convert NIfTI files to NumPy format.

    Args:
        input_path (str): Path to the input NIfTI data.
        output_path (str): Path to the output folder for NumPy files.
    """
    data = nib.load(input_path).get_data()
    file_name = os.path.split(input_path)[-1]
    npy_filename = os.path.join(output_path, f"{file_name}.npy")
    np.save(npy_filename, data)

# Load input args
args = get_conversion_args()
input_path, output_path = args.input_path, args.output_path

# Process each subject
subject_list = [x[1] for x in os.walk(input_path)][0]
print(subject_list)
for subject in subject_list:
    try:
        dcm2nii(os.path.join(input_path, subject), output_path)
    except Exception as e:
        print(f"Error processing {subject}: {str(e)}")
        pass
