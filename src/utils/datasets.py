import pydicom
import os
import numpy as np

from monai.transforms import Orientation

def load_dataset_with_pydicom(path):
    def get_instance_number(fp):
        return pydicom.dcmread(fp).InstanceNumber

    try:
        ref_dicom_files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith('.dcm')
        ]
    except Exception as e:
        raise ValueError(f"Error reading DICOM files from {path}: {e}")

    ref_dicom_files = sorted(ref_dicom_files, key=get_instance_number)
    ref_datasets = [fill_default_fields(pydicom.dcmread(fp)) for fp in ref_dicom_files]

    return ref_datasets

def fill_default_fields(src_img):
    fields = [
        ("StudyInstanceUID", ""),
        ("SeriesInstanceUID", ""),
        ("SeriesNumber", 1),
        ("SOPInstanceUID", ""),
        ("InstanceNumber", 1),
        ("SOPClassUID", ""),
        ("Manufacturer", ""),
        ("Modality", "SEG"),
        ("TransferSyntaxUID", ""),
        ("PatientID", ""),
        ("PatientName", ""),
        ("PatientBirthDate", ""),
        ("PatientSex", ""),
        ("AccessionNumber", ""),
        ("StudyID", ""),
        ("StudyDate", ""),
        ("StudyTime", ""),
    ]

    for attr, default in fields:
        if not hasattr(src_img, attr):
            setattr(src_img, attr, default)

    return src_img

def reorient_to_dicom(img):
    reorient = Orientation(axcodes="LPS")

    CT_reoriented = reorient(img)[0]

    CT_reoriented = np.flip(np.moveaxis(CT_reoriented, [0, 1, 2], [2, 1, 0]), axis=0)

    return CT_reoriented
