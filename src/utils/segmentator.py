import torch
import os

import highdicom as hd
import numpy as np
from pydicom.sr import codes

from .monai_helper import (
    prepare_preprocessing, prepare_dataloader, prepare_inferer,
    prepare_postprocessing, prepare_network
)

from .datasets import reorient_to_dicom

SEGMENTATION_LABELS = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5",
    19: "vertebrae_L4",
    20: "vertebrae_L3",
    21: "vertebrae_L2",
    22: "vertebrae_L1",
    23: "vertebrae_T12",
    24: "vertebrae_T11",
    25: "vertebrae_T10",
    26: "vertebrae_T9",
    27: "vertebrae_T8",
    28: "vertebrae_T7",
    29: "vertebrae_T6",
    30: "vertebrae_T5",
    31: "vertebrae_T4",
    32: "vertebrae_T3",
    33: "vertebrae_T2",
    34: "vertebrae_T1",
    35: "vertebrae_C7",
    36: "vertebrae_C6",
    37: "vertebrae_C5",
    38: "vertebrae_C4",
    39: "vertebrae_C3",
    40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus",
    43: "trachea",
    44: "heart_myocardium",
    45: "heart_atrium_left",
    46: "heart_ventricle_left",
    47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery",
    50: "brain",
    51: "iliac_artery_left",
    52: "iliac_artery_right",
    53: "iliac_vena_left",
    54: "iliac_vena_right",
    55: "small_bowel",
    56: "duodenum",
    57: "colon",
    58: "rib_left_1",
    59: "rib_left_2",
    60: "rib_left_3",
    61: "rib_left_4",
    62: "rib_left_5",
    63: "rib_left_6",
    64: "rib_left_7",
    65: "rib_left_8",
    66: "rib_left_9",
    67: "rib_left_10",
    68: "rib_left_11",
    69: "rib_left_12",
    70: "rib_right_1",
    71: "rib_right_2",
    72: "rib_right_3",
    73: "rib_right_4",
    74: "rib_right_5",
    75: "rib_right_6",
    76: "rib_right_7",
    77: "rib_right_8",
    78: "rib_right_9",
    79: "rib_right_10",
    80: "rib_right_11",
    81: "rib_right_12",
    82: "humerus_left",
    83: "humerus_right",
    84: "scapula_left",
    85: "scapula_right",
    86: "clavicula_left",
    87: "clavicula_right",
    88: "femur_left",
    89: "femur_right",
    90: "hip_left",
    91: "hip_right",
    92: "sacrum",
    93: "face",
    94: "gluteus_maximus_left",
    95: "gluteus_maximus_right",
    96: "gluteus_medius_left",
    97: "gluteus_medius_right",
    98: "gluteus_minimus_left",
    99: "gluteus_minimus_right",
    100: "autochthon_left",
    101: "autochthon_right",
    102: "iliopsoas_left",
    103: "iliopsoas_right",
    104: "urinary_bladder"
}

def run_pipeline(model_path, data_path):
    preprocessing = prepare_preprocessing()

    dataloader = prepare_dataloader(data_path, preprocessing)

    inferer = prepare_inferer()

    postprocessing = prepare_postprocessing(preprocessing)

    model = prepare_network(model_path)

    data = preprocessing({"image": data_path})

    with torch.no_grad():
        data['pred'] = inferer(inputs=data["image"].unsqueeze(0), network=model)

    data['pred'] = data['pred'][0]
    data['image'] = data['image'][0]

    data = postprocessing(data)

    return data['pred'], data['image']

def get_segmentations(pred, ref_datasets, num_segments=10):
    algorithm_identification = hd.AlgorithmIdentificationSequence(
        name='MONAI Whole Body CT Segmentation',
        version='0.1.0',
        family=codes.cid7162.ArtificialIntelligence
    )

    segment_list = []
    segment_description_list = []

    labels, counts = np.unique(pred, return_counts=True)
    sorted_counts = sorted(zip(labels, counts), key=lambda x: x[1], reverse=True)

    top_labels = [val for val, _ in sorted_counts[:num_segments] if val != 0]

    for i in top_labels:
        segment_list.append((pred == i).astype(np.uint8))
        
        description = hd.seg.SegmentDescription(
            segment_number=len(segment_list),
            segment_label=SEGMENTATION_LABELS[i],
            segmented_property_category=codes.cid7150.Tissue,
            segmented_property_type=codes.cid7166.ConnectiveTissue,
            algorithm_type=hd.seg.SegmentAlgorithmTypeValues.AUTOMATIC,
            algorithm_identification=algorithm_identification,
            tracking_uid=hd.UID(),
            tracking_id=f"MONAI_{SEGMENTATION_LABELS[i]}"
        )

        segment_description_list.append(description)

    if not segment_list:
        raise ValueError("No segments found in the prediction.")
    

    segments = np.stack(segment_list, axis=-1)

    seg = hd.seg.Segmentation(
        source_images=ref_datasets,
        pixel_array=segments,
        segmentation_type=hd.seg.SegmentationTypeValues.BINARY,
        segment_descriptions=segment_description_list,
        series_instance_uid=hd.UID(),
        series_number=2,
        sop_instance_uid=hd.UID(),
        instance_number=1,
        manufacturer="MONAI",
        manufacturer_model_name="Whole Body CT Segmentation",
        software_versions=["0.1.0"],
        device_serial_number="12345",
        series_description="Whole Body CT Segmentation",
    )

    return seg

def save_segmentation(seg, output_path):
    seg.save_as(os.path.join(output_path, "segmentation.dcm"))
    print(f"Segmentation saved to {output_path}")

def segment_and_save(data_path, model_path, ref_datasets, logger, seg_num):
    try:
        logger.log("Starting segmentation pipeline...")
        pred, ref_image = run_pipeline(model_path, data_path)
        logger.log("Pipeline finished, processing results...")
        pred_reoriented = reorient_to_dicom(pred)
        seg = get_segmentations(pred_reoriented, ref_datasets, seg_num)

        logger.log("Saving segmentation...")
        save_segmentation(seg, data_path)

        logger.log("Segmentation process completed.")
    except Exception as e:
        logger.log(f"Error during segmentation: {e}")